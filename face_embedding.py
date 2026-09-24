import os
import base64
import urllib.request

import cv2
import numpy as np


# =====================================================
# FACE EMBEDDINGS (YuNet detector + SFace recognizer)
#
# LBPH distances cannot tell two people apart reliably:
# the same student scored 28-73 against their own
# captures while a different student scored 53-77.
#
# SFace turns a face into a 128-number fingerprint.
# Cosine similarity between two fingerprints is high
# (0.6+) for the same person and low (<0.3) for
# different people, so a single threshold works.
# =====================================================

MODELS_DIR = "models"

YUNET_PATH = os.path.join(
    MODELS_DIR,
    "face_detection_yunet_2023mar.onnx"
)

SFACE_PATH = os.path.join(
    MODELS_DIR,
    "face_recognition_sface_2021dec.onnx"
)

MODEL_URLS = {
    YUNET_PATH:
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",

    SFACE_PATH:
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

# OpenCV's published cosine threshold for SFace.
# At or above = same person.
SAME_PERSON_THRESHOLD = float(
    os.environ.get("FACE_MATCH_THRESHOLD", 0.363)
)

# Tight 200x200 crops (stored enrollment images) give
# YuNet no context around the face. Padding fixes that.
CROP_PADDING = 60

# YuNet misses faces larger than ~600px, which is every
# full-resolution phone selfie. Detection runs on a copy
# scaled down to this longest side. SFace only needs a
# 112x112 face, so nothing is lost.
MAX_DETECT_SIDE = 640

# Some Android phones save the photo sideways without an
# EXIF orientation tag. If the upright image has no face,
# these rotations are tried in order.
FALLBACK_ROTATIONS = [
    cv2.ROTATE_90_CLOCKWISE,
    cv2.ROTATE_90_COUNTERCLOCKWISE,
    cv2.ROTATE_180,
]

_detector = None
_recognizer = None


# =====================================================
# DOWNLOAD MODELS
#
# Run at Docker build time so production never
# downloads at request time.
# =====================================================
def download_models():

    os.makedirs(MODELS_DIR, exist_ok=True)

    for path, url in MODEL_URLS.items():

        if os.path.exists(path):
            continue

        print(f"[MODEL DOWNLOAD] {url}", flush=True)

        urllib.request.urlretrieve(url, path)


def _load():

    global _detector, _recognizer

    if _recognizer is not None:
        return

    download_models()

    _detector = cv2.FaceDetectorYN.create(
        YUNET_PATH,
        "",
        (320, 320),
        0.6
    )

    _recognizer = cv2.FaceRecognizerSF.create(
        SFACE_PATH,
        ""
    )


# =====================================================
# DECODE IMAGE (colour)
# =====================================================
def decode_color(image_data):

    if "," in image_data:
        image_data = image_data.split(",")[1]

    nparr = np.frombuffer(
        base64.b64decode(image_data),
        np.uint8
    )

    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# =====================================================
# EMBED
#
# Returns a 128-float list for the largest face,
# or None if no face was found.
# =====================================================
def embed(img, padded=False):

    embedding, _ = detect_and_embed(img, padded)

    return embedding


# =====================================================
# DETECT AND EMBED
#
# Same as embed(), plus the face crop taken from the
# full-resolution (and correctly rotated) image, for
# quality checks such as blur.
# Returns (None, None) if no face was found.
# =====================================================
def detect_and_embed(img, padded=False):

    _load()

    if img is None:
        return None, None

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if padded:
        img = cv2.copyMakeBorder(
            img,
            CROP_PADDING, CROP_PADDING,
            CROP_PADDING, CROP_PADDING,
            cv2.BORDER_REPLICATE
        )

    embedding, face_crop, _ = _embed_upright(img)

    if embedding is not None:
        return embedding, face_crop

    # A sideways photo can yield a plausible "upright"
    # face both at the correct rotation and upside down
    # (mouth read as eyes). Try every rotation and keep
    # the one the detector is most confident in.
    best = (None, None, -1.0)

    for rotation in FALLBACK_ROTATIONS:

        result = _embed_upright(cv2.rotate(img, rotation))

        if result[0] is not None and result[2] > best[2]:
            best = result

    return best[0], best[1]


# Returns (embedding, face_crop, detection_score),
# or (None, None, 0.0) when no upright face is found.
def _embed_upright(img):

    scale = min(
        1.0,
        MAX_DETECT_SIDE / float(max(img.shape[:2]))
    )

    small = img if scale == 1.0 else cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    _detector.setInputSize(
        (small.shape[1], small.shape[0])
    )

    _, faces = _detector.detect(small)

    if faces is None or len(faces) == 0:
        return None, None, 0.0

    # Largest face
    face = max(faces, key=lambda f: f[2] * f[3])

    # YuNet also fires on sideways faces, but with
    # unreliable landmarks that produce a wrong
    # embedding. Reject so the next rotation is tried.
    if not _is_upright(face):
        return None, None, 0.0

    aligned = _recognizer.alignCrop(small, face)

    feature = _recognizer.feature(aligned)

    # Map the box back to full resolution
    x, y, w, h = (
        int(round(v / scale)) for v in face[:4]
    )

    x, y = max(x, 0), max(y, 0)

    face_crop = img[y:y + h, x:x + w]

    return (
        feature.flatten().astype(float).tolist(),
        face_crop,
        float(face[14])
    )


# =====================================================
# IS UPRIGHT
#
# YuNet row: x, y, w, h, then landmarks
#   right eye (4,5)  left eye (6,7)  nose (8,9)
#   right mouth (10,11)  left mouth (12,13)
#
# Upright = eyes roughly level AND above the mouth.
# =====================================================
def _is_upright(face):

    eye_dx = face[6] - face[4]
    eye_dy = face[7] - face[5]

    eyes_y = (face[5] + face[7]) / 2.0
    mouth_y = (face[11] + face[13]) / 2.0

    return abs(eye_dx) > abs(eye_dy) and eyes_y < mouth_y


# =====================================================
# SIMILARITY
#
# Cosine similarity, -1..1. HIGHER = more similar.
# =====================================================
def similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)
