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
# Same as embed(), plus the face box (x, y, w, h) in
# the input image's coordinates, for quality checks.
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

    _detector.setInputSize(
        (img.shape[1], img.shape[0])
    )

    _, faces = _detector.detect(img)

    if faces is None or len(faces) == 0:
        return None, None

    # Largest face
    face = max(faces, key=lambda f: f[2] * f[3])

    aligned = _recognizer.alignCrop(img, face)

    feature = _recognizer.feature(aligned)

    offset = CROP_PADDING if padded else 0

    box = tuple(
        int(v) for v in (
            face[0] - offset,
            face[1] - offset,
            face[2],
            face[3]
        )
    )

    return feature.flatten().astype(float).tolist(), box


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
