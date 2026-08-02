import cv2
import base64
import numpy as np


# =====================================================
# VALIDATION THRESHOLDS
#
# All tuning lives here. Nothing below reads a magic
# number directly.
# =====================================================

CONFIG = {

    # Face must be at least this fraction of the frame width
    "min_face_ratio": 0.18,

    # ...and at least this many pixels wide
    "min_face_px": 90,

    # Laplacian variance. Lower = blurrier.
    # Measured BEFORE Gaussian smoothing.
    "min_blur": 60.0,

    # Mean grey level of the RAW crop, 0-255.
    # Measured BEFORE equalizeHist.
    "min_brightness": 45.0,
    "max_brightness": 225.0,

    # How far the face centre may sit from the frame
    # centre, as a fraction of frame width / height
    "max_offset_x": 0.20,
    "max_offset_y": 0.22,

    # Output size fed to LBPH
    "face_size": (200, 200),
}


# =====================================================
# FACE DETECTOR
#
# Built once at import instead of per-request.
# =====================================================

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_alt2.xml"
)


# =====================================================
# DECODE IMAGE
#
# Accepts a data URL or a bare base64 string.
# Returns RAW greyscale - no equalisation, because
# brightness has to be measured before normalisation.
# =====================================================
def decode_image(image_data):

    try:

        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)

        nparr = np.frombuffer(
            img_bytes,
            np.uint8
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            return None, "Image could not be read."

        # Front camera preview is mirrored
        img = cv2.flip(img, 1)

        return img, None

    except Exception as e:

        return None, f"Image could not be decoded. {str(e)}"


# =====================================================
# DETECT FACE
#
# Detection runs on an equalised COPY - equalisation
# helps Haar, but the raw image is kept for the
# brightness measurement.
#
# Returns exactly one bounding box, or an error.
# =====================================================
def detect_face(gray):

    detect_src = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        detect_src,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None, "no face detected. Position your face inside the guide."

    if len(faces) > 1:
        return None, "Multiple faces found. Only one person should appear."

    # Largest face, in case a future threshold change
    # allows more than one through
    bbox = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    return bbox, None


# =====================================================
# CHECK FACE SIZE
# =====================================================
def check_size(bbox, frame_shape):

    (x, y, w, h) = bbox

    frame_h, frame_w = frame_shape[:2]

    ratio = w / float(frame_w)

    if w < CONFIG["min_face_px"] or ratio < CONFIG["min_face_ratio"]:
        return False, "Face is too small. Move a little closer.", ratio

    return True, None, ratio


# =====================================================
# CHECK CENTER
#
# Compares the face centre against the frame centre.
# =====================================================
def check_center(bbox, frame_shape):

    (x, y, w, h) = bbox

    frame_h, frame_w = frame_shape[:2]

    center_x = x + w / 2.0
    center_y = y + h / 2.0

    offset_x = abs(center_x - frame_w / 2.0) / frame_w
    offset_y = abs(center_y - frame_h / 2.0) / frame_h

    if offset_x > CONFIG["max_offset_x"] or offset_y > CONFIG["max_offset_y"]:
        return False, "Move your face to the center.", (offset_x, offset_y)

    return True, None, (offset_x, offset_y)


# =====================================================
# CHECK BRIGHTNESS
#
# MUST run on the raw crop. equalizeHist normalises
# the tonal range, so a dark frame and a bright one
# both average near 127 afterwards.
# =====================================================
def check_brightness(raw_face):

    brightness = float(np.mean(raw_face))

    if brightness < CONFIG["min_brightness"]:
        return False, "Lighting is too low. Move somewhere brighter.", brightness

    if brightness > CONFIG["max_brightness"]:
        return False, "Too much light. Move away from direct glare.", brightness

    return True, None, brightness


# =====================================================
# CHECK BLUR
#
# MUST run before GaussianBlur, otherwise this scores
# our own smoothing filter rather than the capture.
# =====================================================
def check_blur(face):

    blur_value = float(
        cv2.Laplacian(face, cv2.CV_64F).var()
    )

    if blur_value < CONFIG["min_blur"]:
        return False, "Image is too blurry. Hold the phone steady.", blur_value

    return True, None, blur_value


# =====================================================
# VALIDATE POSE  --  PLACEHOLDER
#
# Haar cascades report a bounding box only, with no
# head orientation, so real pose validation is not
# possible here yet. This returns OK for every pose.
#
# To implement later, pick one:
#
#   - MediaPipe FaceMesh landmarks + cv2.solvePnP
#     for true yaw / pitch / roll
#   - dlib 68-point landmarks + solvePnP
#   - haarcascade_profileface.xml fired against the
#     image and its mirror (cheap, but noisy)
#
# Then compare yaw against the expected pose:
#
#   front : abs(yaw) <= 14
#   left  : yaw between -48 and -12
#   right : yaw between  12 and  48
#
# Note the mirror flip in decode_image() - confirm
# which way "left" reads before trusting the sign.
# =====================================================
def validate_pose(face, expected_pose):

    return True, None, None


# =====================================================
# PREPARE FACE
#
# Crop -> resize -> equalise -> denoise.
# This is the image that gets stored and trained on.
# =====================================================
def prepare_face(gray, bbox):

    (x, y, w, h) = bbox

    raw_face = gray[y:y + h, x:x + w]

    face = cv2.resize(
        raw_face,
        CONFIG["face_size"]
    )

    face = cv2.equalizeHist(face)

    return raw_face, face


# =====================================================
# QUALITY SCORE
#
# Composite 0-100 from the measurements already taken.
# Stored per capture so a weak enrollment can be spotted
# later without re-processing the images.
# =====================================================
def quality_score(blur_value, brightness, face_ratio, offset):

    def clamp(v):
        return max(0.0, min(1.0, v))

    # Sharpness: saturates at 4x the minimum
    sharp = clamp(blur_value / (CONFIG["min_blur"] * 4.0))

    # Lighting: best near the middle of the usable range
    mid = (CONFIG["min_brightness"] + CONFIG["max_brightness"]) / 2.0
    span = (CONFIG["max_brightness"] - CONFIG["min_brightness"]) / 2.0
    light = clamp(1.0 - abs(brightness - mid) / span)

    # Framing: face filling a good share of the frame
    size = clamp(face_ratio / 0.45)

    # Centring: 0 offset is perfect
    offset_x, offset_y = offset
    centred = clamp(
        1.0 - (offset_x / CONFIG["max_offset_x"] +
               offset_y / CONFIG["max_offset_y"]) / 2.0
    )

    score = (
        sharp * 40 +
        light * 25 +
        size * 20 +
        centred * 15
    )

    return int(round(score))


# =====================================================
# VALIDATE CAPTURE
#
# Runs the whole pipeline for ONE image.
#
# Returns:
#   {
#     "ok": bool,
#     "message": str | None,
#     "face": ndarray | None,   # 200x200, ready to store
#     "quality": int,
#     "checks": {...},          # for logging
#   }
# =====================================================
def validate_capture(image_data, expected_pose):

    result = {
        "ok": False,
        "message": None,
        "face": None,
        "quality": 0,
        "checks": {},
    }

    # ---- decode ----
    gray, err = decode_image(image_data)

    if err:
        result["message"] = err
        return result

    # ---- detect ----
    bbox, err = detect_face(gray)

    if err:
        result["message"] = err
        return result

    result["checks"]["face"] = True

    # ---- size ----
    ok, err, face_ratio = check_size(bbox, gray.shape)

    if not ok:
        result["message"] = err
        return result

    result["checks"]["size"] = True

    # ---- centring ----
    ok, err, offset = check_center(bbox, gray.shape)

    if not ok:
        result["message"] = err
        return result

    result["checks"]["centered"] = True

    # ---- crop / normalise ----
    raw_face, face = prepare_face(gray, bbox)

    # ---- brightness (raw crop) ----
    ok, err, brightness = check_brightness(raw_face)

    if not ok:
        result["message"] = err
        return result

    result["checks"]["lighting"] = True

    # ---- blur (before smoothing) ----
    ok, err, blur_value = check_blur(face)

    if not ok:
        result["message"] = err
        return result

    result["checks"]["sharp"] = True

    # ---- pose (placeholder) ----
    ok, err, _ = validate_pose(face, expected_pose)

    if not ok:
        result["message"] = err
        return result

    # ---- denoise last, so it never affects a measurement ----
    face = cv2.GaussianBlur(face, (3, 3), 0)

    result["ok"] = True
    result["face"] = face
    result["quality"] = quality_score(
        blur_value,
        brightness,
        face_ratio,
        offset
    )

    result["checks"]["metrics"] = {
        "blur": round(blur_value, 1),
        "brightness": round(brightness, 1),
        "face_ratio": round(face_ratio, 3),
        "offset_x": round(offset[0], 3),
        "offset_y": round(offset[1], 3),
    }

    return result