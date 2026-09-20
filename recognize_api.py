from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import os
import base64
import json

from extensions import limiter
from database import users_collection

recognize_bp = Blueprint(
    "recognize",
    __name__
)

# =========================
# STORAGE PATHS
# =========================
STORAGE_DIR = "storage"

FACES_DIR = os.path.join(
    STORAGE_DIR,
    "faces"
)

MODEL_PATH = os.path.join(
    STORAGE_DIR,
    "trainer.yml"
)

LABELS_PATH = os.path.join(
    STORAGE_DIR,
    "labels.json"
)

# =========================
# SETTINGS
# =========================
FACE_SIZE = (200, 200)

# LOWER = BETTER (LBPH distance). 110 accepted almost any face;
# ~50-80 is a strict match. Tune with FACE_THRESHOLD if genuine users are rejected.
RECOGNITION_THRESHOLD = float(
    os.environ.get("FACE_THRESHOLD", 70)
)


# =========================
# LOAD MODEL
# =========================
def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise Exception(
            "Trainer model not found"
        )

    if not os.path.exists(
        LABELS_PATH
    ):

        raise Exception(
            "Labels file not found"
        )

    recognizer = (
        cv2.face.LBPHFaceRecognizer_create()
    )

    recognizer.read(
        MODEL_PATH
    )

    with open(
        LABELS_PATH,
        "r"
    ) as f:

        label_map = json.load(f)

    # JSON converts keys to string
    label_map = {
        int(k): v
        for k, v in label_map.items()
    }

    print(
        "[MODEL LOADED]",
        flush=True
    )

    print(
        f"[LABEL MAP] {label_map}",
        flush=True
    )

    return recognizer, label_map


# =========================
# RECOGNIZE FACE
# =========================
@recognize_bp.route(
    "/recognize",
    methods=["POST"]
)
@limiter.limit("10 per minute")
def recognize():

    print(
        "[RECOGNIZE ROUTE HIT]",
        flush=True
    )

    data = request.json

    image_data = data.get("image")

    expected_usn = str(
        data.get("expectedUsn") or ""
    ).strip().lower()

    if not image_data:

        return jsonify({
            "success": False,
            "message": "No image received"
        }), 400

    # =========================
    # DECODE IMAGE
    # =========================
    try:

        img_bytes = base64.b64decode(
            image_data.split(",")[1]
        )

        nparr = np.frombuffer(
            img_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            raise Exception()

    except Exception as e:

        print(
            "[IMAGE ERROR]",
            str(e),
            flush=True
        )

        return jsonify({
            "success": False,
            "message": "Invalid image"
        }), 400

    # =========================
    # LOAD SAVED MODEL
    # =========================
    try:

        recognizer, label_map = (
            load_model()
        )

    except Exception as e:

        print(
            "[MODEL LOAD ERROR]",
            str(e),
            flush=True
        )

        return jsonify({
            "success": False,
            "message": str(e)
        }), 400

    # =========================
    # FACE DETECTOR
    # =========================
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_alt2.xml"
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # =========================
    # NORMALIZE LIGHTING
    # =========================
    gray = cv2.equalizeHist(
        gray
    )

    # =========================
    # DETECT FACE
    # =========================
    detected_faces = (
        face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(80, 80)
        )
    )

    print(
        f"[FACE DETECTED] {len(detected_faces)}",
        flush=True
    )

    if len(detected_faces) == 0:

        return jsonify({
            "success": False,
            "message": "No face detected"
        }), 400

    # =========================
    # USE LARGEST FACE
    # =========================
    (x, y, w, h) = max(
        detected_faces,
        key=lambda face:
        face[2] * face[3]
    )

    face_img = gray[
        y:y+h,
        x:x+w
    ]

    # =========================
    # BLUR CHECK
    # =========================
    blur_value = cv2.Laplacian(
        face_img,
        cv2.CV_64F
    ).var()

    print(
        f"[BLUR VALUE] {blur_value}",
        flush=True
    )

    if blur_value < 40:

        return jsonify({
            "success": False,
            "message": "Image too blurry"
        }), 400

    # =========================
    # PREPROCESS
    # =========================
    face_img = cv2.resize(
        face_img,
        FACE_SIZE
    )

    face_img = cv2.equalizeHist(
        face_img
    )

    face_img = cv2.GaussianBlur(
        face_img,
        (3, 3),
        0
    )

    # =========================
    # PREDICT
    # =========================
    try:

        label, confidence = (
            recognizer.predict(
                face_img
            )
        )

    except Exception as e:

        print(
            "[PREDICT ERROR]",
            str(e),
            flush=True
        )

        return jsonify({
            "success": False,
            "message": "Prediction failed"
        }), 500

    # =========================
    # UNKNOWN LABEL
    # =========================
    if label not in label_map:

        print(
            f"[UNKNOWN LABEL] {label}",
            flush=True
        )

        return jsonify({
            "success": False,
            "message": "Unknown face label"
        }), 400

    predicted_usn = str(
        label_map[label]
    ).strip().lower()

    # =========================
    # MUST BE THE ACCOUNT OWNER
    # =========================
    if expected_usn and predicted_usn != expected_usn:

        print(
            f"[IDENTITY MISMATCH] predicted={predicted_usn} expected={expected_usn}",
            flush=True
        )

        return jsonify({
            "success": False,
            "message":
            "Face does not match the registered face of this account"
        }), 403

    # =========================
    # CHECK USER IN DATABASE
    # =========================
    registered_user = (
        users_collection.find_one({
            "studentId": predicted_usn
        })
    )

    if not registered_user:

        print(
            f"[USER NOT FOUND] {predicted_usn}",
            flush=True
        )

        return jsonify({
            "success": False,
            "message":
            "This student is not registered"
        }), 404

    # =========================
    # CHECK FACE REGISTERED
    # =========================
    if not registered_user.get(
        "isFaceRegistered",
        False
    ):

        print(
            f"[FACE NOT REGISTERED] {predicted_usn}",
            flush=True
        )

        return jsonify({
            "success": False,
            "message":
            "This student is not face registered"
        }), 403

    # =========================
    # FINAL DEBUG
    # =========================
    print(
        "================================",
        flush=True
    )

    print(
        f"[LABEL] {label}",
        flush=True
    )

    print(
        f"[PREDICTED USN] {predicted_usn}",
        flush=True
    )

    print(
        f"[CONFIDENCE] {confidence}",
        flush=True
    )

    print(
        f"[LABEL MAP] {label_map}",
        flush=True
    )

    print(
        "================================",
        flush=True
    )

    # =========================
    # CONFIDENCE CHECK
    # LOWER = BETTER
    # =========================
    if confidence > RECOGNITION_THRESHOLD:

        return jsonify({
            "success": False,
            "message": "Face not recognized",
            "confidence": round(
                confidence,
                2
            )
        }), 401

    # =========================
    # SUCCESS
    # =========================
    return jsonify({

        "success": True,

        "usn": predicted_usn,

        "confidence": round(
            confidence,
            2
        )

    }), 200