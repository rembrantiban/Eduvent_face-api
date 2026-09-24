from flask import Blueprint, request, jsonify
import cv2
import os
import re

from extensions import limiter
from database import (
    users_collection,
    face_users_collection
)

from face_embedding import (
    decode_color,
    detect_and_embed,
    similarity,
    SAME_PERSON_THRESHOLD
)

recognize_bp = Blueprint(
    "recognize",
    __name__
)

# =========================
# SETTINGS
# =========================

# Laplacian variance of the face crop. Lower = blurrier.
MIN_BLUR = float(
    os.environ.get("FACE_MIN_BLUR", 40)
)

# Same size enrollment stores faces at
BLUR_FACE_SIZE = (200, 200)


# =========================
# FIND FACE USER
# =========================
def find_face_user(usn):

    return face_users_collection.find_one(
        {
            "studentId": {
                "$regex": f"^{re.escape(usn)}$",
                "$options": "i"
            },
            "embeddings.0": {"$exists": True}
        },
        {
            "studentId": 1,
            "embeddings": 1
        }
    )


# =========================
# BEST MATCH FOR ONE STUDENT
#
# Highest similarity between the probe and any of the
# student's stored embeddings (front / left / right).
# =========================
def best_similarity(probe, face_user):

    return max(
        similarity(probe, stored)
        for stored in face_user["embeddings"]
    )


# =========================
# SEARCH ALL STUDENTS
#
# Only used when the caller does not say who the
# student should be. Returns (studentId, similarity).
# =========================
def search_all(probe):

    best_usn = None
    best_score = -1.0

    for face_user in face_users_collection.find(
        {"embeddings.0": {"$exists": True}},
        {"studentId": 1, "embeddings": 1}
    ):

        score = best_similarity(probe, face_user)

        if score > best_score:
            best_usn = face_user.get("studentId")
            best_score = score

    return best_usn, best_score


# =========================
# RECOGNIZE FACE
#
# With expectedUsn: 1:1 verification against that
# student only. Without it: 1:N search over everyone.
#
# `confidence` is a 0-100 match percentage.
# HIGHER = BETTER.
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

    data = request.json or {}

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

        frame = decode_color(image_data)

        if frame is None:
            raise Exception("Image could not be read")

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
    # DETECT + EMBED
    # =========================
    probe, face_crop = detect_and_embed(frame)

    if probe is None:

        return jsonify({
            "success": False,
            "message": "No face detected"
        }), 400

    # =========================
    # BLUR CHECK
    # =========================
    # Measured at a fixed size: Laplacian variance drops
    # as resolution rises, so a sharp 12MP selfie would
    # otherwise score as blurry.
    face_gray = cv2.resize(
        cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY),
        BLUR_FACE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    blur_value = cv2.Laplacian(
        face_gray,
        cv2.CV_64F
    ).var()

    print(
        f"[BLUR VALUE] {blur_value}",
        flush=True
    )

    if blur_value < MIN_BLUR:

        return jsonify({
            "success": False,
            "message": "Image too blurry"
        }), 400

    # =========================
    # MATCH
    # =========================
    if expected_usn:

        face_user = find_face_user(expected_usn)

        if not face_user:

            print(
                f"[NO EMBEDDINGS] {expected_usn}",
                flush=True
            )

            return jsonify({
                "success": False,
                "message":
                "This student is not face registered"
            }), 403

        predicted_usn = expected_usn
        score = best_similarity(probe, face_user)

    else:

        predicted_usn, score = search_all(probe)

        if predicted_usn is None:

            return jsonify({
                "success": False,
                "message": "No registered faces"
            }), 400

        predicted_usn = str(
            predicted_usn
        ).strip().lower()

    confidence = round(
        max(score, 0.0) * 100,
        2
    )

    print(
        f"[MATCH] usn={predicted_usn} similarity={round(score, 3)} "
        f"threshold={SAME_PERSON_THRESHOLD}",
        flush=True
    )

    # =========================
    # CONFIDENCE CHECK
    # HIGHER = BETTER
    # =========================
    if score < SAME_PERSON_THRESHOLD:

        if expected_usn:

            return jsonify({
                "success": False,
                "message":
                "Face does not match the registered face of this account",
                "confidence": confidence
            }), 403

        return jsonify({
            "success": False,
            "message": "Face not recognized",
            "confidence": confidence
        }), 401

    # =========================
    # CHECK USER IN DATABASE
    # =========================
    registered_user = users_collection.find_one({
        "studentId": {
            "$regex": f"^{re.escape(predicted_usn)}$",
            "$options": "i"
        }
    })

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
    # SUCCESS
    # =========================
    return jsonify({

        "success": True,

        "usn": predicted_usn,

        "confidence": confidence

    }), 200
