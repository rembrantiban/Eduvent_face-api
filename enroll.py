from flask import Blueprint, request, jsonify
import cv2
import os
import shutil
import numpy as np
import uuid
import json
import re
from datetime import datetime

from extensions import limiter

import cloudinary.uploader

from cloudinary_config import *

from database import (
    users_collection,
    face_users_collection
)

from face_validation import validate_capture

enroll_bp = Blueprint(
    "enroll",
    __name__
)

# =====================================================
# STORAGE CONFIGURATION
# =====================================================

STORAGE_DIR = "storage"

# All enrolled faces
FACES_DIR = os.path.join(
    STORAGE_DIR,
    "faces"
)

# Trained LBPH model
MODEL_PATH = os.path.join(
    STORAGE_DIR,
    "trainer.yml"
)

# Label mapping
LABELS_PATH = os.path.join(
    STORAGE_DIR,
    "labels.json"
)

# Required enrollment poses, in the order the
# frontend captures them
REQUIRED_POSES = [
    "front",
    "left",
    "right",
]

# LBPH distance. LOWER = more similar.
DUPLICATE_CONFIDENCE = 55

# How many of the three captures must match the same
# other student before we call it a duplicate
DUPLICATE_MIN_MATCHES = 2

MODEL_VERSION = "1.0"

# Ensure directories exist
os.makedirs(
    FACES_DIR,
    exist_ok=True
)


# =====================================================
# LOGGING HELPERS
# =====================================================
def log_line():
    print("-" * 40, flush=True)


def log_header(student_id):
    log_line()
    print("[ENROLL]", flush=True)
    print(f"Student: {student_id}", flush=True)
    log_line()


def log_capture(index, pose, result):

    print(f"\n[{index + 1}/{len(REQUIRED_POSES)}]", flush=True)
    print(pose.capitalize(), flush=True)

    checks = result.get("checks", {})

    if checks.get("face"):
        print("  Face detected", flush=True)
    if checks.get("size"):
        print("  Size OK", flush=True)
    if checks.get("centered"):
        print("  Centered", flush=True)
    if checks.get("lighting"):
        print("  Lighting OK", flush=True)
    if checks.get("sharp"):
        print("  Sharp", flush=True)

    if result["ok"]:
        print(f"  Quality: {result['quality']}", flush=True)
        print(f"  Metrics: {checks.get('metrics')}", flush=True)
        print("  Saved", flush=True)
    else:
        print(f"  REJECTED: {result['message']}", flush=True)

    log_line()


# =====================================================
# BUILD RECOGNIZER
#
# Trains an LBPH model over the stored faces.
#
# `exclude` skips one student folder, which is what
# makes the duplicate check meaningful - otherwise the
# student always matches themselves.
#
# Label registration only happens when a folder
# actually yielded images. The previous version
# incremented the label for empty folders too, which
# shifted every following label out of alignment.
# =====================================================
def build_recognizer(exclude=None):

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}

    current_label = 0

    for usn in os.listdir(FACES_DIR):

        if exclude and usn == exclude:
            continue

        folder = os.path.join(FACES_DIR, usn)

        if not os.path.isdir(folder):
            continue

        image_count = 0

        for image_name in os.listdir(folder):

            image_path = os.path.join(folder, image_name)

            img = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            img = cv2.resize(img, (200, 200))
            img = cv2.equalizeHist(img)

            faces.append(img)
            labels.append(current_label)

            image_count += 1

        # Only claim a label if the folder produced samples
        if image_count > 0:

            label_map[current_label] = usn
            current_label += 1

    if len(faces) == 0:
        return None, {}

    recognizer.train(faces, np.array(labels))

    return recognizer, label_map


# =====================================================
# TRAIN AND SAVE MODEL
#
# Same trainer.yml + labels.json as before.
# =====================================================
def train_model():

    recognizer, label_map = build_recognizer()

    if recognizer is None:
        raise Exception("No faces found for training")

    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f)

    for label, usn in label_map.items():
        print(f"[MODEL TRAIN] LABEL={label} USN={usn}", flush=True)

    print("[MODEL SAVED]", flush=True)


# =====================================================
# FIND EXISTING FACE OWNER
#
# Now checks ALL captures against every OTHER student,
# and trains once instead of once per image.
#
# A single frame matching is not enough - at least
# DUPLICATE_MIN_MATCHES of the three must point at the
# same student before enrollment is rejected.
# =====================================================
def find_existing_face_owner(faces, exclude=None):

    try:

        recognizer, label_map = build_recognizer(exclude=exclude)

        # Nobody else enrolled yet
        if recognizer is None:
            return None

        votes = {}

        for face in faces:

            label, confidence = recognizer.predict(face)

            print(
                f"[FACE CHECK CONFIDENCE] {round(confidence, 1)}",
                flush=True
            )

            # LOWER = BETTER
            if confidence < DUPLICATE_CONFIDENCE:

                owner = label_map.get(label)

                if owner:
                    votes[owner] = votes.get(owner, 0) + 1

        if not votes:
            return None

        owner, count = max(votes.items(), key=lambda kv: kv[1])

        if count >= DUPLICATE_MIN_MATCHES:

            print(
                f"[DUPLICATE] {owner} matched {count}/{len(faces)} captures",
                flush=True
            )

            return owner

        return None

    except Exception as e:

        print(f"[FACE CHECK ERROR] {str(e)}", flush=True)

        return None


# =====================================================
# SAVE FACE
#
# Named by pose: front.jpg / left.jpg / right.jpg
# =====================================================
def save_face(folder, pose, face):

    filename = os.path.join(folder, f"{pose}.jpg")

    cv2.imwrite(filename, face)

    print(f"[SAVED] {filename}", flush=True)

    return f"{pose}.jpg"


# =====================================================
# UPLOAD CLOUDINARY
#
# Front capture only - it is the profile image.
# =====================================================
def upload_cloudinary(usn, front_face):

    success, buffer = cv2.imencode(".jpg", front_face)

    if not success:
        raise Exception("Failed to encode front image")

    upload_result = cloudinary.uploader.upload(
        buffer.tobytes(),
        folder="eduvent_faces",
        public_id=usn,
        overwrite=True
    )

    print(f"[CLOUDINARY] {upload_result['secure_url']}", flush=True)

    return (
        upload_result["secure_url"],
        upload_result["public_id"]
    )


# =====================================================
# SAVE DATABASE
# =====================================================
def save_database(
    existing_user,
    usn,
    face_url,
    cloudinary_id,
    captures,
    quality
):

    user_id = str(existing_user["_id"])

    existing_face_user = face_users_collection.find_one({
        "userId": user_id
    })

    # Preserve the token across re-enrollment
    face_token = (
        existing_face_user["faceToken"]
        if existing_face_user and "faceToken" in existing_face_user
        else str(uuid.uuid4())
    )

    name = (
        f"{existing_user['firstName']} "
        f"{existing_user['lastName']}"
    )

    face_users_collection.replace_one(
        {
            "userId": user_id
        },
        {
            "userId": user_id,
            "studentId": usn,
            "name": name,
            "faceToken": face_token,
            "faceUrl": face_url,
            "cloudinaryId": cloudinary_id,
            "captures": captures,
            "quality": quality,
            "registeredAt": datetime.utcnow(),
            "modelVersion": MODEL_VERSION,
        },
        upsert=True
    )

    users_collection.update_one(
        {
            "_id": existing_user["_id"]
        },
        {
            "$set": {
                "isFaceRegistered": True
            }
        }
    )

    print("[DATABASE SAVED]", flush=True)

    return name


# =====================================================
# FIND STUDENT
# =====================================================
def find_student(usn):

    return users_collection.find_one({
        "studentId": {
            "$regex": f"^{re.escape(usn)}$",
            "$options": "i"
        }
    })


# =====================================================
# FAIL
#
# One exit point for errors, so the folder is always
# cleaned up and the response shape stays consistent.
# =====================================================
def fail(message, status=400, folder=None):

    if folder and os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)

    print(f"[FAILED] {message}", flush=True)

    return jsonify({
        "success": False,
        "message": message
    }), status


# =====================================================
# ENROLL PERSON
# =====================================================
@enroll_bp.route(
    "/enroll",
    methods=["POST"]
)
@limiter.limit("5 per minute")
def enroll_person():

    folder = None
    cloudinary_id = None

    try:

        data = request.json or {}

        usn = data.get("usn")
        images = data.get("images")

        if not usn or not images:
            return fail("USN or images missing", 400)

        usn = str(usn).strip()

        log_header(usn)

        # =========================
        # STUDENT LOOKUP
        # =========================
        existing_user = find_student(usn)

        if not existing_user:
            return fail("Student not found", 404)

        if existing_user.get("isFaceRegistered", False):
            return fail("Student already face registered", 400)

        # =========================
        # CAPTURE COUNT
        # =========================
        if len(images) != len(REQUIRED_POSES):
            return fail(
                f"Expected {len(REQUIRED_POSES)} captures, "
                f"received {len(images)}.",
                400
            )

        # =========================
        # VALIDATE EVERY CAPTURE
        #
        # Every pose must pass. A partial enrollment
        # trains the model on an incomplete face and
        # hurts recognition later.
        # =========================
        faces = []
        captures = {}
        quality = {}

        for index, image_data in enumerate(images):

            pose = REQUIRED_POSES[index]

            result = validate_capture(image_data, pose)

            log_capture(index, pose, result)

            if not result["ok"]:
                return fail(
                    f"{pose.capitalize()} capture: {result['message']}",
                    400
                )

            faces.append(result["face"])
            quality[pose] = result["quality"]

        # =========================
        # DUPLICATE CHECK
        #
        # Runs once, over all three captures, against
        # every OTHER student.
        # =========================
        print("Checking for duplicates...", flush=True)

        existing_owner = find_existing_face_owner(
            faces,
            exclude=usn
        )

        if existing_owner and existing_owner != usn:
            return fail(
                "This face is already registered by another student.",
                400
            )

        # =========================
        # SAVE FACES
        # =========================
        folder = os.path.join(FACES_DIR, usn)

        if os.path.exists(folder):
            shutil.rmtree(folder)

        os.makedirs(folder, exist_ok=True)

        for index, face in enumerate(faces):

            pose = REQUIRED_POSES[index]

            captures[pose] = save_face(folder, pose, face)

        log_line()

        # =========================
        # TRAIN
        # =========================
        print("Training...", flush=True)

        train_model()

        # =========================
        # CLOUDINARY
        # =========================
        print("Uploading...", flush=True)

        face_url, cloudinary_id = upload_cloudinary(
            usn,
            faces[0]
        )

        # =========================
        # DATABASE
        # =========================
        name = save_database(
            existing_user,
            usn,
            face_url,
            cloudinary_id,
            captures,
            quality
        )

        log_line()
        print("Enrollment Complete", flush=True)
        log_line()

        return jsonify({

            "success": True,

            "message": "Enrollment complete",

            "studentId": usn,

            "savedImages": len(faces),

            "name": name,

            "faceUrl": face_url

        }), 200

    except Exception as e:

        print(f"[ERROR] {str(e)}", flush=True)

        # Roll back the folder
        if folder and os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)

        # Roll back the upload
        if cloudinary_id:
            try:
                cloudinary.uploader.destroy(cloudinary_id)
            except Exception:
                pass

        return jsonify({
            "success": False,
            "message": "Enrollment failed",
            "error": str(e)
        }), 500