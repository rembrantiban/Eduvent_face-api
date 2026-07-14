from flask import Blueprint, request, jsonify
import cv2
import os
import shutil
import numpy as np
import base64
import uuid
import json
import re
from extensions import limiter

import cloudinary.uploader

from cloudinary_config import *

from database import (
    users_collection,
    face_users_collection
)

enroll_bp = Blueprint(
    "enroll",
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

os.makedirs(
    FACES_DIR,
    exist_ok=True
)

# =========================
# TRAIN AND SAVE MODEL
# =========================
def train_model():

    recognizer = (
        cv2.face.LBPHFaceRecognizer_create()
    )

    faces = []
    labels = []
    label_map = {}

    current_label = 0

    for usn in os.listdir(
        FACES_DIR
    ):

        folder = os.path.join(
            FACES_DIR,
            usn
        )

        if not os.path.isdir(folder):
            continue

        image_count = 0

        for image_name in os.listdir(folder):

            image_path = os.path.join(
                folder,
                image_name
            )

            img = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            img = cv2.resize(
                img,
                (200, 200)
            )

            img = cv2.equalizeHist(
                img
            )

            faces.append(img)

            labels.append(current_label)

            image_count += 1

        if image_count > 0:

            label_map[current_label] = usn

            print(
                f"[MODEL TRAIN] LABEL={current_label} USN={usn}",
                flush=True
            )

            current_label += 1

    if len(faces) == 0:

        raise Exception(
            "No faces found for training"
        )

    recognizer.train(
        faces,
        np.array(labels)
    )

    # =========================
    # SAVE MODEL
    # =========================
    recognizer.save(
        MODEL_PATH
    )

    # =========================
    # SAVE LABEL MAP
    # =========================
    with open(
        LABELS_PATH,
        "w"
    ) as f:

        json.dump(
            label_map,
            f
        )

    print(
        "[MODEL SAVED]",
        flush=True
    )


# =========================
# FIND EXISTING FACE OWNER
# =========================
def find_existing_face_owner(
    new_face
):

    try:

        recognizer = (
            cv2.face.LBPHFaceRecognizer_create()
        )

        faces = []
        labels = []
        label_map = {}

        current_label = 0

        for usn in os.listdir(
            FACES_DIR
        ):

            folder = os.path.join(
                FACES_DIR,
                usn
            )

            if not os.path.isdir(folder):
                continue

            for image_name in os.listdir(folder):

                image_path = os.path.join(
                    folder,
                    image_name
                )

                img = cv2.imread(
                    image_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if img is None:
                    continue

                img = cv2.resize(
                    img,
                    (200, 200)
                )

                img = cv2.equalizeHist(
                    img
                )

                faces.append(img)

                labels.append(current_label)

            label_map[current_label] = usn

            current_label += 1

        if len(faces) == 0:
            return None

        recognizer.train(
            faces,
            np.array(labels)
        )

        label, confidence = (
            recognizer.predict(
                new_face
            )
        )

        print(
            f"[FACE CHECK CONFIDENCE] {confidence}",
            flush=True
        )

        # LOWER = BETTER
        if confidence < 55:

            return label_map.get(label)

        return None

    except Exception as e:

        print(
            f"[FACE CHECK ERROR] {str(e)}",
            flush=True
        )

        return None


# =========================
# ENROLL PERSON
# =========================
@enroll_bp.route(
    "/enroll",
    methods=["POST"]
)
@limiter.limit("5 per minute")
def enroll_person():

    try:

        data = request.json

        print("=================================")
        print("REQUEST DATA:", data)
        print("USN RECEIVED:", data.get("usn"))
        print("=================================")


        usn = data.get("usn")

        images = data.get("images")

        if not usn or not images:

            return jsonify({
                "success": False,
                "message":
                "USN or images missing"
            }), 400

        usn = str(usn).strip()

        print("================================")
        print("Searching studentId:", repr(usn))


        existing_user = users_collection.find_one({
            "studentId": {
                "$regex": f"^{re.escape(usn)}$",
                "$options": "i"
            }
        })

        
        print("Found user:", existing_user)
        print("================================")
        
        print("Incoming studentId:", repr(usn))

        print("Existing student IDs:")

        for user in users_collection.find(
            {},
            {
                "studentId": 1,
                "_id": 0
            }
        ):
            print(user)
                

        if not existing_user:

            return jsonify({
                "success": False,
                "message":
                "Student not found"
            }), 404

        # =========================
        # CHECK FACE ALREADY REGISTERED
        # =========================
        if existing_user.get(
            "isFaceRegistered",
            False
        ):

            return jsonify({
                "success": False,
                "message":
                "Student already face registered"
            }), 400

        # =========================
        # CREATE FOLDER
        # =========================
        folder = os.path.join(
            FACES_DIR,
            usn
        )

        if os.path.exists(folder):

            shutil.rmtree(folder)

        os.makedirs(
            folder,
            exist_ok=True
        )

        saved_faces = []

        # =========================
        # FACE DETECTOR
        # =========================
        face_cascade = (
            cv2.CascadeClassifier(
                cv2.data.haarcascades +
                "haarcascade_frontalface_alt2.xml"
            )
        )

        # =========================
        # LOOP IMAGES
        # =========================
        for index, image_data in enumerate(images):

            print(
                f"[PROCESSING IMAGE] {index+1}",
                flush=True
            )

            try:

                img_bytes = (
                    base64.b64decode(
                        image_data.split(",")[1]
                    )
                )

                nparr = np.frombuffer(
                    img_bytes,
                    np.uint8
                )

                img_np = cv2.imdecode(
                    nparr,
                    cv2.IMREAD_GRAYSCALE
                )

                if img_np is None:

                    print(
                        "[INVALID IMAGE]",
                        flush=True
                    )

                    continue

            except Exception as e:

                print(
                    f"[DECODE ERROR] {str(e)}",
                    flush=True
                )

                continue

            # =========================
            # FIX MIRROR
            # =========================
            img_np = cv2.flip(
                img_np,
                1
            )

            # =========================
            # LIGHTING NORMALIZATION
            # =========================
            img_np = cv2.equalizeHist(
                img_np
            )

            # =========================
            # FACE DETECTION
            # =========================
            detected_faces = (
                face_cascade.detectMultiScale(
                    img_np,
                    scaleFactor=1.1,
                    minNeighbors=2,
                    minSize=(80, 80)
                )
            )

            print(
                f"[FACES DETECTED] {len(detected_faces)}",
                flush=True
            )

            if len(detected_faces) == 0:

                print(
                    "[NO FACE DETECTED]",
                    flush=True
                )

                continue

            # =========================
            # GET LARGEST FACE
            # =========================
            (x, y, w, h) = max(
                detected_faces,
                key=lambda face:
                face[2] * face[3]
            )

            face = img_np[
                y:y+h,
                x:x+w
            ]

            # =========================
            # RESIZE
            # =========================
            face = cv2.resize(
                face,
                (200, 200)
            )

            # =========================
            # NORMALIZE
            # =========================
            face = cv2.equalizeHist(
                face
            )

            # =========================
            # REDUCE NOISE
            # =========================
            face = cv2.GaussianBlur(
                face,
                (3, 3),
                0
            )

            # =========================
            # BLUR CHECK
            # =========================
            blur_value = cv2.Laplacian(
                face,
                cv2.CV_64F
            ).var()

            print(
                f"[BLUR VALUE] {blur_value}",
                flush=True
            )

            if blur_value < 45:

                print(
                    "[BLURRY IMAGE]",
                    flush=True
                )

                continue

            # =========================
            # CHECK DUPLICATE FACE
            # =========================
            existing_owner = (
                find_existing_face_owner(
                    face
                )
            )

            if (
                existing_owner and
                existing_owner != usn
            ):

                shutil.rmtree(
                    folder
                )

                return jsonify({
                    "success": False,
                    "message":
                    f"Face already registered by another user"
                }), 400

            # =========================
            # SAVE FACE
            # =========================
            filename = os.path.join(
                folder,
                f"{usn}_{index+1}.jpg"
            )

            cv2.imwrite(
                filename,
                face
            )

            print(
                f"[SAVED] {filename}",
                flush=True
            )

            saved_faces.append(face)

        # =========================
        # VALIDATE
        # =========================
        if len(saved_faces) < 1:

            shutil.rmtree(folder)

            return jsonify({
                "success": False,
                "message":
                "Need at least 1 valid face capture"
            }), 400

        print(
            f"[TOTAL SAVED] {len(saved_faces)}",
            flush=True
        )

        # =========================
        # RETRAIN MODEL
        # =========================
        train_model()

        print(
            "[MODEL RETRAINED]",
            flush=True
        )

        # =========================
        # CLOUDINARY
        # =========================
        success, buffer = cv2.imencode(
            ".jpg",
            saved_faces[0]
        )

        if not success:

            shutil.rmtree(folder)

            return jsonify({
                "success": False,
                "message":
                "Failed to process image"
            }), 500

        upload_result = (
            cloudinary.uploader.upload(
                buffer.tobytes(),
                folder="eduvent_faces",
                public_id=usn,
                overwrite=True
            )
        )

        face_url = upload_result[
            "secure_url"
        ]

        cloudinary_id = upload_result[
            "public_id"
        ]

        print(
            f"[CLOUDINARY] {face_url}",
            flush=True
        )

        # =========================
        # SAVE DATABASE
        # =========================
        existing_face_user = (
            face_users_collection.find_one({
                "userId":
                str(existing_user["_id"])
            })
        )

        face_token = (
            existing_face_user["faceToken"]
            if existing_face_user and
            "faceToken" in existing_face_user
            else str(uuid.uuid4())
        )

        face_users_collection.replace_one(
            {
                "userId":
                str(existing_user["_id"])
            },
            {
                "userId":
                str(existing_user["_id"]),

                "name":
                f"{existing_user['firstName']} "
                f"{existing_user['lastName']}",

                "faceUrl":
                face_url,

                "cloudinaryId":
                cloudinary_id,

                "faceToken":
                face_token
            },
            upsert=True
        )

        # =========================
        # UPDATE USER
        # =========================
        users_collection.update_one(
            {
                "_id":
                existing_user["_id"]
            },
            {
                "$set": {
                    "isFaceRegistered": True
                }
            }
        )

        return jsonify({

            "success": True,

            "message":
            "Enrollment complete",

            "studentId":
            usn,

            "savedImages":
            len(saved_faces),

            "name":
            f"{existing_user['firstName']} "
            f"{existing_user['lastName']}",

            "faceUrl":
            face_url

        }), 200

    except Exception as e:

        print(
            f"[ERROR] {str(e)}",
            flush=True
        )

        try:

            if "cloudinary_id" in locals():

                cloudinary.uploader.destroy(
                    cloudinary_id
                )

        except:
            pass

        return jsonify({
            "success": False,
            "message":
            "Enrollment failed",
            "error": str(e)
        }), 500