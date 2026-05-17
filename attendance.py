from flask import Blueprint
import cv2
import numpy as np
import os

recognize_bp = Blueprint(
    "recognize",
    __name__
)
# =========================
# TRAIN MODEL
# =========================
def train_model():

    recognizer = (
        cv2.face.LBPHFaceRecognizer_create()
    )

    faces = []
    labels = []
    label_map = {}

    current_label = 0

    os.makedirs(
        "faces",
        exist_ok=True
    )

    for usn in os.listdir("faces"):

        print(
            "[TRAINING USN]",
            usn,
            flush=True
        )

        folder = os.path.join(
            "faces",
            usn
        )

        if not os.path.isdir(folder):
            continue

        normalized_usn = str(
            usn
        ).strip().lower()

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

            try:

                img = cv2.resize(
                    img,
                    (200, 200)
                )

            except:
                continue

            faces.append(img)

            labels.append(current_label)

            image_count += 1

        # =========================
        # REGISTER LABEL ONLY
        # IF IMAGES EXIST
        # =========================
        if image_count > 0:

            label_map[current_label] = (
                normalized_usn
            )

            print(
                f"[LABEL {current_label}] "
                f"=> {normalized_usn}",
                flush=True
            )

            current_label += 1

    # =========================
    # NO FACES
    # =========================
    if len(faces) == 0:

        raise Exception(
            "No enrolled faces found"
        )

    # =========================
    # TRAIN MODEL
    # =========================
    recognizer.train(
        faces,
        np.array(labels)
    )

    print(
        "[TRAINING COMPLETE]",
        flush=True
    )

    print(
        "[TOTAL FACES]",
        len(faces),
        flush=True
    )

    print(
        "[LABEL MAP]",
        label_map,
        flush=True
    )

    return recognizer, label_map


# =========================
# CHECK DUPLICATE FACE
# =========================
def is_face_already_registered(
    new_face,
    current_usn=None
):

    try:

        recognizer, label_map = (
            train_model()
        )

    except Exception as e:

        print(
            "[DUPLICATE CHECK ERROR]",
            str(e),
            flush=True
        )

        return False

    try:

        label, confidence = (
            recognizer.predict(
                new_face
            )
        )

    except Exception as e:

        print(
            "[PREDICT ERROR]",
            str(e),
            flush=True
        )

        return False

    print(
        "[DUPLICATE CHECK]",
        flush=True
    )

    print(
        "[LABEL]",
        label,
        flush=True
    )

    print(
        "[CONFIDENCE]",
        confidence,
        flush=True
    )

    # =========================
    # UNKNOWN LABEL
    # =========================
    if label not in label_map:
        return False

    predicted_usn = str(
        label_map[label]
    ).strip().lower()

    print(
        "[PREDICTED USN]",
        predicted_usn,
        flush=True
    )

    # =========================
    # SAME USER RE-ENROLL
    # =========================
    if (
        current_usn and
        predicted_usn == current_usn
    ):

        print(
            "[SAME USER RE-ENROLL]",
            flush=True
        )

        return False

    # =========================
    # LOWER = BETTER MATCH
    # =========================
    if confidence < 50:

        print(
            "[FACE ALREADY EXISTS]",
            flush=True
        )

        return True

    return False