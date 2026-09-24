"""
One-time backfill: students enrolled before SFace embeddings
existed have no `embeddings` in faceusers, so the duplicate check
cannot match against them. This rebuilds one embedding per student
from their Cloudinary front image (faceUrl).

Usage:
    python backfill_embeddings.py           # dry run, prints only
    python backfill_embeddings.py --write   # saves to MongoDB
"""

import sys
import urllib.request

import cv2
import numpy as np

from database import face_users_collection
from face_embedding import embed


def fetch_image(url):

    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()

    return cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR
    )


def main():

    write = "--write" in sys.argv

    missing = face_users_collection.find({
        "embeddings.0": {"$exists": False}
    })

    done = skipped = 0

    for doc in missing:

        usn = doc.get("studentId")
        url = doc.get("faceUrl")

        if not url:
            print(f"[SKIP] {usn}: no faceUrl")
            skipped += 1
            continue

        try:
            # Stored images are tight 200x200 crops
            embedding = embed(fetch_image(url), padded=True)
        except Exception as e:
            print(f"[SKIP] {usn}: {e}")
            skipped += 1
            continue

        if embedding is None:
            print(f"[SKIP] {usn}: no face found in stored image")
            skipped += 1
            continue

        if write:
            face_users_collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"embeddings": [embedding]}}
            )

        print(f"[OK] {usn}")
        done += 1

    mode = "saved" if write else "dry run, nothing saved"
    print(f"\n{done} backfilled, {skipped} skipped ({mode})")


if __name__ == "__main__":
    main()
