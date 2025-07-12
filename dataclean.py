import os
import cv2
import mediapipe as mp
from tqdm import tqdm
import shutil

def clean_dataset(root_dir, output_bad_dir="bad_samples"):
    mp_face_mesh = mp.solutions.face_mesh
    detector = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    if not os.path.exists(output_bad_dir):
        os.makedirs(output_bad_dir)

    total_removed = 0

    for split in ["train", "test"]:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            continue
        for emotion in os.listdir(split_dir):
            emotion_dir = os.path.join(split_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            bad_emotion_dir = os.path.join(output_bad_dir, split, emotion)
            os.makedirs(bad_emotion_dir, exist_ok=True)
            files = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            removed = 0
            for fname in tqdm(files, desc=f"{split}/{emotion}"):
                fpath = os.path.join(emotion_dir, fname)
                img = cv2.imread(fpath)
                if img is None:
                    shutil.move(fpath, os.path.join(bad_emotion_dir, fname))
                    removed += 1
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = detector.process(img_rgb)
                if not results.multi_face_landmarks:
                    shutil.move(fpath, os.path.join(bad_emotion_dir, fname))
                    removed += 1
            print(f"Removed {removed} images from {split}/{emotion}")
            total_removed += removed

    print(f"Total removed: {total_removed}")

if __name__ == "__main__":
    clean_dataset("facial_expression_dataset")