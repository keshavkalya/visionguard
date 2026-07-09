import cv2
import os
import numpy as np
import json

dataset_path = "dataset"

faces = []   # list of face images
labels = []  # matching list of numeric IDs (LBPH needs numbers, not names)
names = {}   # maps ID number -> person name, e.g. {0: "alice"}

# Go through each person's folder in dataset/
for label_id, person_name in enumerate(sorted(os.listdir(dataset_path))):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue  # skip stray files like .DS_Store

    names[label_id] = person_name

    # Load every photo of this person
    for filename in os.listdir(person_folder):
        img_path = os.path.join(person_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # skip unreadable files

        # LBPH works best when all images are the same size
        img = cv2.resize(img, (200, 200))
        faces.append(img)
        labels.append(label_id)

print(f"Training on {len(faces)} images of {len(names)} person(s): {list(names.values())}")

# Create and train the LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Save what it learned + the name mapping
recognizer.save("trainer.yml")
with open("names.json", "w") as f:
    json.dump(names, f)

print("Done! Saved trainer.yml and names.json")