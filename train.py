import os, cv2
import numpy as np

# Load training images
folder = 'training-images'
images, labels = [], []
label_map = {}
label_id = 0

for person in os.listdir(folder):
    label_map[label_id] = person
    person_folder = os.path.join(folder, person)
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        images.append(img)
        labels.append(label_id)
    label_id += 1

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))
recognizer.save('trained_model.yml')

print("Training completed. Model saved.")
