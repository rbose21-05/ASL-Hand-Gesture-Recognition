from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = load_model("asl_model.keras")

class_names = ["A", "B", "C", "D", "E"]

folder_path = "test_images"

for filename in os.listdir(folder_path):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(folder_path, filename)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"\n{filename}")

    for i, label in enumerate(class_names):
        print(f"{label}: {prediction[0][i]:.4f}")

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")