import mediapipe as mp
import cv2
import os
 
DATA_DIR='./data'
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"{DATA_DIR} does not exist. Please run collect_imgs.py to create the dataset first.")

OUTPUT_DIR='./dataset'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

number_of_classes = 3
for j in range(number_of_classes):
    class_folder = os.path.join(OUTPUT_DIR, str(j))
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

print(f"Output directory structure created at {OUTPUT_DIR}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
padding = 20 
for class_num in range(number_of_classes):
    print (f"Processing class {class_num}...")
    d_size = 100
    for j in range(d_size):
        img_path = os.path.join(DATA_DIR, str(class_num), f"{j}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Converted {img_path} to RGB")
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            img_height, img_width, _ = image.shape
            x_mins, y_mins = img_width, img_height
            x_maxs, y_maxs = 0, 0
            for landmark in results.multi_hand_landmarks[0].landmark:
                # Convert normalized coordinates (0-1) to pixel coordinates
                x = int(landmark.x * img_width)
                y = int(landmark.y * img_height)

                # Update min/max
                if x < x_mins:
                    x_mins = x
                if x > x_maxs:
                    x_maxs = x
                if y < y_mins:
                    y_mins = y
                if y > y_maxs:
                    y_maxs = y
                
            #Padding

            x_mins = max(0, x_mins - padding)  # Don't go below 0
            y_mins = max(0, y_mins - padding)
            x_maxs = min(img_width, x_maxs + padding)  # Don't exceed image width
            y_maxs = min(img_height, y_maxs + padding)  # Don't exceed image height

            print(f"Bounding box for {img_path}: ({x_mins}, {y_mins}) to ({x_maxs}, {y_maxs})")
            
            hand_crop = image[y_mins:y_maxs, x_mins:x_maxs] 
            hand_resized = cv2.resize(hand_crop, (224, 224))
            output_path = os.path.join(OUTPUT_DIR, str(class_num), f"{j}.jpg")
            cv2.imwrite(output_path, hand_resized)
            print(f"Processed image saved to {output_path}")    
        else:
            print(f"No hand landmarks detected in image {img_path}. Skipping.")
            continue