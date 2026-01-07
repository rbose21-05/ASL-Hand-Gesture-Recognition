import os
import shutil
import random
SOURCE_DIR = './dataset'
OUTPUT_DIR = './data_split'
number_of_classes = 3


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory at {OUTPUT_DIR}")



train_dir = os.path.join(OUTPUT_DIR, 'train')
val_dir = os.path.join(OUTPUT_DIR, 'validation')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    print(f"Created {train_dir}")

if not os.path.exists(val_dir):
    os.makedirs(val_dir)
    print(f"Created {val_dir}")


for class_num in range(number_of_classes):
    train_class_dir = os.path.join(train_dir, str(class_num))
    val_class_dir = os.path.join(val_dir, str(class_num))
    
    if not os.path.exists(train_class_dir):
        os.makedirs(train_class_dir)
        print(f"Created {train_class_dir}")
    
    if not os.path.exists(val_class_dir):
        os.makedirs(val_class_dir)
        print(f"Created {val_class_dir}")

    source_class_dir = os.path.join(SOURCE_DIR, str(class_num))
    images = [ f for f in os.listdir(source_class_dir) if f.endswith('.jpg') or f.endswith('.png') ]
    random.shuffle(images)
    split_index = int(0.8 * len(images))
    train_images = images[:split_index]
    val_images = images[split_index:]
    print(f"Class {class_num}: {len(train_images)} images for training, {len(val_images)} images for validation.")
    
    for img_name in train_images:
        src_path = os.path.join(source_class_dir, img_name)
        dst_path = os.path.join(train_class_dir, img_name)
        shutil.copyfile(src_path, dst_path)
  
    for img_name in val_images:
        src_path = os.path.join(source_class_dir, img_name)
        dst_path = os.path.join(val_class_dir, img_name)
        shutil.copyfile(src_path, dst_path)

print("\n✅ Data split complete!")
print(f"Training data: {OUTPUT_DIR}/train/")
print(f"Validation data: {OUTPUT_DIR}/validation/")
