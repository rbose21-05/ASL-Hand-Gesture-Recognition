import os
import shutil
import random
SOURCE_DIR = './asl_alphabet_train'
OUTPUT_DIR = './asl_abc'

classes = ['A', 'B', 'C']


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


for class_name in classes:
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    
    if not os.path.exists(train_class_dir):
        os.makedirs(train_class_dir)
        print(f"Created {train_class_dir}")
    
    if not os.path.exists(val_class_dir):
        os.makedirs(val_class_dir)
        print(f"Created {val_class_dir}")

    source_class_dir = os.path.join(SOURCE_DIR, class_name)
    images = images[:1000]
    images = [ f for f in os.listdir(source_class_dir) if f.endswith('.jpg') or f.endswith('.png') ]
    random.shuffle(images)
    split_index = int(0.8 * len(images))
    train_images = images[:split_index]
    val_images = images[split_index:]
    print(f"Class {class_name}: {len(train_images)} images for training, {len(val_images)} images for validation.")
    
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
