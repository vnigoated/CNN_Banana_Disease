import os
import shutil
from sklearn.model_selection import train_test_split


base_dir = "C:\\myprograms\\cnn-banana\\Banana Disease Recognition Dataset"
augmented_dir = os.path.join(base_dir, "Augmented images", "Augmented images")
original_dir = os.path.join(base_dir, "Original Images", "Original Images")
n
target_dir = "dataset"
train_dir = os.path.join(target_dir, "train")
validation_dir = os.path.join(target_dir, "validation")


os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

disease_classes = [
    "Banana Black Sigatoka Disease",
    "Banana Bract Mosaic Virus Disease",
    "Banana Healthy Leaf",
    "Banana Insect Pest Disease",
    "Banana Moko Disease",
    "Banana Panama Disease",
    "Banana Yellow Sigatoka Disease"
]

for disease in disease_classes:
    # Collect all images
    augmented_path = os.path.join(augmented_dir, f"Augmented {disease}")
    original_path = os.path.join(original_dir, disease)

    all_images = []
    for folder in [augmented_path, original_path]:
        images = [os.path.join(folder, img) for img in os.listdir(folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        all_images.extend(images)

    # Split into train and validation (80/20)
    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

    # Create destination folders
    train_class_dir = os.path.join(train_dir, disease.lower().replace(' ', '_'))
    val_class_dir = os.path.join(validation_dir, disease.lower().replace(' ', '_'))

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # Copy images
    for img_path in train_images:
        shutil.copy(img_path, train_class_dir)

    for img_path in val_images:
        shutil.copy(img_path, val_class_dir)

print("Dataset organized successfully!")
