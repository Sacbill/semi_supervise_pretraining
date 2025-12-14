from torchvision import datasets
from collections import Counter

train_data = datasets.ImageFolder("datasets/datasets/train")
val_data   = datasets.ImageFolder("datasets/datasets/val")

# Count labels in train set
train_labels = [label for _, label in train_data]
train_counts = Counter(train_labels)

# Count labels in val set
val_labels = [label for _, label in val_data]
val_counts = Counter(val_labels)

# Map class indices to class names
classes = train_data.classes

print("Train counts:")
for cls_idx, count in train_counts.items():
    print(f"{classes[cls_idx]}: {count}")

print("\nValidation counts:")
for cls_idx, count in val_counts.items():
    print(f"{classes[cls_idx]}: {count}")
