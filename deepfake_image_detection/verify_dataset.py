from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path

# Path to your dataset folder
DATA_DIR = Path("dataset")

# Simple transform (just to load the images)
transform = transforms.ToTensor()

# Load datasets
train_dataset = ImageFolder(DATA_DIR / "train", transform=transform)
test_dataset  = ImageFolder(DATA_DIR / "test", transform=transform)

# Print dataset info
print("✅ Classes:", train_dataset.classes)
print("✅ Total train samples:", len(train_dataset))
print("✅ Total test samples:", len(test_dataset))

# Print how many images per class in train and test
from collections import Counter
train_counts = Counter([train_dataset.targets[i] for i in range(len(train_dataset))])
test_counts  = Counter([test_dataset.targets[i] for i in range(len(test_dataset))])

print("\n📊 Train distribution:", {train_dataset.classes[k]: v for k, v in train_counts.items()})
print("📊 Test distribution:", {test_dataset.classes[k]: v for k, v in test_counts.items()})
