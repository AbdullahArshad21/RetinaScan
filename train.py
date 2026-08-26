import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import numpy as np
import pandas as pd

from dataset import RetinaDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = RetinaDataset("data/train_split.csv", "data/train_images", train=True)
    val_dataset = RetinaDataset("data/val_split.csv", "data/train_images", train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    train_df = pd.read_csv("data/train_split.csv")
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2, 3, 4]),
        y=train_df["diagnosis"].values,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")

    # ---- Stronger backbone: EfficientNet-B0 instead of ResNet18 ----
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 5)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    NUM_EPOCHS = 20
    best_macro_recall = 0.0
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = 100 * val_correct / val_total

        # Macro recall = average recall across all 5 classes, treating each
        # class equally regardless of how many examples it has. This is what
        # we actually care about: catching Severe/Proliferative cases matters
        # just as much as catching No DR cases, even though there are far
        # fewer of them in the data.
        macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Acc: {val_accuracy:.2f}% | "
            f"Macro Recall: {macro_recall:.4f}"
        )

        if macro_recall > best_macro_recall:
            best_macro_recall = macro_recall
            torch.save(model.state_dict(), "retina_model_best.pth")
            print(f"  -> New best model saved (Macro Recall: {macro_recall:.4f})")

            report = classification_report(
                all_labels, all_preds, target_names=class_names, zero_division=0
            )
            with open("best_model_report.txt", "w") as f:
                f.write(f"Epoch {epoch+1} | Val Accuracy: {val_accuracy:.2f}% | Macro Recall: {macro_recall:.4f}\n\n")
                f.write(report)
                f.write("\n\nConfusion Matrix (rows=actual, cols=predicted):\n")
                f.write(str(confusion_matrix(all_labels, all_preds)))

        scheduler.step()

    print(f"\nTraining complete. Best macro recall: {best_macro_recall:.4f}")
    print("Best model saved as retina_model_best.pth")
    print("Per-class breakdown saved to best_model_report.txt")


if __name__ == "__main__":
    main()