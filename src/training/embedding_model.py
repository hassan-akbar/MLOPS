import mlflow
from sklearn.metrics import classification_report
from dotenv import load_dotenv
from torchvision import models
import torch.nn as nn
import torch
import os
from .config import get_dataloaders
from data.preprocessing import get_transforms
from tqdm import tqdm
import ast


load_dotenv()

train_loader, _, val_loader, test_loader = get_dataloaders()

CLASSES = ast.literal_eval(os.getenv("CLASSES"))


# --- Model setup ---
model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASSES))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
print("✅ Using device:", device)

# --- Training config ---
num_epochs = 1
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=1, eta_min=1e-6
)

# --- Progressive resizing schedule ---
size_schedule = {0: 224, 10: 256, 15: 288}


mlflow.set_experiment("Pressure_Posture_Detection")

with mlflow.start_run(run_name="ConvNeXtTiny_Resize-Oversampled"):
    mlflow.log_params(
        {
            "model": "ConvNeXtTiny_Resize",
            "epochs": num_epochs,
            "lr": 1e-5,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingWarmRestarts",
            "size_schedule": str(size_schedule),
        }
    )

    for epoch in range(num_epochs):

        # --- Dynamically change input size ---
        if epoch in size_schedule:
            new_size = size_schedule[epoch]
            print(f"\n🔁 Changing input size to {new_size}x{new_size}")

            train_trfm, val_trfm = get_transforms(new_size)
            train_loader.dataset.transform = train_trfm
            val_loader.dataset.transform = val_trfm

        print(f"\n🚀 Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        # --- Training ---
        for imgs, labels in tqdm(train_loader, desc="Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + len(train_loader))
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        val_preds, val_true = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validating"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        report = classification_report(
            val_true, val_preds, target_names=CLASSES, output_dict=True
        )
        macro_f1 = report["macro avg"]["f1-score"]

        print(
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Macro F1: {macro_f1:.4f}"
        )

    # --- Final Evaluation ---
    final_report = classification_report(
        val_true, val_preds, target_names=CLASSES, output_dict=True
    )
    mlflow.log_metrics(
        {
            "final_val_accuracy": round(val_acc, 4),
            "final_val_macro_f1": round(macro_f1, 4),
        }
    )

    # Optional: Log classification report as HTML
    # log_classification_report_html(final_report, model_name="ConvNeXtTiny_Resize")

    # --- Save model ---
    model_path = "ConvNeXtTiny_Resize_Oversampled.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(model_path)
