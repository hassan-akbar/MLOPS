from sklearn.metrics import classification_report
from dotenv import load_dotenv
from torchvision import models
import torch.nn as nn
import torch
import os
from .config import get_dataloaders
from tqdm import tqdm
import numpy as np
from xgboost import XGBClassifier
import joblib
import ast


load_dotenv()

train_loader, val_loader, test_loader = get_dataloaders()

CLASSES = ast.literal_eval(os.getenv("CLASSES"))


# Load embedding model

model = models.convnext_tiny(weights=None)  # no pretrained weights now
model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASSES))


# 2️⃣ Load saved weights
model.load_state_dict(
    torch.load(
        "/Users/hassan/Documents/Projects/MLOPS/models/ConvNeXtTiny_Resize_Oversampled.pth",
        map_location="cpu",
    )
)

# 3️⃣ Send to device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = model.to(device)
model.eval()

# Step 1: Remove classifier head
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval().to(device)


# Step 2: Extract embeddings
def get_embeddings(loader):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader):
            imgs = imgs.to(device)
            out = feature_extractor(imgs)
            out = torch.flatten(out, 1)
            features.append(out.cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.hstack(labels)


X_train, y_train = get_embeddings(train_loader)
X_val, y_val = get_embeddings(val_loader)
X_test, y_test = get_embeddings(test_loader)


xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="multi:softmax",
    num_class=len(CLASSES),
    tree_method="hist",  # CPU-friendly
    n_jobs=-1,
    random_state=42,
)

xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_val)

print("📈 XGBoost Results:")
print(classification_report(y_val, xgb_preds, target_names=CLASSES))

joblib.dump(xgb, "xgboost_model.pkl")
