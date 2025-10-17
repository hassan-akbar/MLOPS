import torch
import torch.nn as nn
from torchvision import models
import joblib
import os
import ast
from utils.s3_utils import download_from_s3

# --- Environment ---
BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME", "posture-model-artifacts-au")
CLASSES = ["left", "right", "supine", "outofbed", "prone"]

EMBED_MODEL_KEY = os.getenv(
    "EMBED_MODEL_KEY", "current/ConvNeXtTiny_Resize_Oversampled.pth"
)
XGB_MODEL_KEY = os.getenv("XGB_MODEL_KEY", "current/xgboost_model.pkl")

LOCAL_EMBED_PATH = "/tmp/ConvNeXtTiny_Resize_Oversampled.pth"
LOCAL_XGB_PATH = "/tmp/xgboost_model.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    # --- Download from S3 ---
    embed_path = download_from_s3(BUCKET_NAME, EMBED_MODEL_KEY, LOCAL_EMBED_PATH)
    xgb_path = download_from_s3(BUCKET_NAME, XGB_MODEL_KEY, LOCAL_XGB_PATH)

    # --- Load ConvNeXtTiny feature extractor ---
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASSES))
    model.load_state_dict(torch.load(embed_path, map_location=device))
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()

    # --- Load XGBoost classifier ---
    xgb_model = joblib.load(xgb_path)

    print("✅ Both models loaded into memory.")
    return feature_extractor, xgb_model, device


# Load once globally
feature_extractor, xgb_model, device = load_models()
