import torch
import numpy as np
from model_loader import feature_extractor, xgb_model, device
from utils.preprocess import preprocess_image


def predict_posture(image_bytes_list):
    embeddings = []
    with torch.no_grad():
        for img_bytes in image_bytes_list:
            img_tensor = preprocess_image(img_bytes).to(device)
            out = feature_extractor(img_tensor)
            out = torch.flatten(out, 1)
            embeddings.append(out.cpu().numpy())
    X = np.vstack(embeddings)
    preds = xgb_model.predict(X)
    return preds.tolist()
