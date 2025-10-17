# 🧠 Posture Detection Inference Service

Serverless inference and training code for a **posture detection pipeline** built with:

- **ConvNeXt-Tiny** as the visual feature extractor (embedding model)
- **XGBoost** as the final posture classifier
  Inference runs as an **AWS Lambda container** that dynamically downloads trained models from **Amazon S3**.

---

## ⚙️ Core Components

### 🧩 Data & Preprocessing

- **`PressureDistributionDataset`** — handles image loading and class labeling.
- **`get_transforms`** — defines resize and normalization transforms used in training and inference.

### 🧩 Inference

- **`lambda_function.py`** — AWS Lambda entry point.

  Accepts:

  - S3 URLs via `"image_urls"`.

- **`model_loader.py`** — downloads models from S3 (ConvNeXt `.pth`, XGBoost `.pkl`), loads them into global memory.
- **`inference_handler.py`** — handles preprocessing, embedding extraction, and classification.

### 🧩 Training

- **`embedding_model.py`** — trains or finetunes the ConvNeXt feature extractor.
- **`classifier.py`** — extracts embeddings and trains the XGBoost posture classifier.
- **`config.py`** — provides data loaders and split configuration.

---

## 🚀 Quickstart (Inference)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r src/inference/requirements.txt
```

### 2. Docker Build & Local Lambda Emulation

```bash
cd src/inference
docker build -t posture-inference .
docker run -p 9000:8080 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e MODEL_BUCKET_NAME=posture-model-artifacts-au \
  posture-inference
```

Invoke locally:

```bash
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{"image_urls": ["https://posture-raw-data-au.s3.ap-southeast-2.amazonaws.com/Task_1_data/VALIDATION/left/left_15527489.jpg"]}'
```

---

## 🔐 Environment Variables

| Variable                                      | Description                               |
| --------------------------------------------- | ----------------------------------------- |
| `MODEL_BUCKET_NAME`                           | S3 bucket containing the trained models   |
| `EMBED_MODEL_KEY`                             | S3 key for ConvNeXt model (.pth)          |
| `XGB_MODEL_KEY`                               | S3 key for XGBoost classifier (.pkl)      |
| `CLASSES`                                     | JSON list of class names                  |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | IAM credentials for S3 access             |
| `DEVICE`                                      | (Optional) force “cpu” for Lambda runtime |

---

## 📊 Training & Artifacts

- The ConvNeXt feature extractor is trained using [`embedding_model.py`](src/training/embedding_model.py).
- Embeddings are saved and fed into the XGBoost classifier trained in [`classifier.py`](src/training/classifier.py).
- Dataloaders from [`config.py`](src/training/config.py) and dataset logic in [`dataset_loader.py`](src/data/dataset_loader.py).
- Resulting artifacts are stored on S3:

  - `ConvNeXtTiny_Resize_Oversampled.pth`
  - `xgboost_model.pkl`

---

## 🧪 Example Payloads

**Base64 Input**

```json
{
  "images": ["<base64_encoded_image>"]
}
```

**S3 URL Input**

```json
{
  "image_urls": [
    "https://posture-raw-data-au.s3.ap-southeast-2.amazonaws.com/Task_1_data/VALIDATION/right/right_18394201.jpg"
  ]
}
```

---

## 🧰 Tech Stack

- **Python 3.11**
- **PyTorch + Torchvision**
- **XGBoost**
- **boto3** for S3 interaction
- **Docker + AWS Lambda Runtime Interface Emulator**
- **AWS ECR / Lambda** for deployment

---

## 🧠 Notes

- All models are cached under `/tmp` inside the Lambda container to minimize cold-start overhead.
- Preprocessing follows `224 × 224` resize and ImageNet normalization.
- The repository is structured for easy migration between **local**, **Docker**, and **serverless** inference.
- Example experiments and model-selection notebooks are in [`src/notebooks/model_selection.ipynb`](src/notebooks/model_selection.ipynb).

---

Would you like me to append a short optional **“CI/CD with GitHub Actions → ECR → Lambda”** section at the end of this README so it’s fully ready for publishing?
