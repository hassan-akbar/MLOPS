import json
import boto3
from urllib.parse import urlparse
from inference_handler import predict_posture
from model_loader import CLASSES

s3 = boto3.client("s3")

def lambda_handler(event, context):
    try:
        print("📥 Event:", event)

        # Works for API Gateway & direct invoke
        body = json.loads(event["body"]) if "body" in event else event

        if "image_urls" not in body:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'image_urls' field"})
            }

        image_urls = body["image_urls"]
        if isinstance(image_urls, str):  # support single URL
            image_urls = [image_urls]

        image_bytes_list = []

        for url in image_urls:
            print(f"🔹 Processing {url}")
            parsed = urlparse(url)
            host_parts = parsed.netloc.split(".")
            path = parsed.path.lstrip("/")

            # Handle both formats:
            #   1️⃣ bucket.s3.amazonaws.com/key
            #   2️⃣ bucket.s3.region.amazonaws.com/key
            if len(host_parts) >= 3 and host_parts[1] == "s3":
                bucket = host_parts[0]
            elif len(host_parts) >= 4 and host_parts[1] == "s3":
                bucket = host_parts[0]
            else:
                raise ValueError(f"Invalid S3 URL format: {url}")

            print(f"➡️ Bucket: {bucket}, Key: {path}")

            obj = s3.get_object(Bucket=bucket, Key=path)
            image_bytes = obj["Body"].read()
            image_bytes_list.append(image_bytes)

        preds = predict_posture(image_bytes_list)
        decoded_preds = [CLASSES[p] for p in preds]

        return {"statusCode": 200, "body": json.dumps({"predictions": decoded_preds})}

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
