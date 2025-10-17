import boto3
import os

s3 = boto3.client("s3")


def download_from_s3(bucket_name, s3_key, local_path):
    """Downloads model file from S3 only if not cached locally."""
    if not os.path.exists(local_path):
        print(f"⬇️ Downloading {s3_key} from S3...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket_name, s3_key, local_path)
        print("✅ Download complete:", local_path)
    else:
        print("⚡ Using cached model:", local_path)
    return local_path
