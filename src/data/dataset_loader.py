import os
import io
import random
import boto3
from PIL import Image
from torch.utils.data import Dataset


class PressureDistributionDatasetS3(Dataset):
    def __init__(self, bucket_name, prefix, classes, transform=None):
        """
        Args:
            bucket_name (str): S3 bucket name.
            prefix (str): S3 folder prefix (e.g. 'train/' or 'val/').
            classes (list): List of class names.
            transform: torchvision transforms to apply.
        """
        self.bucket = bucket_name
        self.prefix = prefix
        self.transform = transform
        self.classes = classes
        self.samples = []
        self.skipped = []
        self.counter = 0
        self.s3 = boto3.client("s3")

        # List all objects under the given prefix
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix)

        if "Contents" not in response:
            print(f"No images found in {bucket_name}/{prefix}")
            return

        for obj in response["Contents"]:
            key = obj["Key"]
            if key.lower().endswith(".jpg"):
                label = self.extract_label(os.path.basename(key))
                if label is not None:
                    label_idx = classes.index(label)
                    self.samples.append((key, label_idx))
                else:
                    self.skipped.append(key)
            else:
                self.counter += 1
                self.skipped.append(key)

        print(f"✅ Loaded {len(self.samples)} valid images from {prefix}")
        print(f"⚠️ Skipped {self.counter} non-JPG files.")

    def extract_label(self, filename):
        """Extract label from filename prefix (e.g. left_001.jpg → left)."""
        filename = filename.lower()
        for c_name in self.classes:
            if filename.startswith(c_name):
                return c_name
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, label = self.samples[idx]

        # Load image directly from S3
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        image = Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def return_random_sample(self):
        idx = random.randint(0, len(self.samples) - 1)
        return self.__getitem__(idx)

    def get_skipped_files(self):
        return self.skipped
