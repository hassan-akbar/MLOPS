from torch.utils.data import Dataset
import os
from PIL import Image
CLASSES = ["left", "right", "supine", "outofbed", "prone"]


class PressureDistributionDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        self.samples = []
        self.skipped = []
        self.counter = 0

        # Folders aren't properly labeled so we need to assign labels based on file name.
        for subdir, _, files in os.walk(root_dir):
            for f in files:
                fpath = os.path.join(subdir, f)
                if f.endswith(".jpg"):

                    label = self.extract_label(f)
                    if label is not None:
                        label_indx = CLASSES.index(label)
                        self.samples.append((fpath, label_indx))
                else:
                    self.counter += 1
                    self.skipped.append(fpath)
        print(f"Skipped {self.counter} non-jpg files.")

    def extract_label(self, filename):
        # Extract label from filename
        filename = filename.lower()
        for c_name in CLASSES:
            if filename.startswith(c_name):
                return c_name
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

    def return_random_sample(self):
        indx = random.randint(0, len(self.samples) - 1)
        return self.__getitem__(indx)

    def get_skipped_files(self):
        return self.skipped
