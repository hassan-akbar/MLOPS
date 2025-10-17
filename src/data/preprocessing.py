from torchvision import transforms


def get_transforms(img_size=224):
    train_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.RandomApply([transforms.RandomAdjustSharpness(2)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15),  # moved to the end
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_tfms, val_tfms
