from torchvision.datasets import ImageFolder
from src_images import config
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,RandomAffine,Resize,ToTensor,Normalize,RandomHorizontalFlip,ColorJitter


def dataloader(batch_size,image_size):
    train_transfrom = Compose([
        Resize((image_size, image_size)),
        RandomAffine(
            degrees=10,
            translate=(0.03, 0.03),
            scale=(0.9, 1.1),
            shear=3
        ),
        ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1
        ),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_dataset = ImageFolder(root=config.data_train, transform=train_transfrom)

    val_transfrom = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    val_dataset = ImageFolder(root=config.data_val,transform=val_transfrom)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    test_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    return train_dataloader, test_dataloader

