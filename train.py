import os
import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
import segmentation_models_pytorch as smp
import albumentations as A
from pathlib import Path
import torch

from pathlib import Path

from torch.optim import lr_scheduler

from dataset import *

torch.manual_seed(42)
torch.cuda.manual_seed(42)

NUM_WORKERS = 1

from enum import Enum

if __name__ == '__main__':
    # Calculate means and stds of the trainset and normalize

    dataset_split = Path('dataset_split')
    train_data = torchvision.datasets.ImageFolder(root = dataset_split/'train', transform = transforms.ToTensor())

    means = torch.zeros(3)
    stds = torch.zeros(3)

    for img, label in train_data:
        means += torch.mean(img, dim = (1,2))
        stds += torch.std(img, dim = (1,2))

    means /= len(train_data)
    stds /= len(train_data)
        
    print(f'Calculated means: {means}')
    print(f'Calculated stds: {stds}')

    import albumentations as A
    import albumentations.augmentations.functional as F
    from albumentations.pytorch import ToTensorV2

    data_augmentation = {
        'train': A.Compose([
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=means, std=stds),
        ]),
        'val': A.Compose([
            A.Normalize(mean=means, std=stds),
        ]),
        'test': A.Compose([
            A.Normalize(mean=means, std=stds),
        ]),
    }
    


    BATCH_SIZE = 64
    image_datasets = {x: SemanticSegmentationDataset(image_dir=os.path.join(dataset_split, x, 'images'),
                                                     mask_dir=os.path.join(dataset_split, x, 'masks'), 
                                                     image_names=sorted(os.listdir(os.path.join(dataset_split, x, 'images'))),
                                                     mask_names=sorted(os.listdir(os.path.join(dataset_split, x, 'masks'))),
                                                     transform=None) for x in ['train', 'val', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,shuffle=True, 
                                                  drop_last=True) for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    import segmentation_models_pytorch as smp

    ENCODER = 'efficientnet-b4'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=6, 
        activation=ACTIVATION,
    )

    from torchinfo import summary

    summary(model, 
            input_size=(16, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )

    model = model.to(device)
    metric_UNet = MeanIoU()
    criterion_UNet = MultiDiceLoss()
    optimizer_UNet = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler_UNet = lr_scheduler.StepLR(optimizer_UNet, step_size=7, gamma=0.1)

    # Trainer
    NUM_EPOCHS = 4
    trainer = Trainer(model=model,
                      dataloaders=dataloaders,
                      epochs=NUM_EPOCHS,
                      metric=metric_UNet,
                      criterion=criterion_UNet, 
                      optimizer=optimizer_UNet,
                      scheduler=exp_lr_scheduler_UNet,
                      save_dir='exps',
                      device=device)

    ## Training process
    model_results = trainer.train_model()
