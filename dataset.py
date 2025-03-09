import os
import torch
from torchvision import transforms
import numpy as np
from skimage.io import imread
from enum import Enum
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import time
from tqdm import trange
from pathlib import Path



class MaskColorMap(Enum):
    Unlabelled = (155, 155, 155)
    Building = (60, 16, 152)
    Land = (132, 41, 246)
    Road = (110, 193, 228)
    Vegetation = (254, 221, 58)
    Water = (226, 169, 41)

class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        return F.nll_loss(y_hat.log(), y.argmax(dim=1))

class MultiDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def dice_coef(self, y_pred, y_true, smooth=0.0001):

        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = torch.sum(y_true_f * y_pred_f)

        return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

    def dice_coef_multiclass(self, y_pred, y_true, numLabels=len(MaskColorMap), smooth=0.0001):    
        dice=0

        for index in range(numLabels):
            dice += self.dice_coef(y_true[:,index,:,:], y_pred[:,index,:,:], smooth = 0.0001)

        return 1 - dice/numLabels

    def forward(self, y_pred, y_true):
        #return self.dice_coef_multiclass(torch.softmax(y_pred, dim=1), y_true)
        return self.dice_coef_multiclass(y_pred, y_true)

class MeanIoU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def IoU_coef(self, y_pred, y_true, smooth=0.0001): 

        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = torch.sum(y_true_f * y_pred_f) 
        total = torch.sum(y_true_f + y_pred_f)
        union = total - intersection 
        
        return (intersection + smooth)/(union + smooth)

    def Mean_IoU(self, y_pred, y_true, numLabels=6, smooth=0.0001):
        IoU_Score=0

        for index in range(numLabels):
            IoU_Score += self.IoU_coef(y_true[:,index,:,:], y_pred[:,index,:,:], smooth = 1)

        return IoU_Score/numLabels

    def forward(self, y_pred, y_true):
        #return self.Mean_IoU(torch.softmax(y_pred, dim=1), y_true)
        return self.Mean_IoU(y_pred, y_true)

    

class SemanticSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, image_names, mask_names, transform=None, mask_transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            image_names (list): List of image names.
            mask_names (list): List of mask names.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.mask_names = mask_names
        self.transform = transform
        self.mask_transform = mask_transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def one_hot_encode_masks(self, masks, num_classes):
        img_height, img_width, _ = masks.shape
        encoded_image = np.zeros((img_height, img_width, 1)).astype(int)
        for j, cls in enumerate(MaskColorMap):
            encoded_image[np.all(masks == cls.value, axis=-1)] = j
        encoded_image = np.reshape(np.eye(num_classes, dtype=int)[encoded_image],(img_height,img_width,num_classes))

        return encoded_image

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_names[idx])

        image = imread(img_name)
        mask = imread(mask_name)

        mask = self.one_hot_encode_masks(mask, len(MaskColorMap))

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        tenn = transforms.ToTensor()
        image = tenn(image)
        mask = tenn(mask)
        
        return image, mask


class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 dataloaders: torch.utils.data.Dataset,
                 epochs: int, 
                 metric: torch.nn.Module, 
                 criterion: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 save_dir: str,
                 device: torch.device):
        
        self.model = model
        self.train_dataloader = dataloaders['train']
        self.val_dataloader = dataloaders['val']
        self.epoch = 0
        self.epochs = epochs
        self.metric = metric
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.device = device

        # Create empty results dictionary
        self.results = {"train_loss": [],
                        "train_iou": [],
                        "val_loss": [],
                        "val_iou": []
                        }
        

    def save_model(self, model: torch.nn.Module, target_dir: str, epoch: int):
        target_dir_path = Path(target_dir)
        target_dir_path.mkdir(parents=True, exist_ok=True)

        check_point_name = f"model_epoch_{epoch}"
        model_save_path = target_dir_path / check_point_name

        torch.save(obj=model.state_dict(), f=model_save_path)

    def train_model(self):
        """
        Train the Model.
        """
        start_time = time.time()

        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar:
            # Epoch counter
            self.epoch += 1
            #progressbar.set_description(f"Epoch {self.epoch}")

            # Training block
            self.train_epoch()
            #progressbar.set_description(f'\nTrain loss: {self.results["train_loss"][-1]} Train iou: {self.results["train_iou"][-1]}')

            # Validation block
            self.val_epoch()
            print(f'\nEpoch {self.epoch}: Train loss: {self.results["train_loss"][-1]} Train iou: {self.results["train_iou"][-1]} Val loss: {self.results["val_loss"][-1]} Val iou: {self.results["val_iou"][-1]}')

            # Save checkpoints every epoch
            self.save_model(self.model, self.save_dir, self.epoch)

        time_elapsed = time.time() - start_time
        print('\n')
        print('-' * 20)
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # plot training curve
        # plot_curve(results=self.results, epochs=self.epochs)

        return self.results

    def train_epoch(self):
        """
        Training Mode
        """
        self.model.train() # training mode
        running_ious, running_losses = [], []

        for x, y in self.train_dataloader:
            # Send to device (GPU or CPU)
            inputs = x.to(self.device)
            targets = y.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward - track history if only in train
            outputs = self.model(inputs)
            # Calculate the loss
            loss = self.criterion(outputs, targets)
            loss_value = loss.item()
            running_losses.append(loss_value)

            # Calculate the iou
            iou = self.metric(outputs, targets)
            iou_value = iou.item()
            running_ious.append(iou_value)

            # Backward pass
            loss.backward()
            # Update the parameters
            self.optimizer.step()

        self.scheduler.step()
        self.results["train_loss"].append(np.mean(running_losses))
        self.results["train_iou"].append(np.mean(running_ious))

    def val_epoch(self):
        """
        Validation Mode
        """
        self.model.eval() # Validation mode
        running_ious, running_losses = [], []

        for x, y in self.val_dataloader:
            # Send to device (GPU or CPU)
            inputs = x.to(self.device)
            targets = y.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                # Calculate the loss
                loss = self.criterion(outputs, targets)
                loss_value = loss.item()
                running_losses.append(loss_value)

                # Calculate the iou
                iou = self.metric(outputs, targets)
                iou_value = iou.item()
                running_ious.append(iou_value)

        self.results["val_loss"].append(np.mean(running_losses))
        self.results["val_iou"].append(np.mean(running_ious))

def predict_mask(img: torch.Tensor, 
            model: torch.nn.Module, 
            device: str):

    model.eval()
    model.to(device)

    x = img.to(device)
    with torch.no_grad():
        out = model(x)

    result = torch.softmax(out, dim=1)

    return result
