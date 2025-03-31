import torch
import numpy as np
from dataset import MaskColorMap

def prediction_to_rgb(predicted_mask):
    """takes HxWxC (where C is 1)"""
    img_height, img_width, _ = predicted_mask.shape
    flat_mask = predicted_mask.reshape(-1).astype(int)
    color_lookup = np.array([member.value for member in MaskColorMap], dtype=np.uint8)
    clipped_indices = np.clip(flat_mask, 0, len(MaskColorMap) - 1)
    flat_rgb_image = color_lookup[clipped_indices]
    rgb_image = flat_rgb_image.reshape((img_height, img_width, 3))

    return rgb_image

def mask_to_rgb(ground_truth_mask):
    img_height, img_width, num_classes = ground_truth_mask.shape
    rgb_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    class_indices = np.argmax(ground_truth_mask, axis=2)
    clipped_indices = np.clip(class_indices, 0, num_classes - 1)
    color_lookup = np.array([member.value for member in MaskColorMap], dtype=np.uint8)
    rgb_image = color_lookup[clipped_indices]

    return rgb_image

def evaluate_model(model: torch.nn.Module, 
                   dataloaders: torch.utils.data.DataLoader,
                   metric: torch.nn.Module, 
                   criterion: torch.nn.Module, 
                   device: torch.device):
    model.eval()
    model.to(device)

    running_ious, running_losses = [], []

    for x, y in dataloaders:
    # Send to device (GPU or CPU)
        inputs = x.to(device)
        targets = y.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            # Calculate the loss
            loss = criterion(outputs, targets)
            loss_value = loss.item()
            running_losses.append(loss_value)

            # Calculate the iou
            iou_value = metric(outputs, targets)
            running_ious.append(iou_value.detach().cpu().numpy())
        
    mean_loss = np.mean(running_losses)
    mean_metric = np.mean(running_ious)
        
    return mean_loss, mean_metric
