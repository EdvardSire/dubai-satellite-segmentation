import torch
import numpy as np
import cv2

from dataset import MaskColorMap

def prediction_to_rgb(predicted_mask):
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


def display(original_image, ground_truth_mask, predicted_mask, name='window'):
    original_image = original_image.cpu().numpy().transpose(1, 2, 0)  # CxHxW to HxWxC
    ground_truth_mask = ground_truth_mask.cpu().numpy().transpose(1, 2, 0)  # CxHxW to HxWxC
    predicted_mask = predicted_mask.cpu().numpy()

      
    original_image = (original_image * 255).clip(0, 255).astype(np.uint8)
    predicted_image = prediction_to_rgb(predicted_mask)
    ground_truth_image = mask_to_rgb(ground_truth_mask)

    concatenated_image = np.concatenate((original_image, ground_truth_image, predicted_image), axis=1)

    print(original_image.dtype)
    print(ground_truth_image.dtype)
    print(predicted_image.dtype)

    # Display the concatenated image using OpenCV
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, concatenated_image[:,:,::-1])  # Reverse color channels for OpenCV (BGR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

if __name__ == '__main__':
    from dataset import MeanIoU, MultiDiceLoss, SemanticSegmentationDataset
    import os
    from torch.optim import lr_scheduler
    import torch.optim as optim
    from pathlib import Path

    BATCH_SIZE = 4
    dataset_split = Path('dataset_split')
    image_datasets = {x: SemanticSegmentationDataset(image_dir=os.path.join(dataset_split, x, 'images'),
                                                     mask_dir=os.path.join(dataset_split, x, 'masks'), 
                                                     image_names=sorted(os.listdir(os.path.join(dataset_split, x, 'images'))),
                                                     mask_names=sorted(os.listdir(os.path.join(dataset_split, x, 'masks'))),
                                                     transform=None) for x in ['train', 'val', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,shuffle=True, 
                                                  drop_last=True) for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    images, masks = next(iter(dataloaders['test']))
    images, masks = next(iter(dataloaders['test']))


    ENCODER = 'efficientnet-b4'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation

    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=6, 
        activation=ACTIVATION,
    )
    model_state = torch.load('/home/user/repos/exp-Semantic-Segmentation-of-Aerial-Imagery/exps/model_epoch_4')
    model.load_state_dict(model_state)
    ## Model inItialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # metric_DeepLab_V3 = MeanIoU()
    # criterion_DeepLab_V3 = MultiDiceLoss()
    # optimizer_DeepLab_V3 = optim.Adam(model.parameters(), lr=0.001)
    # exp_lr_scheduler_DeepLab_V3 = lr_scheduler.StepLR(optimizer_DeepLab_V3, step_size=7, gamma=0.1)
    # outputs = evaluate_model(model=model, dataloaders=dataloaders['val'], 
    #                        metric=metric_DeepLab_V3, criterion=criterion_DeepLab_V3,
    #                        device=device)
    for idx in range(10):
        res = predict_mask(img=images, model=model, device=device)
        predicted_mask = np.transpose(np.argmax(res[idx].to('cpu'), axis=0, keepdims=True))
        display(original_image=images[idx], ground_truth_mask=masks[idx], predicted_mask=predicted_mask)

