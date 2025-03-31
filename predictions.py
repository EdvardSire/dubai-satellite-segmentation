import torch
import numpy as np
import cv2
from dataset import MaskColorMap, SemanticSegmentationDataset, predict_mask, SatelliteImages
from pathlib import Path
from tqdm import tqdm

from helper import prediction_to_rgb, mask_to_rgb
from model import load_model, DEVICE, BATCH_SIZE

import uuid
rand_out_file = lambda: Path("out") / (uuid.uuid4().hex + ".png")

def display_or_save(original_images,
                    ground_truth_masks,
                    predicted_masks,
                    batch_size,
                    name='window',
                    save=False):
    """
    original_images:    torch.Size([B, 3, 224, 224])                                                                                                                 
    ground_truth_masks: torch.Size([B, 6, 224, 224])
    predicted_masks:    torch.Size([B, 1, 224, 224])
    """

    np.sin

    for i in range(batch_size):
        original_image = original_images[i].cpu().numpy().transpose(1, 2, 0)  # CxHxW to HxWxC
        ground_truth_mask = ground_truth_masks[i].cpu().numpy().transpose(1, 2, 0)  # CxHxW to HxWxC
        predicted_mask = predicted_masks[i].cpu().numpy().transpose(1, 2, 0)# CxHxW to HxWxC
          
        original_image = (original_image * 255).clip(0, 255).astype(np.uint8)
        predicted_image = prediction_to_rgb(predicted_mask)
        ground_truth_image = mask_to_rgb(ground_truth_mask)

        concatenated_image = np.concatenate((original_image,
                                             ground_truth_image,
                                             predicted_image), axis=1)

        if not save:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, concatenated_image[:,:,::-1])  # Reverse color channels for OpenCV (BGR)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else: 
            cv2.imwrite(rand_out_file().__str__(), concatenated_image)


def inference_over_dataset_test_set():
    dataset_root = Path('dataset_split')
    dataset_test = SemanticSegmentationDataset(
            sorted((dataset_root / "test" / "images").iterdir()),
            sorted((dataset_root / "test" / "masks").iterdir()),
            transform=None)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = load_model(Path("/home/user/repos/dubai-satellite-segmentation/exps/model_epoch_99"))

    for _ in range(len(dataloader_test)):
        images, masks = next(iter(dataloader_test))
        print(images.shape) # torch.Size([16, 3, 224, 224])                                                                                                                                                                         

        output = predict_mask(img=images, model=model, device=DEVICE) # pyright: ignore
        predicted_masks = np.argmax(output.to('cpu'), axis=1, keepdims=True)

        display_or_save(original_images=images,
                ground_truth_masks=masks,
                predicted_masks=predicted_masks,
                batch_size=BATCH_SIZE,
                save=True)


if __name__ == '__main__':
    sat_file = Path("/home/user/repos/datasets/UAV_VisLoc_dataset/04/satellite04.tif")
    sat_dataset = SatelliteImages(sat_file)
    sat_dataloader =  torch.utils.data.DataLoader(sat_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = load_model(Path("/home/user/repos/dubai-satellite-segmentation/exps/model_epoch_99"))


    for _ in tqdm(range(len(sat_dataloader))):
        images = next(iter(sat_dataloader))

        output = predict_mask(img=images, model=model, device=DEVICE) # pyright: ignore
        predicted_masks = np.argmax(output.to('cpu'), axis=1, keepdims=True)
        masks = torch.zeros((BATCH_SIZE, len(MaskColorMap), sat_dataset.patch_size, sat_dataset.patch_size))

        display_or_save(original_images=images,
                ground_truth_masks=masks,
                predicted_masks=predicted_masks,
                batch_size=BATCH_SIZE,
                save=True)
