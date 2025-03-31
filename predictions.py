import torch
import numpy as np
import cv2

from helper import prediction_to_rgb, mask_to_rgb
from model import load_model, DEVICE

def display(original_images,
            ground_truth_masks,
            predicted_masks,
            batch_size,
            name='window',
            save=False,

            ):
    # torch.Size([4, 3, 224, 224])                                                                                                                 
    # torch.Size([4, 6, 224, 224])                                                                                                                        
    # torch.Size([4, 1, 224, 224])
    for i in range(batch_size):
        original_image = original_images[i].cpu().numpy().transpose(1, 2, 0)  # CxHxW to HxWxC
        ground_truth_mask = ground_truth_masks[i].cpu().numpy().transpose(1, 2, 0)  # CxHxW to HxWxC
        predicted_mask = predicted_masks[i].cpu().numpy().transpose(1, 2, 0)# CxHxW to HxWxC
        print(original_image.shape)
        print(ground_truth_mask.shape)
        print(predicted_mask.shape)
          
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
            import uuid
            save_path = Path("out") / (uuid.uuid4().hex[:8] + ".png")
            print(save_path)
            cv2.imwrite(save_path.__str__(), concatenated_image)


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
    from dataset import SemanticSegmentationDataset
    from pathlib import Path

    BATCH_SIZE = 4
    dataset_split = Path('dataset_split')
    x = "test"

    dataset_test = SemanticSegmentationDataset(
            sorted((dataset_split / "test" / "images").iterdir()),
            sorted((dataset_split / "test" / "masks").iterdir()),
            transform=None)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    

    
    model = load_model(Path("/home/user/repos/dubai-satellite-segmentation/exps/model_epoch_99"))

    images, masks = next(iter(dataloader_test))
    output = predict_mask(img=images, model=model, device=DEVICE)
    predicted_masks = np.argmax(output.to('cpu'), axis=1, keepdims=True)

    display(original_images=images,
            ground_truth_masks=masks,
            predicted_masks=predicted_masks,
            batch_size=BATCH_SIZE,
            save=True)

