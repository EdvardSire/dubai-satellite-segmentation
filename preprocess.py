#!/usr/bin/env python3

from pathlib import Path
import zipfile
import shutil


def unzip_and_move():
    dataset_zip = Path('./Semantic-segmentation-dataset-1.zip')
    dataset_out_dir = Path('./dataset_raw')
    assert dataset_zip.exists()

    with zipfile.ZipFile(dataset_zip, 'r') as zip_file:
        zip_file.extractall(dataset_out_dir)

    dataset_raw = Path('dataset_raw/Semantic segmentation dataset')
    assert dataset_raw.exists()

    images_path = Path('dataset_raw/images')
    masks_path = Path('dataset_raw/masks')

    images_path.mkdir(exist_ok=True)
    masks_path.mkdir(exist_ok=True)

    for mask_file in dataset_raw.rglob('*.png'):
        tile_name = mask_file.parts[-3].replace(' ', '_').lower()
        shutil.move(str(mask_file), str(masks_path / f'{tile_name}_{mask_file.name}'))

    for image_file in dataset_raw.rglob('*.jpg'):
        tile_name = image_file.parts[-3].replace(' ', '_').lower()
        shutil.move(str(image_file), str(images_path / f'{tile_name}_{image_file.name}'))

    shutil.rmtree(dataset_raw)

    return  dataset_out_dir

def create_patches(dataset_root, patch_size=224, target_dir=Path('./dataset_processed')):
    import os 
    import cv2
    from PIL import Image
    from torchvision import transforms
    import numpy as np

    target_imgs_path, target_masks_path = target_dir / 'images', target_dir / 'masks'
    #target_encoded_path = Path(target_dir+'/'+'encoded_masks/')
    target_dir.mkdir(parents=True, exist_ok=True)
    target_imgs_path.mkdir(parents=True, exist_ok=True)
    target_masks_path.mkdir(parents=True, exist_ok=True)
    #target_encoded_path.mkdir(parents=True, exist_ok=True)

    images_index, masks_index = 0, 0

    for path, _, _ in sorted(os.walk(dataset_root)):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':
            images = sorted(os.listdir(path))
            for _, image_name in enumerate(images):
                if image_name.endswith(".jpg"):
                    image = cv2.imread(path+"/"+image_name)
                    size_X, size_Y = np.ceil(image.shape[1]/patch_size), np.ceil(image.shape[0]/patch_size)
                    pad_X, pad_Y = (patch_size * size_X - image.shape[1]) / (size_X - 1), (patch_size * size_Y - image.shape[0]) / (size_Y - 1)
                    image = Image.fromarray(image)
                    top = 0
                    for _ in range(size_Y.astype(int)):
                        left = 0
                        for _ in range(size_X.astype(int)):
                            crop_image = transforms.functional.crop(image, top, left, patch_size, patch_size)
                            crop_image = np.array(crop_image)
                            cv2.imwrite(f"{target_imgs_path}/image"+str(images_index).zfill(4)+".jpg", crop_image)
                            images_index += 1
                            left = left + patch_size - pad_X
                        top = top + patch_size - pad_Y
        
        if dirname == 'masks':
            images = sorted(os.listdir(path))
            for _, image_name in enumerate(images):
                if image_name.endswith(".png"):
                    #image = cv2.cvtColor(cv2.imread(path+"/"+image_name), cv2.COLOR_BGR2RGB)
                    image = cv2.imread(path+"/"+image_name)
                    size_X, size_Y = np.ceil(image.shape[1]/patch_size), np.ceil(image.shape[0]/patch_size)
                    pad_X, pad_Y = (patch_size * size_X - image.shape[1]) / (size_X - 1), (patch_size * size_Y - image.shape[0]) / (size_Y - 1)
                    image = Image.fromarray(image)
                    top = 0
                    for _ in range(size_Y.astype(int)):
                        left = 0
                        for _ in range(size_X.astype(int)):
                            crop_image = transforms.functional.crop(image, top, left, patch_size, patch_size)
                            crop_image = np.array(crop_image)
                            cv2.imwrite(f"{target_masks_path}/image"+str(masks_index).zfill(4)+".png", crop_image)
                            #encoded_image = one_hot_encode_masks(crop_image, 6)
                            #cv2.imwrite(f"{target_encoded_path}/image"+str(masks_index).zfill(4)+".png", encoded_image)
                            masks_index += 1
                            left = left + patch_size - pad_X
                        top = top + patch_size - pad_Y

    return target_dir

def split_dataset(dataset_processed, out_dir=Path('dataset_split')):
    from split import ratio
    ratio(dataset_processed, output=str(out_dir), ratio=(0.8, 0.1, 0.1))



if __name__ == '__main__':
    dataset_root = unzip_and_move()
    dataset_processed = create_patches(dataset_root=dataset_root)
    shutil.rmtree(dataset_root)
    split_dataset(dataset_processed=dataset_processed)
    shutil.rmtree(dataset_processed)
