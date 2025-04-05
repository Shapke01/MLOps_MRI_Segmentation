import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import wget
import zipfile
import tarfile
from tqdm import tqdm
import pathlib
import nibabel as nib
import random
import numpy as np
import torch


def is_zip_contents_present(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            output_path = os.path.join(extract_to, member)
            if not os.path.exists(output_path):
                return False
    return True

def download_file(url, output_path):
    """
    Download a file from a URL to a specified output path.
    """
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        wget.download(url, out=output_path)
    else:
        print(f"{output_path} already exists.")
        
def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to a specified directory.
    """
    if not is_zip_contents_present(zip_path, extract_to):
        print(f"\nExtracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print(f"{zip_path} already extracted to {extract_to}.")
        
def extract_tar(tar_path, extract_to):
    """
    Extract a tar file to a specified directory.
    """
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar_ref:
        with tqdm(total=len(tar_ref.getmembers()), unit='file') as pbar:
            extracted_files = set(pathlib.Path(extract_to).glob('**/*'))
            extracted_files = {str(path.relative_to(extract_to)) for path in extracted_files}
            
            for member in tar_ref.getmembers():
                member_path = pathlib.Path(member.name)
                if str(member_path) not in extracted_files:
                    tar_ref.extract(member, extract_to)
                pbar.update(1)
                
class KaggleDataset(Dataset):
    def __init__(self, url, dataset_path, transform=None):
        """
        Args:
            url (str): URL to the dataset.
            dataset_path (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.url = url
        self.dataset_path = dataset_path
        self.transform = transform
    
        self.zip_path = os.path.join(self.dataset_path, 'dataset.zip')
        self.raw_images_path = os.path.join(self.dataset_path, 'raw_images')
        
        download_file(self.url, self.zip_path)
        extract_zip(self.zip_path, self.dataset_path)
        extract_tar(os.path.join(self.dataset_path, 'BraTS2021_Training_Data.tar'), self.raw_images_path)
        
        self.instances = sorted([d for d in os.listdir(self.raw_images_path) if d.startswith('BraTS')])
        
    def __len__(self):
        return len(os.listdir(self.raw_images_path))
    
    def __getitem__(self, idx):
        instance_path = os.path.join(self.raw_images_path, self.instances[idx])
        img_path = os.path.join(instance_path, f'{self.instances[idx]}_flair.nii.gz')
        mask_path = os.path.join(instance_path, f'{self.instances[idx]}_seg.nii.gz') 

        image = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        image = image.astype('float32')
        mask = mask.astype('float32')
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask
        
# Example usage
if __name__ == "__main__":
    
    # Set random seed for reproducibility
    torch.manual_seed(42)


    # Define transformations
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = KaggleDataset(
        url='https://www.kaggle.com/api/v1/datasets/download/dschettler8845/brats-2021-task1',
        dataset_path='./data',
        # transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Iterate through the dataloader
    for images, labels in dataloader:
        print(images[0].shape)
        print(images[0].max(), images[0].min())
        break