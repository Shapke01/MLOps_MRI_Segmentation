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
import shutil
from collections import defaultdict


def is_zip_contents_present(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            output_path = os.path.join(extract_to, member)
            if not os.path.exists(output_path):
                return False
    return True


def download_file(url, output_path):
    """
    Download file from URL to a specific path.
    """
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        wget.download(url, out=output_path)
    else:
        print(f"{output_path} already exists.")


def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to a given directory.
    """
    if not is_zip_contents_present(zip_path, extract_to):
        print(f"\nExtracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print(f"{zip_path} already extracted to {extract_to}.")


def extract_tar(tar_path, extract_to):
    """
    Extract a tar file to a given directory.
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


def analyze_segmentations(raw_images_path, instances):
    """
    Analyze segmentations to assign images to categories based on labels.
    """
    categories = defaultdict(list)
    
    for instance in tqdm(instances, desc="Analyzing segmentations"):
        instance_path = os.path.join(raw_images_path, instance)
        mask_path = os.path.join(instance_path, f'{instance}_seg.nii.gz')
        
        try:
            mask = nib.load(mask_path).get_fdata()
            has_label_1 = np.any(mask == 1)  # Tumor necrosis core (NCR)
            has_label_2 = np.any(mask == 2)  # Peritumoral edema (ED)
            has_label_4 = np.any(mask == 4)  # Enhancing tumor (ET)
            
            category = f"{int(has_label_1)}_{int(has_label_2)}_{int(has_label_4)}"
            categories[category].append(instance)
        except Exception as e:
            print(f"Error analyzing {instance}: {e}")
    
    return categories


def custom_train_test_split(data, train_size, random_state=None):
    """
    Simple train/test split with fixed proportions.
    """
    if random_state is not None:
        random.seed(random_state)
    
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    split_idx = int(len(data_copy) * train_size)
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]
    
    return train_data, test_data


def split_data(categories, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split data into training, validation and test sets while preserving label proportions.
    """
    train_instances = []
    val_instances = []
    test_instances = []
    
    for category, instances in categories.items():
        if len(instances) < 3:
            print(f"Category {category} has only {len(instances)} samples - assigning all to training set")
            train_instances.extend(instances)
            continue
            
        train_temp, temp = custom_train_test_split(instances, train_size=train_ratio, random_state=random_state)
        
        val_size_relative = val_ratio / (val_ratio + test_ratio)
        val, test = custom_train_test_split(temp, train_size=val_size_relative, random_state=random_state)
        
        train_instances.extend(train_temp)
        val_instances.extend(val)
        test_instances.extend(test)
    
    return train_instances, val_instances, test_instances


def organize_data(raw_images_path, output_base_path, instances):
    """
    Organize data into appropriate folders (train, valid, test).
    """
    categories = analyze_segmentations(raw_images_path, instances)
    
    print("\nLabel categories (NCR_ED_ET):")
    for category, instances in categories.items():
        print(f"  {category}: {len(instances)} cases")
    
    train_instances, val_instances, test_instances = split_data(categories)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_instances)} samples ({len(train_instances) / len(instances):.1%})")
    print(f"  Validation: {len(val_instances)} samples ({len(val_instances) / len(instances):.1%})")
    print(f"  Test: {len(test_instances)} samples ({len(test_instances) / len(instances):.1%})")
    
    os.makedirs(os.path.join(output_base_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, 'test'), exist_ok=True)
    
    file_types = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz', '_seg.nii.gz']
    
    def copy_instance_files(instance, target_dir):
        source_dir = os.path.join(raw_images_path, instance)
        target_instance_dir = os.path.join(target_dir, instance)
        os.makedirs(target_instance_dir, exist_ok=True)
        
        for file_type in file_types:
            source_file = os.path.join(source_dir, f"{instance}{file_type}")
            target_file = os.path.join(target_instance_dir, f"{instance}{file_type}")
            
            if os.path.exists(source_file) and not os.path.exists(target_file):
                shutil.copy2(source_file, target_file)
    
    print("\nOrganizing files...")
    for instance in tqdm(train_instances, desc="Copying training data"):
        copy_instance_files(instance, os.path.join(output_base_path, 'train'))
        
    for instance in tqdm(val_instances, desc="Copying validation data"):
        copy_instance_files(instance, os.path.join(output_base_path, 'valid'))
        
    for instance in tqdm(test_instances, desc="Copying test data"):
        copy_instance_files(instance, os.path.join(output_base_path, 'test'))


class ToTensor:
    """Convert numpy.ndarray images to PyTorch tensors."""
    def __call__(self, pic):
        if pic.ndim == 3:
            middle_slice_x = pic[pic.shape[0] // 2, :, :]
            middle_slice_y = pic[:, pic.shape[1] // 2, :]
            middle_slice_z = pic[:, :, pic.shape[2] // 2]
            pic = np.stack([middle_slice_x, middle_slice_y, middle_slice_z], axis=0)
        else:
            pic = pic[np.newaxis, ...]
        
        return torch.from_numpy(pic.copy())


class KaggleDataset(Dataset):
    def __init__(self, url, dataset_path, transform=None, organize_splits=True):
        self.url = url
        self.dataset_path = dataset_path
        self.transform = transform
    
        self.zip_path = os.path.join(self.dataset_path, 'dataset.zip')
        self.raw_images_path = os.path.join(self.dataset_path, 'raw_images')
        
        download_file(self.url, self.zip_path)
        extract_zip(self.zip_path, self.dataset_path)
        extract_tar(os.path.join(self.dataset_path, 'BraTS2021_Training_Data.tar'), self.raw_images_path)
        
        self.instances = sorted([d for d in os.listdir(self.raw_images_path) if d.startswith('BraTS')])
        
        if organize_splits:
            organize_data(self.raw_images_path, self.dataset_path, self.instances)
        
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance_path = os.path.join(self.raw_images_path, self.instances[idx])
        img_path = os.path.join(instance_path, f'{self.instances[idx]}_flair.nii.gz')
        mask_path = os.path.join(instance_path, f'{self.instances[idx]}_seg.nii.gz') 

        image = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        image = image.astype('float32')
        mask = mask.astype('float32')
        
        image_tensor = ToTensor()(image)
        mask_tensor = torch.from_numpy(mask.copy())
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, mask_tensor


class BraTSDataLoader:
    def __init__(self, dataset_path, batch_size=8, shuffle=True, transform=None):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        
        self.train_dataset = self._create_dataset('train')
        self.valid_dataset = self._create_dataset('valid')
        self.test_dataset = self._create_dataset('test')
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
    def _create_dataset(self, split):
        split_path = os.path.join(self.dataset_path, split)
        instances = sorted([d for d in os.listdir(split_path) if d.startswith('BraTS')])
        return BraTSSplitDataset(split_path, instances, transform=self.transform)


class BraTSSplitDataset(Dataset):
    def __init__(self, split_path, instances, transform=None):
        self.split_path = split_path
        self.instances = instances
        self.transform = transform
        
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        instance_path = os.path.join(self.split_path, instance)
        img_path = os.path.join(instance_path, f'{instance}_flair.nii.gz')
        mask_path = os.path.join(instance_path, f'{instance}_seg.nii.gz') 

        image = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        image = image.astype('float32')
        mask = mask.astype('float32')
        
        image_tensor = ToTensor()(image)
        mask_tensor = torch.from_numpy(mask.copy())
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, mask_tensor


# Example usage
if __name__ == "__main__":
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = KaggleDataset(
        url='https://www.kaggle.com/api/v1/datasets/download/dschettler8845/brats-2021-task1',
        dataset_path='./data',
        organize_splits=True
    )
    
    data_loader = BraTSDataLoader(
        dataset_path='./data',
        batch_size=2,
        shuffle=True,
        transform=transform
    )
    
    print("\nChecking loaded data:")
    print(f"  Training samples: {len(data_loader.train_dataset)}")
    print(f"  Validation samples: {len(data_loader.valid_dataset)}")
    print(f"  Test samples: {len(data_loader.test_dataset)}")

    train_features, train_labels = next(iter(data_loader.train_loader))
    print(train_labels)