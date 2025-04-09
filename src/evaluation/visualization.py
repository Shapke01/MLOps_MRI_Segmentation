import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image

def makeGIF(scan, mask):
    """
    Create a 3D GIF from the scan and mask data.
    """
    scan = np.array(scan)
    scan = 255 * (scan - scan.min()) / (scan.ptp() + 1e-8)
    scan = scan.astype(np.uint8)
    scan = np.transpose(scan, (0, 2, 1))
    scan = np.flip(scan, axis=1)
    scan = np.stack((scan, scan, scan), axis=-1)
    
    mask = np.array(mask)
    mask = mask.astype(np.uint8)
    mask = np.transpose(mask, (0, 2, 1))
    mask = np.flip(mask, axis=1)
    
    images = []
    
    for idx in range(scan.shape[0]):
        if scan[idx, :, :].max() == 0:
            continue
        
        slice = scan[idx, :, :]
        mask_slice = mask[idx, :, :]
        
        red_mask = (mask_slice == 1)
        green_mask = (mask_slice == 2)
        blue_mask = (mask_slice == 4)
                        
        slice[red_mask] = [255, 0, 0]
        slice[green_mask] = [0, 255, 0]
        slice[blue_mask] = [0, 0, 255]
               
        image = Image.fromarray(slice)
        images.append(image)
        # image.save(f"/app/data/tmp/plot{idx}_.jpg")  
    images[0].save("/app/data/tmp/animation.gif",save_all=True,append_images=images[1:],duration=50,loop=0)   
       
def plot_slices(scan, mask):
    """
    Plot slices of the scan and mask data.
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    
    non_zero_indices = np.any(scan != 0, axis=(1, 2))
    non_zero_slices = scan[non_zero_indices]
    non_zero_mask = mask[non_zero_indices]
    
    random_indices = np.random.choice(len(non_zero_slices), size=5, replace=False)
    random_indices.sort()
    
    for i, idx in enumerate(random_indices):
        axes[0][i].imshow(non_zero_slices[idx], cmap="gray")
        axes[0][i].set_title(f"Scan Slice {idx}")
        axes[1][i].imshow(non_zero_mask[idx], cmap="viridis")
        axes[1][i].set_title(f"Mask Slice {idx}")
        plt.show()
    plt.savefig("/app/data/tmp/plot.png")
    
    
        
if __name__ == "__main__":
    scan = nib.load("/app/data/raw_images/BraTS2021_00000/BraTS2021_00000_flair.nii.gz").get_fdata()
    mask = nib.load("/app/data/raw_images/BraTS2021_00000/BraTS2021_00000_seg.nii.gz").get_fdata()
    
    # makeGIF(scan, mask)
    plot_slices(scan, mask)