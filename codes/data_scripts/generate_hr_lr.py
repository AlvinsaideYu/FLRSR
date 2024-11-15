import os
import glob
import rasterio
from rasterio.enums import Resampling

def generate_datasets(input_folder, output_folder):
    # Ensure output directories exist
    # os.makedirs(os.path.join(output_folder, 'HR_x2'), exist_ok=True)
    # os.makedirs(os.path.join(output_folder, 'HR_x3'), exist_ok=True)
    # os.makedirs(os.path.join(output_folder, 'HR_x4'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'LR_x2'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'LR_x3'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'LR_x4'), exist_ok=True)
    
    # Get list of all TIF images in the input folder
    img_list = sorted(glob.glob(os.path.join(input_folder, '*.tif')))
    
    for img_path in img_list:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        with rasterio.open(img_path) as src:
            img = src.read()
            height, width = img.shape[1], img.shape[2]
            
            # Generate HR images by resizing the original image
            # hr_x2 = src.read(out_shape=(src.count, height * 2, width * 2), resampling=Resampling.bilinear)
            # hr_x3 = src.read(out_shape=(src.count, height * 3, width * 3), resampling=Resampling.bilinear)
            # hr_x4 = src.read(out_shape=(src.count, height * 4, width * 4), resampling=Resampling.bilinear)
            
            # save_tif(os.path.join(output_folder, 'HR_x2', f'{base_name}_HRx2.tif'), hr_x2, src)
            # save_tif(os.path.join(output_folder, 'HR_x3', f'{base_name}_HRx3.tif'), hr_x3, src)
            # save_tif(os.path.join(output_folder, 'HR_x4', f'{base_name}_HRx4.tif'), hr_x4, src)
            
            # Generate LR images by downscaling the original image
            lr_x2 = src.read(out_shape=(src.count, height // 2, width // 2), resampling=Resampling.bilinear)
            lr_x3 = src.read(out_shape=(src.count, height // 3, width // 3), resampling=Resampling.bilinear)
            lr_x4 = src.read(out_shape=(src.count, height // 4, width // 4), resampling=Resampling.bilinear)
            
            save_tif(os.path.join(output_folder, 'LR_x2', f'{base_name}.tif'), lr_x2, src)
            save_tif(os.path.join(output_folder, 'LR_x3', f'{base_name}.tif'), lr_x3, src)
            save_tif(os.path.join(output_folder, 'LR_x4', f'{base_name}.tif'), lr_x4, src)
            
            print(f'Processed {base_name}')

def save_tif(file_path, img, src):
    profile = src.profile
    profile.update({
        'height': img.shape[1],
        'width': img.shape[2],
        'transform': rasterio.transform.from_bounds(
            *src.bounds, img.shape[2], img.shape[1]
        )
    })
    with rasterio.open(file_path, 'w', **profile) as dst:
        dst.write(img)

if __name__ == '__main__':
    input_folder = r'D:\Wz_Project_Learning\Super_Resolution_Reconstruction\dataset\sentinel2\val\HR'
    output_folder = r'D:\Wz_Project_Learning\Super_Resolution_Reconstruction\dataset\sentinel2\val'
    generate_datasets(input_folder, output_folder)
