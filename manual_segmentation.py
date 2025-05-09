import napari
import numpy as np # extra
import os # extra
# import glob
from pathlib import Path
from woundcompute import image_analysis as ia
from woundcompute import segmentation as seg
from woundcompute import compute_values as com
from skimage import io
# from skimage.filters import gabor,try_all_threshold,threshold_multiotsu,threshold_triangle,threshold_otsu

def find_yaml_folders(base_path):
    yaml_folders = set()
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.yaml'):
                yaml_folders.add(root)
                break

    return list(yaml_folders)

save_path='via2_masks/'
dataset_selection = int(input("Enter 1 for Viability1 dataset, and 2 for Viability2 dataset: "))
if dataset_selection==1:
    base_path = '/projectnb/lejlab2/quan/wound_healing/wound_compute_env/woundcompute/quan_playground/data/20240909_R7P1_Viability/Sorted2/tissue_ai'
elif dataset_selection==2:
    base_path = '/projectnb/lejlab2/quan/wound_healing/wound_compute_env/woundcompute/quan_playground/data/20240913_R7P2_Viability2/Sorted2/tissue_ai1'
else:
    raise('Invalid dataset number, please run the code again.')

yaml_folders = find_yaml_folders(base_path)
yaml_folders.sort()

for ii,folder in enumerate(yaml_folders):
    print(ii,folder[-50:])

# file_ind = 1
file_ind = int(input("Enter file index: "))

file_name = yaml_folders[file_ind][-8:]
print(f'file_name = {file_name}')
input_file = Path(yaml_folders[file_ind])
input_dict, input_path_dict, output_path_dict = ia.input_info_to_dicts(input_file)
zoom_fcn = com.select_zoom_function(input_dict)
# print(f'zoom_fcn={zoom_fcn}')
input_path = input_path_dict["ph1_images_path"]
output_path = output_path_dict["segment_ph1_path"]
thresh_fcn = seg.select_threshold_function(input_dict, False, False, True)
selection_idx = 4
# print(f'thresh_fcn={thresh_fcn}')

img_list = ia.read_all_tiff(input_path)
img_arr = np.zeros((len(img_list),img_list[0].shape[0],img_list[0].shape[1]))
for ind,img_ in enumerate(img_list):
    img_arr[ind] = img_

continue_seg = 1
while continue_seg==1:
    
    viewer, image_layer = napari.imshow(img_arr)

    segment_path = Path(f'{save_path}{file_name}.tif')
    if segment_path.is_file():
        label_layer_loaded = io.imread(segment_path)
        # list_layers_loaded=[]
        # for ll in label_layer_loaded:
        #     list_layers_loaded.append(ll)
        viewer.add_labels(label_layer_loaded)
    
    napari.run()
    
    label_layer = viewer.layers[-1]
    label_layer_arr = label_layer.data.astype(np.uint8)
    
    label_images = []
    for layer_ind,layer in enumerate(label_layer_arr):
        # if layer_ind%5==4 or layer_ind==0:
        label_images.append(layer)
    label_images = np.array(label_images)
    
    io.imsave(f'{save_path}{file_name}.tif',label_images)
    print(f'successfully saved manually segmented tissue masks for {file_name}')
    
    continue_seg = int(input("Do you want to continue segmentation? (1 for yes, 0 for no): "))
    if continue_seg==0:
        break
    elif continue_seg != 1:
        print('Please input either the integer 1 or 0 for the following question:')
        continue_seg = int(input("Do you want to continue segmentation? (1 for yes, 0 for no): "))
