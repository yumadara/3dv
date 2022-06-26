# AT3DCV Novel View Synthesis

## Installation
1. Install the requirements.txt
2. Install [COLMAP](https://colmap.github.io/install.html)

## Car Segmentation
Modify the parameters in segment_scene.py. You should give NuScenes path and its version, and
the scenes you want to segment. To select scenes, just put the name of the first front cam frame.
For example, scene-0061 -> n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg. Also, 
download the [weight file](https://syncandshare.lrz.de/getlink/fiHCQqo2CsCJAjk6YSUt2kEU/centernet2_checkpoint.pth) 
for centernet2. Then run the script. It will create a folder named {dataset_name}_segmented.

## Extract necessary car information from NuScenes
After obtaining the segmented dataset, you should modify the parameters in extract_cars.py in a
similar manner. scene_filenames, dataset_folder, dataset_version must be the same as in segment_scene.py
Running this script will create a new folder named {dataset_name}_processed which contains all necessary
information (frames, masks and camera parameters) for the cars in the given scenes.

## Preparing data for Plenoxels
1. Install COLMAP  (https://colmap.github.io/install.html)
2. Run plenoxels_prepare.py. Give the path of the cars as a list named "processed_car_folders" in the code.
3. Since colmap is randomized process, you can try again running the script if it fails to calibrate at the first time.
4. Use view_data.py for visualising COLMAP output. (Eg. python3 svox2/opt/scripts/view_data.py dataset/v1.0-mini_processed/scene-0061_cc8c0bf57f984915a77078b10eb33198/colmap_out/c1958768d48640948f6053d04cffd35b/)
