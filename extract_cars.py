import math
import os, json, tqdm, glob
import shutil

import cv2
import torch
import numpy as np
import torchvision.ops.boxes as bops

from nuscenes_helper.nuscenes import NuScenesHelper
from nuscenes_helper.utils import find_count_category_name


scene_filenames = ["n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg",]
#scene_filenames = ["n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg",]
                   #"n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg",
                   #"n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385092112404.jpg"] 
# "n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915452012476.jpg"

dataset_folder = "./dataset/v1.0-mini" # "./dataset/v1.0-trainval02_blobs"
dataset_version = "v1.0-mini" # "v1.0-trainval" 
visibility = 4 # 1 (no filter), 2, 3, 4 (only the most visible ones)
iou_thresh = 0.7 # threshold to match instance seg bbox and nuscenes bbox
extract_lidar = True
num_of_view_splits = [5, 15, 25, 35] # cars are categorized as 10-, 10-20, 20-30, ... 50+
min_size_filter = 0.002 # percentage of the image size, min 10% (0.1) means min 10 pixels in a 10x10 image
max_displ_limit = 0.7 #maximum displacement allowed to determine if a car is moving or not
lidar_low_limit = 5  #minimum amount of lidar points for the car
min_vis_mask_reg = 0.45#How much of the 2D box where the mask is should be occupied by car; (0.7 is too much)

if __name__ == "__main__":
    db_util = NuScenesHelper(dataset_version, dataset_folder)
    processed_cars = []
    processed_cars_box_world_centers = {}
    if dataset_folder[-1] == "/":
        dataset_folder = dataset_folder[:-1]
    out_folder = dataset_folder+"_processed"
    segment_folder = dataset_folder+"_segmented"
    assert os.path.exists(segment_folder), "First you need to run segment_scene.py"
    for filename in scene_filenames:
        scene_token, scene_name, samples = db_util.extract_cars_from_scene(filename, visibility)
        scene_folder = os.path.join(out_folder, scene_name+"_"+scene_token)
        print(scene_name)
        for sample in tqdm.tqdm(samples):
            sample_token = sample["sample_token"]
            cars = sample["cars"]
            image_folder = os.path.join(scene_folder, "images")
            for car in cars:
                car["sample_token"] = sample_token
                car_folder = os.path.join(scene_folder, car["instance_token"])
                im_path = db_util.get_frame_path(car["filename"])
                frame = cv2.imread(im_path)
                det = np.load(os.path.join(segment_folder, im_path.split("/")[-1].split(".")[0]+".npz"))
                masks = det["masks"]
                boxes = det["boxes"]
                car_box = car["bbox_corners"]
                ious = bops.box_iou(torch.tensor([car_box]), torch.tensor(boxes))[0]
                if len(ious)==0:
                    continue
                max_id = ious.argmax()
                if ious[max_id]<iou_thresh:
                    continue
                #check that is big enough
                mask = masks[max_id]
                car_size = mask.sum()/(mask.shape[0]*mask.shape[1])
                if min_size_filter is not None and car_size < min_size_filter:
                    continue
                #check that is not too occluded
                bb = boxes[max_id]
                w_s, w_e = int(math.floor(bb[0])), int(math.floor(bb[2]))
                h_s, h_e = int(math.floor(bb[1])), int(math.floor(bb[3]))
                mask_car_sec = mask[h_s:h_e, w_s:w_e]
                car_size_sec = mask_car_sec.sum()/(mask_car_sec.shape[0]*mask_car_sec.shape[1])
                if min_vis_mask_reg is not None and car_size_sec < min_vis_mask_reg:
                    continue

                camera_info = db_util.get_camera_info(car["cam_token"])
                car.update(camera_info)
                P_w2c = np.array(car["P_w2c"])
                car_BOX = db_util.nusc.get_box(car['anno_token'])
                car_box_3d_world = car_BOX.corners()
                car_box_center_world = car_BOX.center
                car_box_3d_cam = np.ones((4,8))
                car_box_3d_cam[:3,:] = car_box_3d_world
                car_box_3d_cam = (P_w2c@car_box_3d_cam)[:-1,:].tolist()
                car_box_3d_world = car_box_3d_world.tolist()
                car.update({"car_box_3d_world":car_box_3d_world, "car_box_3d_cam":car_box_3d_cam})

                if extract_lidar:
                    lidar_info, can_be_used = db_util.get_lidar_info(car["lidar_token"],
                                                        cam_world_to_cam=P_w2c,
                                                        bounding_box_world=car_box_3d_world,
                                                        cam_info=camera_info,
                                                        N_low_limit=lidar_low_limit,
                                                        sample_token=sample_token)
                    car.update(lidar_info)
                    if not can_be_used:
                        continue

                car_box_3d_world = np.array(car_box_3d_world)
                plane_point_pairs = [[0,1], [2,3], [4,5], [6,7]]
                plane_points = []
                for pair1, pair2 in plane_point_pairs:
                    plane_points.append((car_box_3d_world[:,pair1]+car_box_3d_world[:,pair2])/2)
                car.update({"cutting_plane":np.array(plane_points).tolist()})
                processed_cars.append(car_folder)
                if car_folder not in processed_cars_box_world_centers:
                    processed_cars_box_world_centers[car_folder] = []
                processed_cars_box_world_centers[car_folder].append(car_box_center_world)
                os.makedirs(car_folder, exist_ok=True)
                os.makedirs(image_folder, exist_ok=True)
                cv2.imwrite(os.path.join(image_folder, im_path.split("/")[-1]), frame)
                cv2.imwrite(os.path.join(car_folder, car["anno_token"]+".png"), mask.astype("uint8")*255)
                cv2.imwrite(os.path.join(car_folder, car["anno_token"]+"_im.png"), frame)
                with open(os.path.join(car_folder, car["anno_token"]+".json"), "w+") as f:
                    json.dump(car, f)

    # select statis
    # categorize cars
    processed_cars = list(set(processed_cars))
    static_cars_folder = []
    for car_folder in processed_cars:
        centers = processed_cars_box_world_centers[car_folder]
        if len(centers) == 1: #there cannot be a displacement
            max_displ = 0.
        else:
            displs = np.array([np.linalg.norm(ov - centers[0]) for ov in centers[1:]])
            max_displ = displs.max()
        # print(car_folder.split("/")[-1])
        # print("max displ: ", max_displ)
        # print("___________")
        if max_displ > max_displ_limit:
            new_folder = "/".join(car_folder.split("/")[:-1]) + "/moving"
        else:
            car_f_split = car_folder.split("/")
            new_folder = "/".join(car_f_split[:-1]) + "/static"
            static_cars_folder.append(new_folder+"/"+car_f_split[-1])
        os.makedirs(new_folder, exist_ok=True)
        shutil.move(car_folder, new_folder)

    print(f"From all processed cars {len(static_cars_folder)/len(processed_cars)} ({len(static_cars_folder)}/{len(processed_cars)}) are static")

    # categorize cars
    for car_folder in static_cars_folder:
        num_views = len(glob.glob(os.path.join(car_folder, "*.json")))
        cat_name = find_count_category_name(num_of_view_splits, num_views)
        new_folder = "/".join(car_folder.split("/")[:-1])+f"/{cat_name}"
        os.makedirs(new_folder, exist_ok=True)
        shutil.move(car_folder, new_folder)
