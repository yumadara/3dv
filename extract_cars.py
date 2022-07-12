import os, json, tqdm

import cv2
import torch
import numpy as np
import torchvision.ops.boxes as bops

from nuscenes_helper.nuscenes import NuScenesHelper


scene_filenames = ["n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg"]
dataset_folder = "./dataset/v1.0-mini"
dataset_version = "v1.0-mini"
visibility = 4 # 1 (no filter), 2, 3, 4 (only the most visible ones)
iou_thresh = 0.7 # threshold to match instance seg bbox and nuscenes bbox

if __name__ == "__main__":
    db_util = NuScenesHelper(dataset_version, dataset_folder)
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
                mask = masks[max_id]
                camera_info = db_util.get_camera_info(car["cam_token"])
                car.update(camera_info)
                
                P = np.array(car["P"])
                car_box_3d_world = db_util.nusc.get_box(car['anno_token']).corners()
                car_box_3d_cam = np.ones((4,8)) 
                car_box_3d_cam[:3,:] = car_box_3d_world
                car_box_3d_cam = (P@car_box_3d_cam)[:-1,:].tolist()
                car_box_3d_world = car_box_3d_world.tolist()
                car.update({"car_box_3d_world":car_box_3d_world, "car_box_3d_cam":car_box_3d_cam})

                car_box_3d_world = np.array(car_box_3d_world)
                plane_point_pairs = [[0,1], [2,3], [4,5], [6,7]]
                plane_points = []
                for pair1, pair2 in plane_point_pairs:
                    plane_points.append((car_box_3d_world[:,pair1]+car_box_3d_world[:,pair2])/2)
                car.update({"cutting_plane":np.array(plane_points).tolist()})

                os.makedirs(car_folder, exist_ok=True)
                os.makedirs(image_folder, exist_ok=True)
                cv2.imwrite(os.path.join(image_folder, im_path.split("/")[-1]), frame)
                cv2.imwrite(os.path.join(car_folder, car["anno_token"]+".png"), mask.astype("uint8")*255)
                with open(os.path.join(car_folder, car["anno_token"]+".json"), "w+") as f:
                    json.dump(car, f)
