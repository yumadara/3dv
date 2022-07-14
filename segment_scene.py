import os, cv2
import numpy as np
from tqdm import tqdm
from nuscenes_helper.nuscenes import NuScenesHelper
from segmentation.api import CenterNetAPI

## Parameters   ##
scene_filenames = [#"n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg",
                   "n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915452012476.jpg",]
                   #"n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915617012465.jpg",
                   #"n008-2018-07-26-12-13-50-0400__CAM_FRONT__1532621788662404.jpg",
                   #"n008-2018-07-26-12-13-50-0400__CAM_FRONT__1532621914612404.jpg"]
dataset_folder = "./dataset/v1.0-trainval02_blobs"
dataset_version = "v1.0-trainval"


if __name__ == "__main__":
    if dataset_folder[-1] == "/":
        dataset_folder = dataset_folder[:-1]
    out_folder = dataset_folder+"_segmented"
    db_util = NuScenesHelper(dataset_version, dataset_folder)
    engine = CenterNetAPI()
    #os.makedirs(os.path.join(out_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_folder), exist_ok=True)
    for i, file in enumerate(scene_filenames):
        print("Scene", str(i))
        frame_paths = db_util.get_frame_paths_by_scene(file)
        for frame_path in tqdm(frame_paths):
            frame_path = db_util.get_frame_path(frame_path)
            img = cv2.imread(frame_path)
            boxes, masks = engine.run(img)
            #cv2.imwrite(os.path.join(out_folder, "images", frame_path.split("/")[-1]), img)
            npz_file = os.path.join(out_folder, frame_path.split("/")[-1].split(".")[0]+".npz")
            np.savez(npz_file, boxes=boxes, masks=masks)
