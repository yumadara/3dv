import os, json
import cv2
from nuscene_helper import NuScenesExtractor
import matplotlib.pyplot as plt


if __name__ == "__main__":
    db_util = NuScenesExtractor('v1.0-mini', './dataset/v1.0-mini')
    num_of_scenes = -1 # -1 for all scenes
    out_folder = "MiniNusceneCropped"
    visibility = ["4"]
    size_threshold = 40000
    
    size_hist = []
    if num_of_scenes == -1:
        num_of_scenes = db_util.get_num_scene()
    for i in range(num_of_scenes):
        scene_token, scene_name, samples = db_util.extract_cars_from_scene(i, visibility)
        scene_folder = os.path.join(out_folder, scene_name+"_"+scene_token)
        for sample in samples:
            sample_token = sample["sample_token"]
            cars = sample["cars"]        
            for car in cars:
                car["sample_token"] = sample_token
                car_folder = os.path.join(scene_folder, car["instance_token"])
                im_path = db_util.get_frame_path(car["filename"])
                bbox = list(map(int, car["bbox_corners"]))
                crop = cv2.imread(im_path)[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
                size = crop.shape[0]*crop.shape[1]
                if size > size_threshold:
                    os.makedirs(car_folder, exist_ok=True)
                    size_hist.append(size)
                    cv2.imwrite(os.path.join(car_folder, car["anno_token"]+".png"), crop)
                    camera_info = db_util.get_camera_info(car["cam_token"])
                    car.update(camera_info)
                    with open(os.path.join(car_folder, car["anno_token"]+".json"), "w+") as f:
                        json.dump(car, f)

    #plt.hist(size_hist, bins=100)
    #plt.savefig("hist.png")    