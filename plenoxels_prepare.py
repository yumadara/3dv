import os, glob, json, shutil, tqdm, cv2
import numpy as np
from nuscenes_helper.utils import Plane

### Parameters ###
processed_car_folders = ["dataset/v1.0-mini_processed/scene-0061_cc8c0bf57f984915a77078b10eb33198/61dd7d03d7ad466d89f901ed64e2c0dd"]
use_nuscene_poses = True
augment = True
cam_type = None
##################


def preprocess():
    out = []
    for folder in processed_car_folders:
        car_id = folder.split("/")[-1]
        colmap_out_folder = os.path.join("/".join(folder.split("/")[:-1]), "colmap_out", car_id)
        out.append(colmap_out_folder)
        json_paths = sorted(glob.glob(os.path.join(folder, "*.json")))
        os.makedirs(os.path.join(colmap_out_folder, "raw"), exist_ok=True)
        os.makedirs(os.path.join(colmap_out_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(colmap_out_folder, "masks"), exist_ok=True)
        if use_nuscene_poses:
            os.makedirs(os.path.join(colmap_out_folder, "pose"), exist_ok=True)
        for i, json_path in tqdm.tqdm(enumerate(json_paths)):
            with open(json_path, "r") as f:
                frame_data = json.load(f)
            filename = frame_data["filename"].split("/")[-1]
            frame_cam_type = filename.split("__")[-2]
            if cam_type is not None and frame_cam_type != cam_type:
                continue
            img_path = os.path.join("/".join(folder.split("/")[:-1]), "images", filename)
            ext = filename.split(".")[-1]
            new_path = os.path.join(colmap_out_folder, "raw", str(i).zfill(5)+"."+ext)
            new_path_masked = os.path.join(colmap_out_folder, "images", str(i).zfill(5)+"."+ext)
            mask_path = os.path.join(colmap_out_folder, "masks", str(i).zfill(5)+"."+ext)
            mask = cv2.imread(json_path.replace(".json", ".png")) > 127
            img = cv2.imread(img_path)*mask + 255*np.logical_not(mask)
            shutil.copy2(img_path, new_path)
            cv2.imwrite(new_path_masked, img)
            cv2.imwrite(mask_path, mask.astype("uint8")*255)
            if use_nuscene_poses:
                P = np.array(frame_data["P"])
                camera_intrinsic = np.array(frame_data["camera_intrinsic"])
                np.savetxt(os.path.join(colmap_out_folder, "pose", str(i).zfill(5)+".txt"), P)
                if not os.path.exists(os.path.join(colmap_out_folder, "intrinsics.txt")): # single cam?
                    np.savetxt(os.path.join(colmap_out_folder, "intrinsics.txt"), camera_intrinsic)
    return out

def augment_sym():
    for folder in processed_car_folders:
        car_id = folder.split("/")[-1]
        colmap_out_folder = os.path.join("/".join(folder.split("/")[:-1]), "colmap_out", car_id)
        json_paths = sorted(glob.glob(os.path.join(folder, "*.json")))
        for i, json_path in tqdm.tqdm(enumerate(json_paths)):
            with open(json_path, "r") as f:
                frame_data = json.load(f)
            filename = frame_data["filename"].split("/")[-1]
            img_path = os.path.join("/".join(folder.split("/")[:-1]), "images", filename)
            ext = filename.split(".")[-1]
            new_path_masked = os.path.join(colmap_out_folder, "images", str(i).zfill(5)+"."+ext)
            mask_path = os.path.join(colmap_out_folder, "masks", str(i).zfill(5)+"."+ext)
            mask = cv2.imread(json_path.replace(".json", ".png")) > 127
            img = cv2.imread(img_path)*mask + 255*np.logical_not(mask)
            sym_img = img[:,::-1,:]
            sym_mask = mask[:,::-1,:]
            cv2.imwrite(new_path_masked.replace("."+ext, "_sym."+ext), sym_img)
            cv2.imwrite(mask_path.replace("."+ext, "_sym."+ext), sym_mask.astype("uint8")*255)
            pose_path = os.path.join(colmap_out_folder, "pose", str(i).zfill(5)+".txt")
            P = np.loadtxt(pose_path).reshape(4, 4)
            plane_points = np.array(frame_data["cutting_plane"])
            plane = Plane(*plane_points[:3].tolist())
            P_sym = plane.get_sym_extr(P)
            np.savetxt(pose_path.replace(".txt", "_sym.txt"), P_sym)


def run_colmap(processed_car_folders):
    for folder in processed_car_folders:
        if folder[-1] == "/":
            folder = folder[:-1]
        extractor_cmd = f"colmap feature_extractor --database_path={folder}/database.db --image_path={folder}/raw --ImageReader.single_camera=1"
        matcher_cmd = f"colmap exhaustive_matcher --database_path={folder}/database.db"
        mapper_cmd = f"colmap mapper --database_path={folder}/database.db --image_path={folder}/raw --output_path={folder}/sparse"
        # plenoxels settings (doesn't work good imo)
        #extractor_cmd = f"colmap feature_extractor --database_path={folder}/database.db --image_path={folder}/raw --ImageReader.single_camera=1 --ImageReader.default_focal_length_factor=0.69388 --SiftExtraction.peak_threshold=0.004 --SiftExtraction.edge_threshold=16"
        #matcher_cmd = f"colmap exhaustive_matcher --database_path={folder}/database.db --SiftMatching.max_num_matches=132768"
        #mapper_cmd = f"colmap mapper --database_path={folder}/database.db --image_path={folder}/raw --output_path={folder}/sparse"
        os.system(extractor_cmd)
        os.system(matcher_cmd)
        os.makedirs(folder+"/sparse")
        os.system(mapper_cmd)


if __name__ == "__main__":
    out_folders = preprocess()
    if not use_nuscene_poses:
        run_colmap(out_folders)
        for folder in out_folders:
            sparse_folder = os.path.join(folder, "sparse", "0")
            colmap2nsvf_cmd = f"python3 svox2/opt/scripts/colmap2nsvf.py {sparse_folder}"
            split_cmd = f"python3 svox2/opt/scripts/create_split.py -y {folder}"
            os.system(colmap2nsvf_cmd)
            #os.system(split_cmd)
    if augment:
        augment_sym()
