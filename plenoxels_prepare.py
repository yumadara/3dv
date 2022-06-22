import os, glob, json, shutil, tqdm

processed_car_folders = [
    "dataset/v1.0-mini_processed/scene-0061_cc8c0bf57f984915a77078b10eb33198/c1958768d48640948f6053d04cffd35b",
    "dataset/v1.0-mini_processed/scene-0061_cc8c0bf57f984915a77078b10eb33198/61dd7d03d7ad466d89f901ed64e2c0dd"
    ]


# todo resize
def preprocess(folders):
    out = []
    for folder in folders:
        car_id = folder.split("/")[-1]
        colmap_out_folder = os.path.join("/".join(folder.split("/")[:-1]), "colmap_out", car_id)    
        out.append(colmap_out_folder)
        json_paths = sorted(glob.glob(os.path.join(folder, "*.json")))
        for i, json_path in tqdm.tqdm(enumerate(json_paths)):
            with open(json_path, "r") as f:
                frame_data = json.load(f)
            filename = frame_data["filename"].split("/")[-1]
            img_path = os.path.join("/".join(folder.split("/")[:-1]), "images", filename)
            ext = filename.split(".")[-1]
            new_path = os.path.join(colmap_out_folder, "images", str(i).zfill(5)+"."+ext)
            os.makedirs(os.path.join(colmap_out_folder, "images"), exist_ok=True)
            shutil.copy2(img_path, new_path)
    return out


def run_colmap(folders):
    # todo
    """
    colmap feature_extractor \
        --database_path=database.db \
        --image_path=images \
        --ImageReader.single_camera=1 \
        --ImageReader.default_focal_length_factor=0.69388 \
        --SiftExtraction.peak_threshold=0.004 \
        --SiftExtraction.edge_threshold=16
        # also possible to give intrinsics

    colmap exhaustive_matcher \
        --database_path=database.db \
        --SiftMatching.max_num_matches=132768

    """
    for folder in folders:
        if folder[-1] == "/":
            folder = folder[:-1]
        extractor_cmd = f"colmap feature_extractor --database_path={folder}/database.db --image_path={folder}/images --ImageReader.single_camera=1"
        matcher_cmd = f"colmap exhaustive_matcher --database_path={folder}/database.db"
        mapper_cmd = f"colmap mapper --database_path={folder}/database.db --image_path={folder}/images --output_path={folder}/sparse"
        os.system(extractor_cmd)
        os.system(matcher_cmd)
        os.makedirs(folder+"/sparse")
        os.system(mapper_cmd)

def train_plenoxels():
    pass

if __name__ == "__main__":
    out_folders = preprocess(processed_car_folders)
    run_colmap(out_folders)
    for folder in out_folders:
        sparse_folder = os.path.join(folder, "sparse", "0")
        colmap2nsvf_cmd = f"python3 svox2/opt/scripts/colmap2nsvf.py {sparse_folder}"
        split_cmd = f"python3 svox2/opt/scripts/create_split.py -y {folder}"
        os.system(colmap2nsvf_cmd)
        os.system(split_cmd)
    print("done\n")