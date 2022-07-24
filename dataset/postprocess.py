import os, glob, json
import cv2
import numpy as np
import math

np.random.seed(42)


def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom


if __name__ == "__main__":
    AABB_SCALE = 2
    run_split = True
    #car_folders = glob.glob("*_processed/*/colmap_out/*") 
    car_folders = ["v1.0-mini_processed/scene-0061_cc8c0bf57f984915a77078b10eb33198/static/25/colmap_out/61dd7d03d7ad466d89f901ed64e2c0dd"]
    for car_folder in car_folders:
        # transparency        
        images = glob.glob(os.path.join(car_folder, "images", "*"))
        os.makedirs(os.path.join(car_folder, "transparent_images"))
        for imgp in images:
            im_name = imgp.split("/")[-1].replace(".jpg", ".png")
            image_bgr = cv2.imread(imgp)
            mask_image = cv2.imread(imgp.replace("images","masks"))
            h, w, c = image_bgr.shape
            image_bgra = np.concatenate([image_bgr, np.full((h, w, 1), 255, dtype=np.uint8)], 
                                        axis=-1)
            white = np.all(mask_image != [255, 255, 255], axis=-1)
            image_bgra[white, -1] = 0
            cv2.imwrite(os.path.join(car_folder, "transparent_images", im_name), image_bgra)

        ## split
        if run_split:
            images = glob.glob(os.path.join(car_folder, "images", "*"))
            idx = list(range(len(images)))
            split_idx = int(0.9*len(idx))
            np.random.shuffle(idx)
            train_idx, val_idx = idx[:split_idx], idx[split_idx:]
            for i, imgp in enumerate(images):
                im_name = imgp.split("/")[-1]
                posep = imgp.replace("images", "pose").replace(".jpg", ".txt")
                pose_name = posep.split("/")[-1]
                trans_p = imgp.replace("images", "transparent_images").replace(".jpg", ".png")
                trans_name = trans_p.split("/")[-1]
                if i in train_idx:
                    os.rename(imgp, os.path.join(car_folder, "images", "0_"+im_name))
                    os.rename(posep, os.path.join(car_folder, "pose", "0_"+pose_name))
                    os.rename(trans_p, os.path.join(car_folder, "transparent_images", "0_"+trans_name))
                else:
                    os.rename(imgp, os.path.join(car_folder, "images", "1_"+im_name))
                    os.rename(posep, os.path.join(car_folder, "pose", "1_"+pose_name))
                    os.rename(trans_p, os.path.join(car_folder, "transparent_images", "1_"+trans_name))

        # ngp poses
        os.makedirs(os.path.join(car_folder, "ngp_pose"), exist_ok=True)
        w, h = 1600, 900
        intr_p = os.path.join(car_folder, "intrinsics.txt")
        intr = np.loadtxt(intr_p).reshape(4,4)
        fl_x = intr[0,0]
        fl_y = intr[1,1]
        cx = intr[0,2]
        cy = intr[1,2]
        angle_x = math.atan(w / (fl_x * 2)) * 2
        angle_y = math.atan(h / (fl_y * 2)) * 2
        ngp_pose = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": 0,
            "k2": 0,
            "p1": 0,
            "p2": 0,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aabb_scale": AABB_SCALE,
            "scale": 0.7,
            "frames": [],
        }
        up = np.zeros(3)
        poses = glob.glob(os.path.join(car_folder, "pose", "*"))
        for pose in poses:
            c2w = np.loadtxt(pose)
            im_path = pose.replace("pose","transparent_images").replace(".txt", ".png")
            im_name = im_path.split("/")[-1]
            assert os.path.exists(im_path)
            b = sharpness(im_path)
            c2w[0:3,2] *= -1 # flip the y and z axis
            c2w[0:3,1] *= -1
            c2w = c2w[[1,0,2,3],:] # swap y and z
            c2w[2,:] *= -1 # flip whole world upside down
            up += c2w[0:3,1]
            frame={"file_path":f"transparent_images/{im_name}",
                   "sharpness":b,"transform_matrix": c2w}
            ngp_pose["frames"].append(frame)

        nframes = len(ngp_pose["frames"])
        up = up / np.linalg.norm(up)
        R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        R = np.pad(R,[0,1])
        R[-1, -1] = 1
        for f in ngp_pose["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis
        # find a central point they are all looking at
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in ngp_pose["frames"]:
            mf = f["transform_matrix"][0:3,:]
            for g in ngp_pose["frames"]:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.01:
                    totp += p*w
                    totw += w
        totp /= totw
        for f in ngp_pose["frames"]:
            f["transform_matrix"][0:3,3] -= totp
        avglen = 0.
        for f in ngp_pose["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        for f in ngp_pose["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized
        
        for f in ngp_pose["frames"]:
            pose_path = f["file_path"].replace("transparent_images", "ngp_pose").\
                                       replace(".png", ".txt")
            np.savetxt(os.path.join(car_folder, pose_path), f["transform_matrix"])
        
        for f in ngp_pose["frames"]:
            f["transform_matrix"] = f["transform_matrix"].tolist()
        OUT_PATH = os.path.join(car_folder, "transforms.json")
        with open(OUT_PATH, "w") as outfile:
            json.dump(ngp_pose, outfile, indent=2)
