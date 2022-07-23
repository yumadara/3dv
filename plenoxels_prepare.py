import os, glob, json, shutil, tqdm, cv2
import numpy as np
from nuscenes_helper.utils import Plane

### Parameters ###
#processed_car_folders = ["dataset/v1.0-mini_processed/scene-0061_cc8c0bf57f984915a77078b10eb33198/61dd7d03d7ad466d89f901ed64e2c0dd"]

processed_car_folders = ["dataset/v1.0-trainval01_blobs_processed/scene-0099_5af9c7f124d84e7e9ac729fafa40ea01/27df1b26d07343a9b2c7b85e3b81b13f"]
#processed_car_folders = glob.glob("dataset/v1.0-mini_processed/scene-0061_cc8c0bf57f984915a77078b10eb33198/*")
use_nuscene_poses = True
augment = True
cam_type = None
extract_lidar=True
mask_png_ext=False
visualize_gr = False
visualize2_gr = False
visualize3_gr = False
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
        os.makedirs(os.path.join(colmap_out_folder, "intrinsics"), exist_ok=True)
        if use_nuscene_poses:
            os.makedirs(os.path.join(colmap_out_folder, "pose"), exist_ok=True)
        if extract_lidar:
            os.makedirs(os.path.join(colmap_out_folder, "lidar"), exist_ok=True)
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
            if mask_png_ext:
                mask_path = mask_path + ".png"
            cv2.imwrite(mask_path, mask.astype("uint8")*255)
            if use_nuscene_poses:
                P = np.array(frame_data["P"])
                camera_intrinsic_ns = np.array(frame_data["camera_intrinsic"])
                camera_intrinsic = np.eye(4)
                camera_intrinsic[:3,:3] = camera_intrinsic_ns
                np.savetxt(os.path.join(colmap_out_folder, "pose", str(i).zfill(5)+".txt"), P)
                np.savetxt(os.path.join(colmap_out_folder, "intrinsics", str(i).zfill(5) + ".txt"), camera_intrinsic)
                if not os.path.exists(os.path.join(colmap_out_folder, "intrinsics.txt")): # single cam?
                    np.savetxt(os.path.join(colmap_out_folder, "intrinsics.txt"), camera_intrinsic)
            if extract_lidar:
                lidar_points = np.array(frame_data['lidar_cam_in'])
                lidar_to_world_matrix = np.array(frame_data['lidar_to_world'])
                np.savetxt(os.path.join(colmap_out_folder, "lidar_to_world.txt"), lidar_to_world_matrix)
                np.savetxt(os.path.join(colmap_out_folder, "lidar", str(i).zfill(5) + ".txt"), lidar_points)
                pass
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
            P_sym_c2w = plane.get_sym_extr(P)
            np.savetxt(pose_path.replace(".txt", "_sym.txt"), P_sym_c2w)
            intr_path = os.path.join(colmap_out_folder, "intrinsics", str(i).zfill(5) + ".txt")
            camera_intrinsic = np.loadtxt(intr_path)
            np.savetxt(intr_path.replace(".txt", "_sym.txt"), camera_intrinsic)#intrinsics do not need conversion
            if extract_lidar:
                #since P sym needs points in world space need to load the points in world space
                lidar_points_w = np.array(frame_data['lidar_world_in'])
                lidar_points_w_coords = np.ones_like(lidar_points_w)
                lidar_points_w_coords[:3, :] = lidar_points_w[:3, :]
                #reflect points on plane
                lidar_points_w_sym = plane.sym_mat @ lidar_points_w_coords
                #Express points in symmetric coordinate system
                lidar_points_c_sym = np.linalg.inv(P_sym_c2w) @ lidar_points_w_sym
                #put back intensity value
                lidar_points_c_sym[-1, :] = lidar_points_w[-1, :]
                np.savetxt(os.path.join(colmap_out_folder, "lidar", str(i).zfill(5) + "_sym.txt"), lidar_points_c_sym)


def run_colmap(processed_car_folders):
    for folder in processed_car_folders:
        if folder[-1] == "/":
            folder = folder[:-1]
        extractor_cmd = f"colmap feature_extractor --database_path={folder}/database.db --image_path={folder}/raw --ImageReader.single_camera=1"
        matcher_cmd = f"colmap exhaustive_matcher --database_path={folder}/database.db"
        mapper_cmd = f"colmap mapper --database_path={folder}/database.db --image_path={folder}/raw --output_path={folder}/sparse"
        os.system(extractor_cmd)
        os.system(matcher_cmd)
        os.makedirs(folder+"/sparse")
        os.system(mapper_cmd)


def visualize():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    for folder in processed_car_folders:
        car_id = folder.split("/")[-1]
        colmap_out_folder = os.path.join("/".join(folder.split("/")[:-1]), "colmap_out", car_id)
        json_paths = sorted(glob.glob(os.path.join(folder, "*.json")))
        for i, json_path in tqdm.tqdm(enumerate(json_paths)):
            with open(json_path, "r") as f:
                frame_data = json.load(f)

            pose_path = os.path.join(colmap_out_folder, "pose", str(i).zfill(5)+".txt")
            plane_points = np.array(frame_data["cutting_plane"])
            plane = Plane(*plane_points[:3].tolist())

            #Load camera
            c2w = np.loadtxt(pose_path).reshape(4, 4) # c2w
            c2w_sym = np.loadtxt(pose_path.replace(".txt","_sym.txt")).reshape(4, 4) # c2w
            w2c = np.linalg.inv(c2w)
            w2c_sym = np.linalg.inv(c2w_sym)
            
            #c2w = P#np.linalg.inv(P)
            cam_loc = c2w @ np.array([0., 0., 0., 1.])
            cam_loc = cam_loc[:3]
            cam_dir_Z = c2w @ np.array([0., 0., 1., 1.])
            cam_dir_Z = cam_dir_Z[:3]
            # load points of bounding box
            bb = np.array(frame_data["car_box_3d_world"])

            #P_sym = plane.get_sym_extr(P)

            #c2w_sym = np.linalg.inv(P_sym)
            cam_loc_sym = c2w_sym @ np.array([0., 0., 0., 1.])
            cam_loc_sym = cam_loc_sym[:3]
            cam_dir_Z_sym = c2w_sym @ np.array([0., 0., 1., 1.])
            cam_dir_Z_sym = cam_dir_Z_sym[:3]


            #load lidar points
            lidar_points_w = np.array(frame_data['lidar_world_in'])
            lidar_points_w_coords = np.ones_like(lidar_points_w)
            lidar_points_w_coords[:3, :] = lidar_points_w[:3, :]

            lidar_points_w_sym = plane.sym_mat @ lidar_points_w_coords

            # ###################################################
            #
            # visualization
            fig = plt.figure()
            # Add a 3d axis to the figure
            ax = fig.add_subplot(111, projection='3d')

            #placing cameras
            # ax.scatter([cam_loc[0], cam_dir_Z[0]],
            #            [cam_loc[1], cam_dir_Z[1]],
            #            [cam_loc[2], cam_dir_Z[2]], color='m')
            # ax.plot3D(*zip(cam_loc, cam_dir_Z), color='m')
            #
            # ax.scatter([cam_loc_sym[0], cam_dir_Z_sym[0]],
            #            [cam_loc_sym[1], cam_dir_Z_sym[1]],
            #            [cam_loc_sym[2], cam_dir_Z_sym[2]], color='g')
            # ax.plot3D(*zip(cam_loc_sym, cam_dir_Z_sym), color='g')

            #normal points
            ax.scatter(lidar_points_w_coords[0, :], lidar_points_w_coords[1, :], lidar_points_w_coords[2, :], color="blue")
            #Symmetric points
            ax.scatter(lidar_points_w_sym[0, :], lidar_points_w_sym[1, :], lidar_points_w_sym[2, :],
                       color="yellow")

            plane_point_pairs = [[0, 1], [1, 5], [5, 4], [4, 0], [3, 2], [2, 6], [6, 7], [7, 3],
                                 [2, 1], [3, 0], [4, 7], [6, 5]
                                 ]

            for pair1, pair2 in plane_point_pairs:
                c1 = bb[:, pair1]
                c2 = bb[:, pair2]
                ax.plot3D(*zip(c1, c2), color='r')

            # ##plane
            # d = -plane.ref_point.dot(plane.normal)
            # xx, yy = np.meshgrid(range(200), range(200))
            # zz = (-plane.normal[0] * xx - plane.normal[1]*yy - d) * 1. /plane.normal[2]
            #
            # # ax.plot_surface(xx, yy, zz)
            ax.set_xlabel('$X$', fontsize=20)
            ax.set_ylabel('$Y$')
            ax.set_zlabel('$Z$')

            plt.show()

            #projection of lidar points to 2D
            camera_intrinsics = np.array(frame_data["camera_intrinsic"])

            temp_coords = w2c @ lidar_points_w_coords
            lidar_points_c_coords_2D = temp_coords[:3, :] / temp_coords[2, :]
            lidar_points_c_coords_2D = camera_intrinsics @ lidar_points_c_coords_2D

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.scatter(lidar_points_c_coords_2D[0, :],
                        lidar_points_c_coords_2D[1, :], color='blue')

            #Now symetric
            temp_coords_sym = w2c_sym @ lidar_points_w_sym
            lidar_points_c_coords_2D_sym = temp_coords_sym[:3, :] / temp_coords_sym[2, :]
            lidar_points_c_coords_2D_sym = camera_intrinsics @ lidar_points_c_coords_2D_sym
            ax2.scatter(lidar_points_c_coords_2D_sym[0, :],
                        lidar_points_c_coords_2D_sym[1, :], color='yellow')
            plt.show()
            _, N = temp_coords_sym.shape
            for j in range(N):
                b = np.isclose(temp_coords[2, j], temp_coords_sym[2, j])
                if not b:
                    print("Points ", temp_coords[:, j], " and ", temp_coords_sym[:, j], "differ in depth")


def visualize2():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    for folder in processed_car_folders:
        car_id = folder.split("/")[-1]
        colmap_out_folder = os.path.join("/".join(folder.split("/")[:-1]), "colmap_out", car_id)
        json_paths = sorted(glob.glob(os.path.join(folder, "*.json")))
        cameras_locs = []
        cameras_locs_sym = []
        cameras_dirs_X = []
        cameras_dirs_Y = []
        cameras_dirs_Z = []
        cameras_dirs_Z_sym = []
        bb_points = []
        for i, json_path in tqdm.tqdm(enumerate(json_paths)):
            with open(json_path, "r") as f:
                frame_data = json.load(f)
            pose_path = os.path.join(colmap_out_folder, "pose", str(i).zfill(5)+".txt")
            plane_points = np.array(frame_data["cutting_plane"])
            plane = Plane(*plane_points[:3].tolist())

            bb = np.array(frame_data["car_box_3d_world"])
            bb_points.append(bb[:, 0])
            bb_points.append(bb[:, 1])
            bb_points.append(bb[:, 2])
            bb_points.append(bb[:, 3])
            bb_points.append(bb[:, 4])
            bb_points.append(bb[:, 5])
            bb_points.append(bb[:, 6])
            bb_points.append(bb[:, 7])


            #Load camera
            c2w = np.loadtxt(pose_path).reshape(4, 4)#c2w
            cam_loc = c2w @ np.array([0., 0., 0., 1.])
            cam_loc = cam_loc[:3]
            cam_dir_X = c2w @ np.array([2., 0., 0., 1.])
            cam_dir_X = cam_dir_X[:3]
            cam_dir_Y = c2w @ np.array([0., 2., 0., 1.])
            cam_dir_Y = cam_dir_Y[:3]
            cam_dir_Z = c2w @ np.array([0., 0., 2., 1.])
            cam_dir_Z = cam_dir_Z[:3]
            cameras_locs.append(cam_loc)
            cameras_dirs_Z.append(cam_dir_Z)
            cameras_dirs_X.append(cam_dir_X)
            cameras_dirs_Y.append(cam_dir_Y)

            c2w_sym = plane.get_sym_extr(c2w)
            cam_loc_sym = c2w_sym @ np.array([0., 0., 0., 1.])
            cam_loc_sym = cam_loc_sym[:3]
            cam_dir_Z_sym = c2w_sym @ np.array([0., 0., 2., 1.])
            cam_dir_Z_sym = cam_dir_Z_sym[:3]
            cameras_locs_sym.append(cam_loc_sym)
            cameras_dirs_Z_sym.append(cam_dir_Z_sym)

        cameras_locs = np.stack(cameras_locs, axis=0)
        cameras_dirs_Z = np.stack(cameras_dirs_Z, axis=0)
        cameras_dirs_X = np.stack(cameras_dirs_X, axis=0)
        cameras_dirs_Y = np.stack(cameras_dirs_Y, axis=0)
        cameras_locs_sym = np.stack(cameras_locs_sym, axis=0)
        cameras_dirs_Z_sym = np.stack(cameras_dirs_Z_sym, axis=0)
        bb_points = np.stack(bb_points, axis=0)


        print(bb_points.mean(axis=0))
        print("______")
        print(bb_points[0])
        print(bb_points[1])
        print(bb_points[2])
        print(bb_points[3])
        print(bb_points[4])
        print(bb_points[5])
        print(bb_points[6])
        print(bb_points[7])


        # visualization
        fig = plt.figure()
        # Add a 3d axis to the figure
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('$X$', fontsize=20)
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')

        #placing cameras
        ax.scatter(bb_points[:, 0], bb_points[:, 1], bb_points[:, 2], color='r')
        ax.scatter(cameras_locs[:, 0], cameras_locs[:, 1], cameras_locs[:, 2], color="b")
        ax.scatter(cameras_locs_sym[:, 0], cameras_locs_sym[:, 1], cameras_locs_sym[:, 2], color="m")
        #ax.scatter(cameras_dirs_Z[:, 0], cameras_dirs_Z[:, 1], cameras_dirs_Z[:, 2], color='y')
        for ti in range(len(cameras_locs)):
            ax.plot3D(*zip(cameras_locs[ti], cameras_dirs_X[ti]), color='r')
            ax.plot3D(*zip(cameras_locs[ti], cameras_dirs_Y[ti]), color='g')
            ax.plot3D(*zip(cameras_locs[ti], cameras_dirs_Z[ti]), color='b')
        for ti in range(len(cameras_locs_sym)):
            ax.plot3D(*zip(cameras_locs_sym[ti], cameras_dirs_Z_sym[ti]), color='g')


        plt.show()

def visualize3():
    import matplotlib.pyplot as plt

    for folder in processed_car_folders:
        car_id = folder.split("/")[-1]
        colmap_out_folder = os.path.join("/".join(folder.split("/")[:-1]), "colmap_out", car_id)
        json_paths = sorted(glob.glob(os.path.join(folder, "*.json")))
        for i, json_path in tqdm.tqdm(enumerate(json_paths)):
            with open(json_path, "r") as f:
                frame_data = json.load(f)

            filename = frame_data["filename"].split("/")[-1]
            img_path = os.path.join("/".join(folder.split("/")[:-1]), "images", filename)
            mask = cv2.imread(json_path.replace(".json", ".png")) > 127
            img = cv2.imread(img_path) * mask + 255 * np.logical_not(mask)
            mask_a = np.array(mask).astype(np.uint8) * 255
            img_a = np.array(img)

            # Load camera
            pose_path = os.path.join(colmap_out_folder, "pose", str(i).zfill(5) + ".txt")
            intr_path = os.path.join(colmap_out_folder, "intrinsics", str(i).zfill(5) + ".txt")
            P = np.loadtxt(pose_path).reshape(4, 4)  # c2w
            intrinsic = np.loadtxt(intr_path)
            #plane_points = np.array(frame_data["cutting_plane"])
            #plane = Plane(*plane_points[:3].tolist())
            #P_sym_c2w = plane.get_sym_extr(P)

            #camera_points
            lidar_path = os.path.join(colmap_out_folder, "lidar", str(i).zfill(5) + ".txt")
            #lidar_points = np.array(frame_data['lidar_cam_in'])
            lidar_points = np.loadtxt(lidar_path).reshape(4, -1)
            lidar_points2D = lidar_points[:3, :] / lidar_points[2:3, :]

            lidar_points2D = intrinsic[:3, :3] @ lidar_points2D
            lidar_points2D = lidar_points2D[:2, :].astype(int)
            plt.imshow(img_a)
            plt.scatter(lidar_points2D[0, :], lidar_points2D[1, :],c='r',s=0.3)
            plt.title(str(i).zfill(5))
            plt.show()

            # lidar_points_w = np.array(frame_data['lidar_world_in'])
            # lidar_points_w_coords = np.ones_like(lidar_points_w)
            # lidar_points_w_coords[:3, :] = lidar_points_w[:3, :]
            # # reflect points on plane
            # lidar_points_w_sym = plane.sym_mat @ lidar_points_w_coords
            # # Express points in symmetric coordinate system
            # lidar_points_c_sym = np.linalg.inv(P_sym_c2w) @ lidar_points_w_sym
            # lidar_points2D_sym = lidar_points_c_sym[:3, :] / lidar_points_c_sym[2:3, :]
            # lidar_points2D_sym = intrinsics[:3, :3] @ lidar_points2D_sym
            # lidar_points2D_sym = lidar_points2D_sym[:2, :].astype(int)
            # img_sym = img[:,::-1,:]
            # plt.imshow(img_sym)
            # plt.scatter(lidar_points2D_sym[0, :], lidar_points2D_sym[1, :], c='r', s=0.3)
            # plt.title(str(i).zfill(5)+"_sym")
            # plt.show()


if __name__ == "__main__":
    if visualize_gr:
        visualize()
        exit()
    if visualize2_gr:
        visualize2()
        exit()
    if visualize3_gr:
        visualize3()
        exit()

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
