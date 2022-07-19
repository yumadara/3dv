import os

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion.quaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, points_in_box

from nuscenes_helper.utils import *


class NuScenesHelper:
    def __init__(self, version, dataroot, verbose=True):
        self.dataroot = os.path.abspath(dataroot)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        self.frontcam_filename_to_scene_token = reverse_indexing_scene_names(self.nusc)
    
    def get_scene_token(self, frontcam_filename):
        return self.frontcam_filename_to_scene_token[frontcam_filename]

    def get_num_scene(self):
        return len(self.nusc.scene)
    
    def get_frame_path(self, filename):
         return os.path.join(self.dataroot, filename)
     
    def extract_cars_from_scene(self, frontcam_filename, visibilities):
        visibilities = [str(i) for i in range(1,visibilities+1)]
        scene_idx = self.nusc._token2ind["scene"][self.get_scene_token(frontcam_filename)]
        scene_meta = self.nusc.scene[scene_idx]
        scene_token = scene_meta["token"]
        scene_name = scene_meta["name"]
        scene_cars = []
        first_sample_token = scene_meta["first_sample_token"]
        current_sample_meta = self.nusc.get("sample", first_sample_token)
        while current_sample_meta is not None:
            sample_token = current_sample_meta["token"]
            cars = self._get_cars_from_sample(current_sample_meta, visibilities)
            scene_cars.append({
                "sample_token":sample_token,
                "cars":cars
            })
            next_token = current_sample_meta["next"]
            if next_token == "":
                current_sample_meta = None
            else:
                current_sample_meta = self.nusc.get("sample", next_token)
        return scene_token, scene_name, scene_cars

    def get_frame_paths_by_scene(self, frontcam_filename):
        scene_token = self.get_scene_token(frontcam_filename)        
        fs = self.nusc.get("scene", scene_token)["first_sample_token"]
        sample = self.nusc.get("sample", fs)
        file_names = []
        while True:
            for cam in sample["data"]:
                if not cam.startswith("CAM"):
                    continue
                sample_data = self.nusc.get("sample_data", sample["data"][cam])
                filename = sample_data["filename"]
                file_names.append(filename)
            if sample["next"] == "":
                break
            sample = self.nusc.get("sample", sample["next"])
        return file_names

    def get_camera_info(self, cam_token):
        data = self.nusc.get("sample_data", cam_token)

        ego_pose_token = data["ego_pose_token"]
        calibrated_sensor_token = data["calibrated_sensor_token"]
        height = data["height"]
        width = data["width"]
        
        calibrated_sensor = self.nusc.get("calibrated_sensor", calibrated_sensor_token)
        clb_translation = calibrated_sensor["translation"]
        clb_rotation = calibrated_sensor["rotation"]
        camera_intrinsic = calibrated_sensor["camera_intrinsic"]
        
        ego_pose = self.nusc.get("ego_pose", ego_pose_token)
        ego_rotation = ego_pose["rotation"]
        ego_translation = ego_pose["translation"]
        
        # Get the calibrated sensor and ego pose record to get the transformation matrices.
        # 1st transformation
        P1 = np.eye(4)
        R1 = quaternion_rotation_matrix(ego_pose['rotation']).T
        t1 = np.array(ego_pose['translation']).reshape(3,1)
        P1[:3,:3] = R1
        P1[:3,-1] = -(R1@t1).reshape(-1)
        # 2nd transformation
        P2 = np.eye(4)
        R2 = quaternion_rotation_matrix(calibrated_sensor['rotation']).T
        t2 = np.array(calibrated_sensor['translation']).reshape(3,1)
        P2[:3,:3] = R2
        P2[:3,-1] = -(R2@t2).reshape(-1)
        P = P2@P1
        
        
        ## colmap format fix
        #P[:3,-1] = -P[:3,:3].T@P[:3,-1]
        R_w2c = P[:3,:3]
        t_w2c = P[:3,-1].reshape(3,1)
        """
        #print(R_w2c, t_w2c)
        P = np.linalg.inv(P)
        R_c2w = P[:3,:3]
        t_c2w = P[:3,-1].reshape(3,1)
        print(R_c2w, t_c2w)
        """
        R_c2w, t_c2w = R_w2c.T, -R_w2c.T@t_w2c
        

        #t_world = -R_w2c@t_w2c
        #[-1.55208194  0.02911663  3.70275983] 8.366715640918061
        # [ 0.55772764 -0.54403343 10.41659385] 5.8048647975004
        #cen = np.array([-1.55208194,  0.02911663,  3.70275983])
        #meddist = 8.366715640918061
        #t_c2w = (t_c2w - cen[:, None]) * 2 / meddist
        P = np.concatenate([np.concatenate([R_c2w, t_c2w], 1), np.array([0, 0, 0, 1.0]).\
            reshape([1, 4])], 0)
        #P[0:3,2] *= -1 # flip the y and z axis
        #P[0:3,1] *= -1
        #P = P[[1,0,2,3],:] # swap y and z
        #P[2,:] *= -1 # flip whole world upside down
        #Rt = np.matmul(so3, -r)                                                                                                                                                                                                                                                                                        
        #P = np.vstack((np.hstack((so3.T, -r.reshape(-1, 1))), [0, 0, 0, 1]))
        ret = {
            "height": height,
            "width": width,
            "ego_pose_token": ego_pose_token,
            "calibrated_sensor_token": calibrated_sensor_token,
            "calibrated_translation":clb_translation,
            "calibrated_rotation": clb_rotation,
            "camera_intrinsic":camera_intrinsic,
            "ego_rotation":ego_rotation,
            "ego_translation":ego_translation,
            "P": P.tolist(),
               }
        return ret

    def get_lidar_info(self, lidar_token, cam_world_to_cam, bounding_box_world):
        data = self.nusc.get("sample_data", lidar_token)

        lidar_filename = "./dataset/v1.0-mini/" + data['filename']#TODO correct how this name is being appended

        #4xn
        lidar_points = LidarPointCloud.from_file(lidar_filename)
        
        #First we transform the points to world coordinate ((lidar->ego) and then (ego->world))
        calibrated_sensor_token = data["calibrated_sensor_token"]
        calibrated_sensor = self.nusc.get("calibrated_sensor", calibrated_sensor_token)
        R1 = quaternion_rotation_matrix(calibrated_sensor['rotation'])
        t1 = np.array(calibrated_sensor['translation'])
        lid_to_ego = np.eye(4)
        lid_to_ego[:3, :3] = R1
        lid_to_ego[:3, -1] = t1

        ego_pose_token = data["ego_pose_token"]
        ego_pose = self.nusc.get("ego_pose", ego_pose_token)
        R2 = quaternion_rotation_matrix(ego_pose['rotation'])
        t2 = np.array(ego_pose['translation'])
        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = R2
        ego_to_world[:3, -1] = t2
        
        lid_to_world = ego_to_world @ lid_to_ego
        
        temp_coords = np.ones_like(lidar_points.points)
        temp_coords[:3, :] = lidar_points.points[:3, :]
        lidar_world = lid_to_world @ temp_coords
        
        
        #Now we select the correct ones
        bb = np.array(bounding_box_world)
        p1 = bb[:, 0]
        p_x = bb[:, 4]
        p_y = bb[:, 1]
        p_z = bb[:, 3]

        i = p_x - p1
        j = p_y - p1
        k = p_z - p1

        v = lidar_world[:3, :] - p1.reshape((-1, 1))

        iv = np.dot(i, v)
        jv = np.dot(j, v)
        kv = np.dot(k, v)

        mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
        mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
        mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
        mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

        lidar_world_in = lidar_world[:, mask]

        #Now we transform them to camera space
        lidar_cam_in = cam_world_to_cam @ lidar_world_in

        #change last coordinate for the intensity
        lidar_world_in[-1, :] = (lidar_points.points[:, mask])[-1, :]
        lidar_cam_in[-1, :] = (lidar_points.points[:, mask])[-1, :]
        lidar_world[-1, :] = lidar_points.points[-1, :]

        #print(lidar_world_in.shape)

        ret = {
            "lidar_to_world": lidar_world.tolist(),
            "lidar_world_in":lidar_world_in.tolist(),
            "lidar_cam_in":lidar_cam_in.tolist(),
            "lidar_world_all":lidar_world.tolist()
        }
        return ret


    def _get_2d_boxes(self, sample_data_token, visibilities):
        """
        Get the 2D annotation records for a given `sample_data_token`.
        :param sample_data_token: Sample data token belonging to a camera keyframe.
        :param visibilities: Visibility filter.
        :return: List of 2D annotation record that belongs to the input `sample_data_token`
        """

        # Get the sample data and the sample corresponding to that sample data.
        sd_rec = self.nusc.get('sample_data', sample_data_token)

        assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
        if not sd_rec['is_key_frame']:
            raise ValueError('The 2D re-projections are available only for keyframes.')

        s_rec = self.nusc.get('sample', sd_rec['sample_token'])

        # Get the calibrated sensor and ego pose record to get the transformation matrices.
        cs_rec = self.nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_rec = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

        # Get all the annotation with the specified visibilties.
        ann_recs = [self.nusc.get('sample_annotation', token) for token in s_rec['anns']]
        ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in visibilities)]

        repro_recs = []

        for ann_rec in ann_recs:
            # Augment sample_annotation with token information.
            ann_rec['sample_annotation_token'] = ann_rec['token']
            ann_rec['sample_data_token'] = sample_data_token

            # Get the box in global coordinates.
            box = self.nusc.get_box(ann_rec['token'])

            # Move them to the ego-pose frame.
            box.translate(-np.array(pose_rec['translation']))
            box.rotate(Quaternion(pose_rec['rotation']).inverse)

            # Move them to the calibrated sensor frame.
            box.translate(-np.array(cs_rec['translation']))
            box.rotate(Quaternion(cs_rec['rotation']).inverse)

            # Filter out the corners that are not in front of the calibrated sensor.
            corners_3d = box.corners()
            in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
            corners_3d = corners_3d[:, in_front]

            # Project 3d box to 2d.
            corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

            # Keep only corners that fall within the image.
            final_coords = post_process_coords(corner_coords)

            # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
            if final_coords is None:
                continue
            else:
                min_x, min_y, max_x, max_y = final_coords

            # Generate dictionary record to be included in the .json file.
            repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec['filename'])
            repro_recs.append(repro_rec)

        return repro_recs

    def _get_cars_from_sample(self, sample_meta, visibilities=["1", "2", "3", "4"]):
        """
        Returns list of cars as tuple in the format of (instance_token, anno_token, 2D bbox)
        for given frame.
        """
        cars = []
        data = sample_meta["data"]# todo get camera parameters and visibility information
        data_lidar_token = data ["LIDAR_TOP"]
        for cam_type, cam_token in data.items():
            if not cam_type.startswith("CAM"):
                continue
            cars_2d = self._get_2d_boxes(cam_token, visibilities=visibilities)
            for car2d in cars_2d:
                category = car2d["category_name"]
                if category.split(".")[1] != "car":
                    continue # ignore the objects that are not vehicle
                cars.append({
                    "anno_token": car2d["sample_annotation_token"],
                    "instance_token": car2d["instance_token"], # unique id of each vehicle,
                    "category": category,
                    "filename":car2d["filename"],
                    "bbox_corners":car2d["bbox_corners"],
                    "visibility":car2d["visibility_token"],
                    "cam_type": cam_type,
                    "cam_token": cam_token,
                    "lidar_token": data_lidar_token
                })
        return cars

