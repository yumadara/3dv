import os

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion.quaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points

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
        # ego -> world
        P1 = np.eye(4)
        R1 = quaternion_rotation_matrix(ego_pose['rotation'])
        t1 = np.array(ego_pose['translation'])
        P1[:3,:3] = R1
        P1[:3,-1] = t1
        # camera -> ego
        P2 = np.eye(4)
        R2 = quaternion_rotation_matrix(calibrated_sensor['rotation'])
        t2 = np.array(calibrated_sensor['translation'])
        P2[:3,:3] = R2
        P2[:3,-1] = t2
        # camera -> world
        P_c2w = P1@P2
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
            "P": P_c2w.tolist(),
            "P_w2c": np.linalg.inv(P_c2w).tolist(),
            "channel": data["channel"]
               }
        return ret

    def get_lidar_info(self, lidar_token, cam_world_to_cam, bounding_box_world, N_low_limit, cam_info, sample_token=None):
        data = self.nusc.get("sample_data", lidar_token)

        #4xn
        sample = self.nusc.get('sample', sample_token)
        lidar_points, _ = LidarPointCloud.from_file_multisweep(self.nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP',
                                                                     nsweeps=1)


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

        _, N_in = lidar_world_in.shape
        lidar_lidar_in = lidar_points.points[:, mask]
        lidar_point_in = LidarPointCloud(lidar_lidar_in)

        if N_in >= N_low_limit:
            can_be_used = True
        else:
            can_be_used = False

        """if sample_token is not None and not can_be_used:#last check is to still get a warning
            #from mpl_toolkits.mplot3d import Axes3D
            #import matplotlib.pyplot as plt
            # fig = plt.figure()
            # # Add a 3d axis to the figure
            # ax = fig.add_subplot(111, projection='3d')
            #
            #
            #
            # ax.scatter(lidar_world_in[0, :], lidar_world_in[1, :], lidar_world_in[2, :], color="blue")
            #
            # plane_point_pairs = [[0, 1], [1, 5], [5, 4], [4, 0], [3, 2], [2, 6], [6, 7], [7, 3],
            #                      [2, 1], [3, 0], [4, 7], [6, 5]
            #                      ]
            #
            # for pair1, pair2 in plane_point_pairs:
            #     c1 = bb[:, pair1]
            #     c2 = bb[:, pair2]
            #     ax.plot3D(*zip(c1, c2), color='r')
            #
            # plt.show()
            #
            # fig = plt.figure()
            # # Add a 3d axis to the figure
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(lidar_world[0, :], lidar_world[1, :], lidar_world[2, :], color="green")
            # plane_point_pairs = [[0, 1], [1, 5], [5, 4], [4, 0], [3, 2], [2, 6], [6, 7], [7, 3],
            #                      [2, 1], [3, 0], [4, 7], [6, 5]
            #                      ]
            #
            # for pair1, pair2 in plane_point_pairs:
            #     c1 = bb[:, pair1]
            #     c2 = bb[:, pair2]
            #     ax.plot3D(*zip(c1, c2), color='r')
            #
            # plt.show()
            sample = self.nusc.get('sample', sample_token)
            print("Warning, too few points !!!")
            print(lidar_world_in.shape)

            # self.nusc.render_pointcloud_in_image(sample_token, pointsensor_channel='LIDAR_TOP',
            #                                      camera_channel=cam_info['channel'], dot_size=15,
            #                                      pc_at_sensor_coords=lidar_point_in)
            #self.nusc.render_sample_data(sample['data']['CAM_FRONT'])
            #self.nusc.render_sample_data(sample['data']['LIDAR_TOP'], underlay_map=True)
            #self.nusc.render_sample_data(sample['data']['LIDAR_TOP'], nsweeps=10, underlay_map=True)"""


        ret = {
            "lidar_to_world": lidar_world.tolist(),
            "lidar_world_in":lidar_world_in.tolist(),
            "lidar_cam_in":lidar_cam_in.tolist(),
            "lidar_world_all":lidar_world.tolist()
        }
        return ret, can_be_used

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
        data = sample_meta["data"]
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

