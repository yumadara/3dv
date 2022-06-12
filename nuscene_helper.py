import os

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion.quaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points

from nuscene_utils import generate_record, post_process_coords



class NuScenesExtractor:

    def __init__(self, version, dataroot, verbose=True):
        self.dataroot = os.path.abspath(dataroot)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)

    def get_num_scene(self):
        return len(self.nusc.scene)


    def get_2d_boxes(self, sample_data_token, visibilities):
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


    def extract_cars_from_scene(self, scene_idx, visibilities=["1", "2", "3", "4"]):
        scene_meta = self.nusc.scene[scene_idx]
        scene_token = scene_meta["token"]
        scene_name = scene_meta["name"]
        scene_cars = []
        first_sample_token = scene_meta["first_sample_token"]
        current_sample_meta = self.nusc.get("sample", first_sample_token)
        while current_sample_meta is not None:
            sample_token = current_sample_meta["token"]
            cars = self.get_cars_from_sample(current_sample_meta, visibilities)
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

    def get_cars_from_sample(self, sample_meta, visibilities=["1", "2", "3", "4"]):
        """
        Returns list of cars as tuple in the format of (instance_token, anno_token, 2D bbox)
        for given frame.
        """
        cars = []
        data = sample_meta["data"] # todo get camera parameters and visibility information
        for cam_type, cam_token in data.items():
            if not cam_type.startswith("CAM"):
                continue
            cars_2d = self.get_2d_boxes(cam_token, visibilities=visibilities)
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
                    "cam_token": cam_token
                })
        return cars


    def get_frame_path(self, filename):
         return os.path.join(self.dataroot, filename)


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
        
        ret = {
            "height": height,
            "width": width,
            "ego_pose_token": ego_pose_token,
            "calibrated_sensor_token": calibrated_sensor_token,
            "calibrated_translation":clb_translation,
            "calibrated_rotation":clb_rotation,
            "camera_intrinsic":camera_intrinsic,
            "ego_rotation":ego_rotation,
            "ego_translation":ego_translation
               }
        return ret