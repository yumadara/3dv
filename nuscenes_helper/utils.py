# nuScenes dev-kit.
# Code written by Sergi Adipraja Widjaja, 2019.

"""
Export 2D annotations (xmin, ymin, xmax, ymax) from re-projections of our annotated 3D bounding boxes to a .json file.

Note: Projecting tight 3d boxes to 2d generally leads to non-tight boxes.
      Furthermore it is non-trivial to determine whether a box falls into the image, rather than behind or around it.
      Finally some of the objects may be occluded by other objects, in particular when the lidar can see them, but the
      cameras cannot.
"""

from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
from shapely.geometry import MultiPoint, box


class Plane:
    def __init__(self, p1, p2, p3):
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        v1 = p2 - p3
        v2 = p2 - p1
        normal = np.cross(v1, v2)
        self.ref_point = p1
        self.normal = normal/np.linalg.norm(normal)
        self.k = self.normal@p1
        self.sym_mat = self.get_sym_mat()
        print("Normal:", normal)
        print("k:", self.k)
        print("S:", self.sym_mat)
        print()
    
    def angle(self, vec):
        vec = np.array(vec)
        # angle with the normal of plane
        cos_theta = (vec @ self.normal) / np.linalg.norm(vec)
        return np.arccos(cos_theta)

    def dist(self, point):
        # dist from point to plane
        point = np.array(point)
        return np.abs(point @ self.normal + self.k) 
    
    def get_sym_mat(self):
        n = self.normal
        S_R = np.eye(3)-2*np.outer(n,n)
        S_t = 2*self.k*n
        S = np.eye(4)
        S[:3,:3] = S_R
        S[:3,-1] = S_t
        return S
    
    def get_sym_extr(self, original_ext):
        original_ext = np.linalg.inv(original_ext)
        P_sym = original_ext@self.sym_mat
        P_sym[0,:] *= -1
        P_sym = np.linalg.inv(P_sym)
        return P_sym


def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict,
                    x1: float,
                    y1: float,
                    x2: float,
                    y2: float,
                    sample_data_token: str,
                    filename: str) -> OrderedDict:
    """
    Generate one 2D annotation record given various informations on top of the 2D bounding box coordinates.
    :param ann_rec: Original 3d annotation record.
    :param x1: Minimum value of the x coordinate.
    :param y1: Minimum value of the y coordinate.
    :param x2: Maximum value of the x coordinate.
    :param y2: Maximum value of the y coordinate.
    :param sample_data_token: Sample data token.
    :param filename:The corresponding image file where the annotation is present.
    :return: A sample 2D annotation record.
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    return repro_rec


def reverse_indexing_scene_names(nusc):
    # name is the filename of the first front cam frame of the scene
    name2token = {}
    for i in range(len(nusc.scene)):
        sample_data = nusc.get("sample_data", nusc.get("sample", nusc.scene[i]["first_sample_token"])["data"]["CAM_FRONT"])
        sample_token = sample_data["sample_token"]
        file_name = sample_data["filename"].split("/")[-1]
        scene_token = nusc.get("sample", sample_token)["scene_token"]
        assert file_name not in name2token
        name2token[file_name] = scene_token
    return name2token


def quaternion_rotation_matrix(Q):
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix
