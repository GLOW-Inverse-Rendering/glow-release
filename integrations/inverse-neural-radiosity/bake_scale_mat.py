import argparse
import pathlib
import numpy as np
import cv2
from bs4 import BeautifulSoup
import json
import copy
def compose_transf(P1, P2):

    R1 = P1[:3,:3]
    T1 = P1[:3,3:4]
    R2 = P2[:3,:3]
    T2 = P2[:3,3:4]

    R = R1@R2
    T = T1 + R1@T2
    return np.concatenate([R,T], axis=-1)
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def bake_scale_mat(transforms, scale_mat):
    edited_transforms = {}
    for key, transform in transforms.items():
        to_world = np.array(transform["to_world"])
        # print("=================== WARNING: not accounting for scaling ==========================")
        # print("scale_mat", scale_mat)
        print("to_world", to_world)
        # to_world = compose_transf(np.linalg.inv(scale_mat), to_world)
        world_mat = compose_transf(np.linalg.inv(to_world), scale_mat)
        intrinsics, pose = load_K_Rt_from_P(None, world_mat)
        print("intrinsics", intrinsics)
        print("pose", pose)
        print("diff", intrinsics - np.eye(intrinsics.shape[0]))
        assert np.allclose(intrinsics, np.eye(intrinsics.shape[0]), atol=1e-6), intrinsics
        
        # fx = intrinsics[0,0]
        # fy = intrinsics[1,1]
        # hfov = 360 * np.arctan(512 / (2 * fx)) / np.pi
        # print("hfov", hfov)
        # print("pose", pose)
        # to_world = np.linalg.inv(scale_mat) @ to_world 

        # to_world[:3, 3] -= scale_mat[:3, 3]
        # scale = np.array()
        # to_world[:, ]
        to_world = pose
        print("to_world after", to_world)

        edited_transforms[key] = copy.deepcopy(transform)
        edited_transforms[key]["to_world"] = to_world.tolist()
        pass
    return edited_transforms


def main(transforms_path, cameras_sphere_path, output_path):
    with open(transforms_path) as f:
        transforms = json.load(f)

    cameras_sphere = np.load(cameras_sphere_path)
    scale_mat_0 = cameras_sphere["scale_mat_0"]
    
    dic = bake_scale_mat(transforms, scale_mat_0)
    with open(output_path, "wt") as f:
        json.dump(dic, f)
        pass
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("transforms_path", type=pathlib.Path)
    parser.add_argument("cameras_sphere_path", type=pathlib.Path)
    parser.add_argument("output_path", type=pathlib.Path)
    args = parser.parse_args()
    main(**vars(args))
