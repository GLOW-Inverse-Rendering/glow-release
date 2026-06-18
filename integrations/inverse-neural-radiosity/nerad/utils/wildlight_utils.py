import pathlib
import numpy as np
import mitsuba as mi

class WildLightCamerasUtil:
    def __init__(self, sdf_cameras: pathlib.Path):
        self.sdf_cameras_path = sdf_cameras
        self.cameras = np.load(self.sdf_cameras_path)
        self.scale_mat_0 = self.cameras["scale_mat_0"]
        self.aabb = None
        self.update_bbox()
    def get_world_pos(self, x):
        return (x * float(self.scale_mat_0[0,0]))  + self.scale_mat_0[:3, 3][None]
        
    def get_sdf_pos(self, x):
        # return x * self.scale_mat_0[0,0] + self.scale_mat_0[:3, 3][None]
        return (x - self.scale_mat_0[:3, 3][None]) / self.scale_mat_0[0,0]

    def update_bbox(self):
        self.aabb = mi.ScalarBoundingBox3f()
        self.aabb.expand(self.get_world_pos(mi.ScalarPoint3f(-1.0, -1.0, -1.0)))
        self.aabb.expand(self.get_world_pos(mi.ScalarPoint3f(-1.0, -1.0, 1.0)))
        self.aabb.expand(self.get_world_pos(mi.ScalarPoint3f(-1.0, 1.0, -1.0)))
        self.aabb.expand(self.get_world_pos(mi.ScalarPoint3f(-1.0, 1.0, 1.0)))
        self.aabb.expand(self.get_world_pos(mi.ScalarPoint3f(1.0, -1.0, -1.0)))
        self.aabb.expand(self.get_world_pos(mi.ScalarPoint3f(1.0, -1.0, 1.0)))
        self.aabb.expand(self.get_world_pos(mi.ScalarPoint3f(1.0, 1.0, -1.0)))
        self.aabb.expand(self.get_world_pos(mi.ScalarPoint3f(1.0, 1.0, 1.0)))

    def get_min_max(self):
        # https://github.com/mitsuba-renderer/mitsuba3/blob/5016304dc529a800ca7983fc5d9d8332f5811087/include/mitsuba/core/bbox.h#L124C16-L124C16
        scene_min, scene_max = self.aabb.corner(0), self.aabb.corner(7)
        print("WARNING: using configured scene_min/scene_max override")
        # print(self.scale_mat_0)
        # scene_min = self.scale_mat_0[:3, 3]
        # scene_max =  self.scale_mat_0[:3, 3] + self.scale_mat_0[0,0]
        return np.array([-2.47659676, -0.01449606, -2.47996356]), np.array([-1.03374959,  1.96495601, -0.82325995])
        return scene_min, scene_max
