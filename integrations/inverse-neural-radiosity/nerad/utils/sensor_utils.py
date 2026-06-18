import random
import re
from pathlib import Path

import mitsuba as mi
import drjit as dr
import numpy as np
from nerad.sensors import registered_sensors

def create_transforms(scene: str, n_views: int):
    # Hardcoded transformations only valid for lego scene
    if 'nerf_scenes' in scene:
        fov = 40
        if 'dragon' in scene:
            fov = 70
        transforms = {}
        scene_object = mi.load_file(scene)
        center = 0.5 * (scene_object.bbox().min + scene_object.bbox().max)

        steps = int(dr.sqrt(n_views))
        for k in range(steps):
            for j in range(steps):
                i = k*steps + j
                radius = 4.2
                space = mi.scalar_rgb.warp.square_to_uniform_hemisphere
                h, w = j/steps, k/steps
                if 'dragon' in scene:
                    radius = 20
                    space = mi.scalar_rgb.warp.square_to_uniform_sphere
                    h, w = max(j/steps, 0.05), max(k/steps, 0.05)
                vec = space(mi.ScalarVector2f(h, w))
                temp = vec[1]
                vec[1] = vec[2]
                vec[2] = temp
                origin = vec*radius

                to_world = mi.ScalarTransform4f \
                    .look_at(target=center,
                            origin=origin,
                            up=[0, 1, 0])
                transforms[str(i)] = {
                    "to_world": to_world.matrix.numpy().tolist(),
                    "fov": fov,
                }
    elif 'custom_kitch' or 'colloc-'in scene:
        if False:
            scene_object = mi.load_file(str(scene))
            scene_name = str(Path(scene).parents[0].name)
            data = sensor_data[scene_name]
            transforms = {}
            i = 0
            while len(transforms) < n_views:
                origin = gen_camera_origin(data['true_center'], data['radius_min_max'])
                target = gen_camera_lookat(data['lookat_bbox'])
                if not ((origin >= data['origin_bbox'][0]).all() and origin <= data['origin_bbox'][1]).all():
                    continue

                print("origin", origin, "target", target)

                trans = mi.ScalarTransform4f.look_at(origin=origin,
                                                        target=target,
                                                        up=[0,1,0])
                transforms[str(i)] = {
                    'to_world': mi.ScalarTransform4f(trans).matrix.numpy().tolist(),
                    'fov': data['fov'],
                }
                i+=1
                pass
        else:
            path = str(Path(scene).parent / "cameras.npz")
            cameras = np.load(path, allow_pickle=True)["cameras"]
            scene_name = str(Path(scene).parents[0].name)
            transforms = {}
            data = sensor_data[scene_name]

            for i in range(max(n_views, len(cameras))):

                trans = mi.ScalarTransform4f().look_at(origin=mi.ScalarPoint3f(cameras[i]['origin']),
                                                target=mi.ScalarPoint3f(cameras[i]['target']),
                                                up=mi.ScalarPoint3f(cameras[i]['up']))

                transforms[str(i)] = {
                    'to_world': mi.ScalarTransform4f(trans).matrix.numpy().tolist(),
                    'fov': data['fov'],
                }

    elif 'cornell-box-nobox' in scene or 'cornell-box' in scene or 'living-room-2' in scene or 'staircase' in scene or 'kitchen' in scene or 'veach_ajar' in scene or 'cube' in scene or 'bunny' in scene:
        path = str(Path(scene).parent / "cameras.xml")
        sensors = sensors = mi.load_file(path).sensors()
        transforms = {}
        for i in range(len(sensors)):
            fov = int(re.findall(r"\d+", re.findall(r"x_fov = \[\d+\]", str(sensors[i]))[0])[0])
            transforms[str(i)] = {
                "to_world": mi.ScalarTransform4f(sensors[i].world_transform().matrix.numpy()).matrix.numpy().tolist(),
                "fov": fov,
            }

    elif 'myLivingRoom' in scene:
        path = str(Path(scene).parent / "cameras.xml")
        scene = mi.load_file(path)
        transforms = {}
        camerapose = scene.shapes()[-3].bbox().min
        right = scene.shapes()[-1].bbox().min
        lookat = scene.shapes()[-2].bbox().min
        lookat_start = lookat
        lookat_end = lookat
        end = camerapose
        start = right

        for i in range(n_views):
            current = (i)/n_views
            cam_pos = (end-start) *current + start
            lookat = (lookat_end-lookat_start)*current + lookat_start
            trans = mi.ScalarTransform4f.look_at(origin=cam_pos,
                                                    target=lookat,
                                                    up=[0,1,0])

            transforms[str(i)] = {
                'to_world': mi.ScalarTransform4f(trans).matrix.numpy().tolist(),
                'fov': 55,
            }


    else:
        raise Exception('no camera generation strategy for the scene is specified!')
    return transforms

def gen_cameras_on_unit_sphere(center, radius):
    nrm = np.random.normal(0,1, 3) #https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
    lam_sq = (nrm * nrm).sum()
    lam = np.sqrt(lam_sq)
    cam_pos = nrm / lam
    return cam_pos * radius + center


def gen_camera_origin(true_center, radius_min_max):
    radius_min, radius_max = radius_min_max
    sampled_radius = np.random.uniform(radius_min, radius_max)

    nrm = np.random.normal(0,1, 3) #https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
    lam_sq = (nrm * nrm).sum()
    lam = np.sqrt(lam_sq)
    cam_pos = nrm / lam

    return cam_pos * sampled_radius + true_center


def gen_camera_lookat(lookat_bbox): #3
    rand = np.random.rand(3)
    lookat_bbox_np = np.array(lookat_bbox).reshape(2, 3)
    range = lookat_bbox_np[1] - lookat_bbox_np[0]
    sampled_point = lookat_bbox_np[0] + range * rand
    return sampled_point



def create_sensor(resolution_x, resolution_y, transform, random_crop=False, crop_size=None, valid_offsets=None):
    return mi.load_dict(sensor_dict(resolution_x=resolution_x, resolution_y=resolution_y, fov=transform["fov"], to_world=transform["to_world"], random_crop=random_crop, crop_size=crop_size, valid_offsets=valid_offsets))

def sensor_dict2(resolution_x, resolution_y, fov, to_world, random_crop, crop_size, valid_offsets):
    sensor = {
        "type": "random_ray",
        "to_world": mi.ScalarTransform4f(to_world),
        "film": {
                "type": "hdrfilm",
                "width": resolution_x,
                "height": resolution_y,
                "filter": {"type": "box"},
                "pixel_format": "rgba"
        },
        "wrapped_sensor": sensor_dict(resolution_x, resolution_y, fov, to_world, random_crop, crop_size, valid_offsets)
        # TODO: All scene MUST be rgba in this scenario and use a box filter, even for ground truth
    }

    if random_crop:
        assert crop_size > 0
        assert (resolution_x-crop_size) >= 0
        assert (resolution_y-crop_size) >= 0

        if valid_offsets is not None and len(valid_offsets)>0:
            crop_offset = random.choice(valid_offsets)
            crop_offset = [crop_offset[1].item(), crop_offset[0].item()]
        else:
            crop_offset = [
                random.randint(0, resolution_x-crop_size),
                random.randint(0, resolution_y-crop_size)
            ]
        # print("WARNING: forcing crop_offset to 0")
        # crop_offset[0] = 0
        # crop_offset[1] = 0
        sensor["film"]["crop_width"] = crop_size
        sensor["film"]["crop_height"] = crop_size
        sensor["film"]["crop_offset_x"] = crop_offset[0]
        sensor["film"]["crop_offset_y"] = crop_offset[1]

    return sensor


def sensor_dict(resolution_x, resolution_y, fov, to_world, random_crop, crop_size, valid_offsets):
    sensor = {
        "type": "perspective",
        "fov": fov,
        "to_world": mi.ScalarTransform4f(to_world),
        "film": {
                "type": "hdrfilm",
                "width": resolution_x,
                "height": resolution_y,
                "filter": {"type": "box"},
                "pixel_format": "rgba",
                "component_format": "float32"
        },
        # "sample": {
        #     'type': 'orthogonal',
        #     'sample_count': 4489
        # }
        # TODO: All scene MUST be rgba in this scenario and use a box filter, even for ground truth
    }

    if random_crop:
        assert crop_size > 0
        assert (resolution_x-crop_size) >= 0
        assert (resolution_y-crop_size) >= 0

        if valid_offsets is not None and len(valid_offsets)>0:
            crop_offset = random.choice(valid_offsets)
            crop_offset = [crop_offset[1].item(), crop_offset[0].item()]
        else:
            crop_offset = [
                random.randint(0, resolution_x-crop_size),
                random.randint(0, resolution_y-crop_size)
            ]
        # print("WARNING: forcing crop_offset to 0")
        # crop_offset[0] = 0
        # crop_offset[1] = 0
        # print("crop offset", crop_offset)
        sensor["film"]["crop_width"] = crop_size
        sensor["film"]["crop_height"] = crop_size
        sensor["film"]["crop_offset_x"] = crop_offset[0]
        sensor["film"]["crop_offset_y"] = crop_offset[1]

    return sensor


#Hardcoded
sensor_data = {'custom_kitch': {'true_center': [-0.57, 1.7, 1.2], 'radius_min_max':[1,3], 'lookat_bbox':[-2.138698, 0.983547, -1.48239, -1.6504, 0.983547, -2.46994], 'origin_bbox':[-1.6, 0.3, -2.4,  -0.768551, 2.37743, -0.728473], 'fov':67},
               'custom_kitchen_ver9_principled_path': {'fov': 67},
               'colloc-living-room-1': {'fov': 67},
               'colloc-closed_living-room-2': {'fov': 67},
               'colloc-living-room-2': {'fov': 67},
               'colloc-simplified-living-room-2': {'fov': 67},
               'colloc-staircase-1_modified': {'fov': 67},
               'colloc-bedroom-1': {'fov': 67},
               'colloc-living-room-3': {'fov': 67},
               'kitchen': {'fov': 67}
               }
