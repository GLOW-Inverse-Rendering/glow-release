import math
import itertools
from pathlib import Path
from typing import Union

import drjit as dr
import mitsuba as mi
import torch

from nerad.model.config import RenderingConfig
from nerad.utils.image_utils import save_image

from nerad.utils.mitsuba_utils import block_sum_image

def mis_weight(pdf_a, pdf_b):
    pdf_a *= pdf_a
    pdf_b *= pdf_b
    w = pdf_a / (pdf_a + pdf_b)
    return dr.detach(dr.select(dr.isfinite(w), w, 0))


def process_nerad_output(img):
    residual = img[:, :, -3:]
    LHS = img[:, :, -7:-3]
    RHS = img[:, :, :4]
    return residual, LHS, RHS

def process_nerf_output(img):
    rgb_fine = img[:, :, :-3]
    rgb_coarse = img[:, :, -3:]
    return rgb_fine, rgb_coarse


def adaptive_render(scene, integrator, sensor, spp_per_batch=64, init_batches=32, z_val=1.96, max_tol=0.05):
    # images = []
    # aov_img = mi.render(scene, spp=1, integrator=aov_integrator, sensor=sensor)
    # for i in range(init_batches):

        # images.append(img.numpy())
    # total_samples = spp_per_batch * init_batches
    # first_momemt_sum = np.sum(images, axis=0)
    # second_moment_sum = np.sum(images**2, axis=0)
    total_batches = mi.Float(0)
    first_moment_sum = mi.Float(0)
    second_moment_sum = mi.Float(0)
    max_CI_val = mi.Float(0)
    # i = mi.Float(0)
    dr.set_log_level(dr.LogLevel.Info)


    # img_sum = mi.Float(0)
    loop = mi.Loop("adpt_sampling", lambda: (total_batches,first_moment_sum, second_moment_sum,  max_CI_val))
    while loop((total_batches<32) |  (max_CI_val > max_tol)):
        print('here')
        img = integrator.render(scene=scene, spp=spp_per_batch, sensor=sensor)
        print("Reached here")
        img = mi.TensorXf(img)
        first_moment_sum += img
        second_moment_sum += dr.power(img, 2)
        total_batches += 1
        mean = first_moment_sum / total_batches
        sigma = (second_moment_sum - dr.power(mean, 2) * total_batches) / (total_batches - 1)
        max_CI_val = z_val * sigma / dr.sqrt(total_batches)
        max_CI_val /= mean
    image = first_moment_sum / total_batches

    return mi.Bitmap(image, pixel_format=img.pixel_format, channel_names=img.channel_names)

def render_and_save_image(
    folder: Path,
    name: str,
    scene: mi.Scene,
    integrator: mi.Integrator,
    rendering: RenderingConfig,
    sensor: Union[int, mi.Sensor] = 0,
    formats: list[str] = None,
    use_optix: bool = False,
    use_adapt_spp: bool = False,
    albedo_integrator = None,
    last_frame_img = None
) -> list[mi.Bitmap]:
    if formats is None:
        formats = ["png", "exr"]

    with torch.no_grad():
        with dr.suspend_grad():
            img = mi.render(scene, spp=rendering.spp, integrator=integrator, sensor=sensor)
            print("rendering", integrator, sensor)
            print(img)
            if use_optix:
                albedo = mi.render(scene, spp=32, integrator=albedo_integrator, sensor=sensor)
                aov_integrator = mi.load_dict({
                    'type': 'aov',
                    'aovs': 'albedo:albedo,normals:sh_normal',
                    'integrator': {
                        'type': 'direct',
                    }
                })
                aov_img = mi.render(scene, spp=1, integrator=aov_integrator, sensor=sensor)
                normal = aov_img[:, :, -3:]
                flow = dr.zeros(mi.TensorXf, (img.shape[0], img.shape[1], 2))
                print("last_frame_img here", last_frame_img)
                if last_frame_img is None:
                    last_frame_img = img
                #denoiser = mi.OptixDenoiser(input_size=img.shape[:2], albedo=True, normals=True, temporal=True)
                #img = denoiser(img, True, albedo=albedo[:, :, :3], normals=normal, to_sensor=sensor.world_transform().inverse(), flow=flow, previous_denoised=last_frame_img)
                denoiser = mi.OptixDenoiser(input_size=(img.shape[1], img.shape[0]), albedo=True, normals=True, temporal=True)
                # print(img.shape, albedo.shape)
                img = denoiser(img, False, albedo=albedo[:, :, :3], normals=normal, to_sensor=sensor.world_transform().inverse(), flow=flow, previous_denoised=last_frame_img) # ,
            # import imageio
            # import numpy as np
            # imageio.imwrite("debug.exr", (img[:, :, :4].numpy()))
            # input("saved")
            if rendering.integrator.startswith("nerad"):
                _, LHS, RHS = process_nerad_output(img)
                save_image(folder / "rhs", name, formats, RHS)
                save_image(folder / "lhs", name, formats, LHS)
                return [LHS, RHS]
            elif rendering.integrator == "nerf":
                rgb_fine, rgb_coarse= process_nerf_output(img)
                save_image(folder / "rgb_fine", name, formats, rgb_fine)
                save_image(folder / "rgb_coarse", name, formats, rgb_coarse)
                return [rgb_fine, rgb_coarse]
            else:
                save_image(folder, name, formats, img[:, :, :4])
                return [img]

def multi_scale_img(img, step, multi_scale, multi_scale_iters):
    assert len(multi_scale)-1 == len(multi_scale_iters)
    for scale, iters in zip(multi_scale, multi_scale_iters+[float('inf')]):
        print(scale, iters)
        if iters >= step:
            break

    img = block_sum_image(img, scale[0], scale[1])
    return  img
