import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import traceback
import matplotlib
matplotlib.use('Agg')
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)

import os
import time
import logging
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import SDFNetwork, SingleVarianceNetwork, SDFNetworkNGP
from models.physicalshader import PhysicalRenderingNetwork, PhysicalNeRF, PhysicalNeRFNGP, PhysicalNeRFSep, NeRF, RadianceCache, RadianceCacheMLP, RadianceCacheMLPOcclusion
from models.renderer import NeuSRenderer
from models.ssim import loss_dssim
from models.ncc import loss_ncc
import xatlas
import scipy
import scipy.interpolate
from tabulate import tabulate
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio
import imageio
from models.uv_mapping import generate_uv_map
from models import distiller
import utils

import time
import loss
import math
from omegaconf import DictConfig, OmegaConf
import hydra
import models.giphysicalshader
import collections.abc
from wildlightutils import render_utils
from wildlightutils import reg_utils
from wildlightutils import image_utils
import drjit as dr
PLOT_S_GRAD=False
def parametrize(vertices, faces):
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.padding = 2
    pack_options.create_image = True
    atlas.generate(chart_options, pack_options, True)
    return atlas[0], max(atlas.chart_image.shape)


def batch_feed(func, data, batch_size=4096):
    ret_data = dict()

    for chunk in tqdm(np.array_split(data, data.shape[0]//batch_size, axis=0)):
        rst = func(chunk)
        for k in rst:
            ret_data[k] = ret_data.get(k, list())
            ret_data[k].append(rst[k])
    
    return {k: np.concatenate(ret_data[k]) for k in ret_data}

def batch_feed2(func, data, batch_size=4096):
    ret_data = []
    for chunk in tqdm(np.array_split(data, data.shape[0]//batch_size, axis=0)):
        rst = func(chunk)
        ret_data.append(rst)
    return torch.concatenate(ret_data)

class Runner:
    def __init__(self, conf, mode='train', case='CASE_NAME', is_continue=False, exp_suffix=None, distill_suffix=None, distill_conf=None, disable_scaler=False, set_s=None, mitsuba_renderer=None, latest_model_name=None):
        self.device = torch.device('cuda')
        self.case = case

        self.mitsuba_renderer = mitsuba_renderer
        self.conf = conf
        if self.conf.dataset.override_case is not None:
            self.conf.dataset.data_dir = self.conf.dataset.data_dir.replace('CASE_NAME', self.conf.dataset.override_case)
        else:   
            self.conf.dataset.data_dir = self.conf.dataset.data_dir.replace('CASE_NAME', case)
        print("self.conf.dataset.data_dir", self.conf.dataset.data_dir)
        self.base_exp_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        if False:
            self.dataset = Dataset(self.conf['dataset_val'])
        elif mode.startswith('validate_mesh') or mode.startswith("validate_grid") :
            self.dataset = Dataset(self.conf.dataset, load_images=False)
        else:
            self.dataset = Dataset(self.conf.dataset)
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.train.end_iter
        self.save_freq = self.conf.train.save_freq
        self.save_freq_latest = self.conf.train.save_freq_latest
        self.report_freq = self.conf.train.report_freq
        self.val_freq = self.conf.train.val_freq
        self.val_mesh_freq = self.conf.train.val_mesh_freq
        self.rgb_batch_size = self.conf.train.batch_size
        self.alpha_batch_size = self.conf.train.batch_size
        assert self.rgb_batch_size*self.alpha_batch_size > 0
        self.batch_size = self.rgb_batch_size
        self.validate_resolution_level = self.conf.train.validate_resolution_level
        self.learning_rate = self.conf.train.learning_rate
        self.learning_rate_flash = self.conf.train.learning_rate_flash
        self.learning_rate_alpha = self.conf.train.learning_rate_alpha
        self.use_white_bkgd = self.conf.train.use_white_bkgd
        self.warm_up_end = self.conf.train.warm_up_end
        self.anneal_end = self.conf.train.anneal_end

        self.samples_per_pixel = self.conf.train.samples_per_pixel
        # Weights
        self.rgb_loss_type = self.conf.train.rgb_loss_type
        self.color_weight = self.conf.train.color_weight
        self.igr_weight = self.conf.train.igr_weight
        self.mask_weight = self.conf.train.mask_weight
        self.dssim_weight = self.conf.train.dssim_weight
        self.dssim_window_size = self.conf.train.dssim_window_size
        self.ncc_weight = self.conf.train.ncc_weight
        self.pts_radiance_weight = self.conf.train.pts_radiance_weight
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None
        self.disable_scaler = disable_scaler
        # self.debug_error = debug_error
        if self.rgb_loss_type == 'adpt_l2_log_lin':
            self.loss_func = loss.AdaptiveL2LogLinLoss()
        self.orient_loss_period = self.conf.train.orient_loss_period
        # Networks
        params_to_train = []
        self.nerf_outside = None

        if self.conf.model.sdf_type == "default":
            self.sdf_network = SDFNetwork(**self.conf.model.sdf_network).to(self.device)
        elif self.conf.model.sdf_type == "ngp":
            print("Using NGP!")
            self.sdf_network = SDFNetworkNGP(**self.conf.model.sdf_network).to(self.device)
        else:
            raise NotImplementedError()
        self.deviation_network = SingleVarianceNetwork(**self.conf.model.variance_network).to(self.device)

        if self.conf.model.view_net.view_net_type == "radiance_cache":
            self.color_network = RadianceCache(**self.conf.model.view_net.radiance_cache_network).to(self.device)
        elif self.conf.model.view_net.view_net_type == "radiance_cache_mlp":
            self.color_network = RadianceCacheMLP(**self.conf.model.view_net.radiance_cache_network).to(self.device)
        elif self.conf.model.view_net.view_net_type == "radiance_cache_mlp_occlusion":
            self.color_network = RadianceCacheMLPOcclusion(**self.conf.model.view_net.radiance_cache_network).to(self.device)
        else:
            self.color_network = PhysicalRenderingNetwork(self.conf.model.view_net.physical_rendering_network, self.conf.model.view_net.brdf_settings, view_net_type=self.conf.model.view_net.view_net_type).to(self.device)

        if self.nerf_outside is not None:
            params_to_train += list(self.nerf_outside.parameters())
        if self.conf.train.optimize_geo:
            params_to_train += list(self.sdf_network.parameters())
            if not self.conf.train.freeze_s:
                params_to_train += list(self.deviation_network.parameters())

        
        if hasattr(self.color_network, "light_net"):
            print("training light net")
            params_to_train += list(self.color_network.light_net.parameters())
        if hasattr(self.color_network, "ambient_net"):
            params_to_train += list(self.color_network.ambient_net.parameters())
        else:
            # params_to_train += list(self.color_network.parameters())
            raise RuntimeError("not supported. color_network neither have light_net nor have ambient_net")
        params = [{'params': params_to_train, "lr": self.learning_rate} ]
        if hasattr(self.color_network, "gamma"):
            params += [{'params': [self.color_network.gamma], "lr": self.learning_rate_flash}]
        
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate, amsgrad=self.conf.train.amsgrad)
        

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                    #  brdf_settings=self.conf.model.brdf_settings,
                                     **self.conf.model.neus_renderer,
                                     need_hess=self.conf.train.hess_error_weight>0)

            
        # Load checkpoint
        # latest_model_name = None
        if latest_model_name is None:
            models_path = os.path.join(self.base_exp_dir, 'checkpoints')
            if is_continue and os.path.exists(models_path):
                if os.path.exists(os.path.join(models_path, "latest.pth")):
                    latest_model_name = "latest.pth"
                else:
                    model_list_raw = os.listdir(models_path)
                    model_list = []
                    for model_name in model_list_raw:
                        if model_name[-3:] == 'pth':
                            model_list.append(model_name)
                    model_list.sort(key=lambda a: int(a.replace("ckpt_", "").replace(".pth", "")))
                    if len(model_list) > 0:
                        latest_model_name = model_list[-1]
        if latest_model_name is not None:
            logging.info('NeuS: Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)
        self.mitsuba_trainer = None
        if self.mitsuba_renderer is not None: 
            self.extract_mesh_and_reinit_mitsuba(mode)
        # if mitsuba_renderer is not None: 
        #     # self.mitsuba_renderer = models.giphysicalshader.PhysicalShaderGI(self.sdf_network, mitsuba_renderer)
        #     # self.mitsuba_saving_hooks = saving_hooks
        #     self.extract_mesh_and_reinit_mitsuba()

            # self.mitsuba_renderer = None
            
        if distill_suffix is not None:
            teacher_conf = ConfigFactory.parse_string(open(distill_conf).read())

            if teacher_conf.model.sdf_type == "default":
                self.teacher_sdf_network = SDFNetwork(**teacher_conf.model.sdf_network).to(self.device)
            elif teacher_conf.model.sdf_type == "ngp":
                print("teacher Using NGP!")
                self.teacher_sdf_network = SDFNetworkNGP(**teacher_conf.model.sdf_network).to(self.device)
            else:
                raise NotImplementedError()

            self.distill_exp_dir = distill_suffix
            model_list_raw = os.listdir(os.path.join(self.distill_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth':
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]
            assert latest_model_name is not None
            checkpoint = torch.load(os.path.join(self.distill_exp_dir, 'checkpoints', latest_model_name), map_location=self.device)
            self.teacher_sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
            self.teacher_sdf_network.requires_grad_(False)
            self.distiller = distiller.Distiller(self.teacher_sdf_network, self.sdf_network, self.renderer)
        self.is_distill = distill_suffix is not None
        
        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()
        if set_s is not None:
            with torch.no_grad():
                self.deviation_network.variance.copy_(torch.log(torch.as_tensor(1.0/set_s))/10)
                print("s is",  1/self.deviation_network(torch.zeros([1,3])))
        self.debug_all_s_grad = []
        self.debug_all_get_s_grad = []
    
    @torch.no_grad()
    def validate_images(self, resolution_level=1, basic_only=False, mitsuba_repeats=1):
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        psnr, ssim = [], []


        os.makedirs(os.path.join(self.base_exp_dir, 'novel_view'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'gt'), exist_ok=True)

        pbar = tqdm(range(self.dataset.n_images))

        for i in pbar:
            img = self.dataset.image_at(i, resolution_level, False)
            mask = self.dataset.mask_at(i, resolution_level, False)

            img_rendered = self.validate_image(i, resolution_level, printf=pbar.set_description, basic_only=basic_only, mitsuba_repeats=mitsuba_repeats)

            # img = img.clip(0, self.dataset.saturation_intensity)
            # img_rendered = img_rendered.clip(0, self.dataset.saturation_intensity)

            # img[mask < 0.5] = 0
            # img_rendered[mask < 0.5] = 0

            # psnr.append(peak_signal_noise_ratio(img, img_rendered, data_range=img.max()))
            # # ssim.append(structural_similarity(img, img_rendered, data_range=img.max(),multichannel=True,channels_axis=-1))

            # cv.imwrite(os.path.join(self.base_exp_dir, 'novel_view', f"{i:02}.exr"), np.concatenate([img_rendered[...,::-1], mask.reshape(img_rendered.shape[:-1] + (-1,))[...,:1]], axis=-1))
            # cv.imwrite(os.path.join(self.base_exp_dir, 'gt', f"{i:02}.exr"), img[..., ::-1])

        psnr = np.array(psnr)
        ssim = np.array(ssim)

        has_flash_light = np.stack(self.dataset.light_energies).max(-1) > 0

        psnr_ambient, psnr_flash, psnr_all = psnr[has_flash_light==False].mean(), psnr[has_flash_light].mean(), psnr.mean()
        # ssim_ambient, ssim_flash, ssim_all = ssim[has_flash_light==False].mean(), ssim[has_flash_light].mean(), ssim.mean()

        # print( tabulate([['img', 'ambient', 'flash', 'all'], 
        #                  ['psnr', psnr_ambient, psnr_flash, psnr_all], 
        #                  ['ssim', ssim_ambient, ssim_flash, ssim_all]], headers='firstrow', tablefmt='github'))

        return psnr, ssim    
    def extract_mesh_and_reinit_mitsuba(self, mode):
        print("extracting mesh and reinitializing mitsuba...")
        # if self.mitsuba_trainer is not None:
        #     print("trying to empty backward graph")
        #     import mitsuba as mi
        #     dr.traverse(mi.Float, dr.ADMode.Forward, dr.ADFlag.Default) 
        #     dr.traverse(mi.Float, dr.ADMode.Backward, dr.ADFlag.Default)
        #     print("sync thread")
        #     dr.sync_thread()
            
        #     drjit_path = os.path.join(os.environ["HOME"], ".drjit")
        #     print("clear drjit cache", drjit_path)

        #     import shutil
        #     shutil.rmtree(drjit_path, ignore_errors=True)
        #     os.makedirs(drjit_path)
        #     del self.mitsuba_trainer
        #     self.mitsuba_trainer = None
        #     dr.flush_malloc_cache()
        #     torch.cuda.empty_cache()
        mesh_path = self.mitsuba_renderer.mesh_path
        was_dummy = mesh_path == "dummy"
        try:
            
            # print(self.iter_step >= 49999, mode != "validate_mesh", mesh_path=="dummy", self.iter_step < self.end_iter, (not self.mitsuba_renderer.config.render.no_bsdf_sample))
            if self.iter_step >= 49999 and mode != "validate_mesh"  and mesh_path=="dummy" and self.iter_step < self.end_iter and (not self.mitsuba_renderer.config.render.no_bsdf_sample):
                mesh_path = self.validate_mesh_hires(world_space=False, resolution=1024, threshold=0, simplify=False, bake_texture_maps=False, bake_vert_maps=False, texture_resolution=4096)
        except:
            import traceback as tb
            tb.print_exc()
            print("val mesh error!!!!")
        # print(mesh_path, type)
        if (was_dummy and mesh_path != "dummy" and (not self.mode.startswith("validate_mitsuba"))) and self.conf.train.prune_outside_cam:
            self.check_mesh_face_visibility(mesh_path, mesh_path)
        scene, integrator, learned_info, ckpt_path = models.giphysicalshader.init_scene(self.mitsuba_renderer.scene_path, mesh_path, self.mitsuba_renderer.out_dir, self.mitsuba_renderer.config)        
        self.mitsuba_trainer = models.giphysicalshader.PhysicalShadingTrainer(self.sdf_network, self.color_network, self.renderer, self.deviation_network, scene, integrator, learned_info, ckpt_path, self.mitsuba_renderer.config, self.dataset).to(self.device)
        pass

    def check_mesh_face_visibility(self, mesh_file, out_mesh):
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer
        import trimesh
        ply_mesh = trimesh.load(mesh_file)
        V = torch.as_tensor(ply_mesh.vertices, dtype=torch.float32, device=self.device)
        F = torch.as_tensor(ply_mesh.faces, dtype=torch.int64, device=self.device)
        # mesh = Meshes(verts=[V], faces=[F])

        n_images = self.dataset.n_images
        assert self.dataset.images is not None
        H, W = self.dataset.images.shape[1], self.dataset.images.shape[2]
        H, W = H , W 
        face_mask = torch.zeros(ply_mesh.faces.shape[0], dtype=torch.bool, device=self.device)
        for img_idx in range(n_images):
            print("img_idx", img_idx)
            K = self.dataset.intrinsics_all[img_idx]
            c2w = self.dataset.pose_all[img_idx]
            R = torch.as_tensor(c2w[:3, :3], dtype=torch.float32, device=self.device)
            T = torch.as_tensor(c2w[:3, 3], dtype=torch.float32, device=self.device)
            verts_cam = (V - T[None, :]) @ R   # world → camera
            face_mask_i = self.faces_in_frustum_and_facing_camera(verts_cam, F, K, W, H)

            import pathlib
            import imageio
            out_path = pathlib.Path("debug_out")
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{img_idx}.ply"
            # imageio.imwrite(out_file, ((face_mask_i*255).cpu().numpy()).astype(np.uint8))
            vis_i_mesh = ply_mesh.copy()
            vis_i_mesh.update_faces(face_mask_i.cpu().numpy())
            vis_i_mesh.export(out_file)
            face_mask[face_mask_i] = True

        ply_mesh.update_faces(face_mask.cpu().numpy())
        ply_mesh.export(out_mesh)

    def faces_in_frustum_and_facing_camera(
        self,
        verts_cam: torch.Tensor,  # (N_v, 3), in camera coordinates
        faces: torch.Tensor,      # (N_f, 3), long
        K: torch.Tensor,          # (3, 3)
        image_width: int,
        image_height: int,
        z_near: float = 1e-3,
        z_far: float = 1e6,
    ):
        """
        Returns:
            face_mask: (N_f,) bool tensor, True if:
                - all 3 verts are inside frustum, and
                - face normal is facing the camera
            visible_face_indices: 1D tensor of indices where mask is True
        """
        device = verts_cam.device
        K = K.to(device)

        # ---- 1) Per-vertex frustum test ----
        x = verts_cam[:, 0]
        y = verts_cam[:, 1]
        z = verts_cam[:, 2]

        # Must be in front of camera and within near/far
        z_valid = (z > z_near) & (z < z_far)

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Project to pixel coordinates
        # u = fx * x / z + cx
        # v = fy * y / z + cy
        # (avoid division by zero with eps if needed)
        eps = 1e-8
        z_safe = z.clamp(min=eps)
        u = fx * x / z_safe + cx
        v = fy * y / z_safe + cy

        u_valid = (u >= 0) & (u < image_width)
        v_valid = (v >= 0) & (v < image_height)

        vert_in_frustum = z_valid & u_valid & v_valid   # (N_v,)

        # ---- 2) Require all 3 vertices of each face to be in frustum ----
        v0_idx = faces[:, 0]
        v1_idx = faces[:, 1]
        v2_idx = faces[:, 2]

        face_verts_in_frustum = (
            vert_in_frustum[v0_idx]
            & vert_in_frustum[v1_idx]
            & vert_in_frustum[v2_idx]
        )  # (N_f,)

        # ---- 3) Face normals & front-facing test ----
        # Gather vertex positions in camera space
        v0 = verts_cam[v0_idx]  # (N_f, 3)
        v1 = verts_cam[v1_idx]
        v2 = verts_cam[v2_idx]

        # Unnormalized face normal (right-handed triangle winding)
        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (N_f, 3)

        # Centroid in camera space
        centroids = (v0 + v1 + v2) / 3.0  # (N_f, 3)

        # Camera at origin, so vector from face to camera is -centroid
        to_camera = -centroids

        # Front-facing if normal points roughly towards the camera
        # i.e., angle between normal and to_camera is < 90 degrees:
        # dot(normal, to_camera) > 0
        facing = (normals * to_camera).sum(dim=-1) > 0  # (N_f,)

        # Optionally, ignore nearly-degenerate faces (very small normal length)
        # normal_len_sq = (normals ** 2).sum(dim=-1)
        # not_degenerate = normal_len_sq > 1e-12

        # ---- 4) Combine conditions ----
        face_mask = face_verts_in_frustum & facing   # (N_f,)
        # visible_face_indices = torch.nonzero(face_mask, as_tuple=False).squeeze(-1)

        return face_mask


    def train(self, sample_mode="batch"):
        print("s is",  1/self.deviation_network(torch.zeros([1,3])))
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        # if self.conf["model.sdf_type"] != "ngp":
        #     self.update_learning_rate()
        self.update_learning_rate()
        res_step = max(0, self.end_iter - self.iter_step)
        image_perm = self.get_image_perm()
        scaler = torch.cuda.amp.GradScaler()
        validation_set = np.linspace(0, self.dataset.n_images, num=30, dtype=int, endpoint=False)
        # print("self.mitsuba_renderer", self.mitsuba_renderer)
        # if self.mitsuba_trainer is None:
        # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], profile_memory=True,  record_shapes=True, with_stack=True, with_modules=True) as prof: 
        for iter_i in tqdm(range(res_step)):
            print("start iter")
            print("cuda allocated in torch:", torch.cuda.memory.memory_allocated())
            # for g in self.optimizer.param_groups:
            #     print("real lr in train", g["lr"])
            # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            import gc
            # gc.collect()
            # torch.cuda.empty_cache()
            if iter_i % 100 == 0:
                dr.flush_malloc_cache()
                torch.cuda.empty_cache()
            t_init = time.time()
            if self.conf.model.sdf_type == "ngp":
                perc_iters = self.iter_step / self.end_iter
                self.sdf_network.set_encoding_level(perc_iters=perc_iters)
                # self.color_network.ambient_net.set_encoding_level(perc_iters=perc_iters)
                pass

            img_idx = self.iter_step % len(image_perm)
            cap_pixel_val = self.dataset.cap_pixel_val(image_perm[img_idx])

            seed = int(time.time() * 1000) % 1000000 + iter_i

            data, pixels = self.dataset.gen_random_rays_at(image_perm[img_idx], self.rgb_batch_size, True, shift=(0.0,0.0), seed=seed)
            pixels_x, pixels_y = pixels[..., 0:1], pixels[..., 1:2]
            light_o, light_lumen = self.dataset.gen_light_params(image_perm[img_idx])
            #print("light_o", light_o, "light_lumen", light_lumen)
            rays_o, rays_d, true_rgb, mask = data[..., :3], data[..., 3: 6], data[..., 6: 9], data[..., 9: 10]#, data[..., 10:11], data[..., 11:12]
            pts, zs, normals, valid_mask = self.mitsuba_trainer.physical_shader_gi.renderer.cast_ray_torch(rays_o, rays_d)
            dirs = rays_d
            pts = pts.reshape(-1, 3)
            dirs = dirs.reshape(-1, 3)
            zs = zs.reshape(-1)
            normals = normals.reshape(-1, 3)
            render_out = {
                "z": zs,
                "gradients": normals,
                "valid_mask": valid_mask
            }
            t_render_end = time.time()
            t_loss_end = time.time()
            if self.mitsuba_trainer is not None:
                # print("color_fine_loss", color_fine_loss)
                light_to_world, mitsuba_light_lumen = self.dataset.gen_light_params_pose(image_perm[img_idx])
                # mitsuba_loss, extra_losses, extra_out = self.mitsuba_trainer()
                light_o = light_to_world[:3, 3]
                self.mitsuba_trainer.init_per_step()
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
                combined_mask = mask * (true_rgb < cap_pixel_val) 
                # print("here", true_rgb.max(), cap_pixel_val)
                if self.conf.train.use_shadow_mask:
                    shadow_mask = self.dataset.gen_shadow_mask_at(image_perm[img_idx], pixels_x.squeeze(dim=-1), pixels_y.squeeze(dim=-1))
                    # combined_mask = combined_mask * shadow_mask
                else:
                    shadow_mask = None
                mitsuba_loss, mitsuba_losses, extra_output = self.mitsuba_trainer(render_out, rays_o, rays_d, near, far, light_to_world, light_o, mitsuba_light_lumen, self.iter_step, self.end_iter,true_rgb, image_perm[img_idx], pixels_x, pixels_y, combined_mask, shadow_mask)
                loss = mitsuba_loss
                print(loss)
                self.writer.add_scalar('Loss/mitsuba_loss', mitsuba_loss, self.iter_step)
                for k,v in mitsuba_losses.items():
                    self.writer.add_scalar('Loss/mitsuba_{}'.format(k), v, self.iter_step)
                if "light_conv_factor" in extra_output:
                    self.writer.add_scalar('Loss/mitsuba_light_conv_factor', extra_output["light_conv_factor"], self.iter_step)

                t_mitsuba_end = time.time()
            if not self.disable_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            t_backward_end = time.time()

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint(is_latest=False)
            
            if self.iter_step % self.save_freq_latest == 0:
                self.save_checkpoint(is_latest=True)



            
            if self.iter_step % self.val_freq == 0:
                
                val_idx = validation_set[np.random.randint(len(validation_set))]
                print(validation_set, val_idx)
                self.validate_image(log_to_tb=False, idx=int(val_idx))

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(world_space=False, resolution=128)
                pass
            t_optimizer_right_before = time.time()
            #self.optimizer.step()
            if (self.iter_step + 1) % self.conf.train.accum_grad == 0:
                has_nan = False
                t_find_nan_end = time.time()
                if not has_nan:
                    if self.mitsuba_trainer is not None:
                        self.mitsuba_trainer.step()
                    print("optimizer step")
            print("loss")
            self.iter_step += 1
            t_optimizer_end = time.time()
            self.update_learning_rate()
            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()
            t_misc_end = time.time()
    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor
            break ## only update the first group
        
    def file_backup(self):
        dir_lis = self.conf.general.recording
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        # copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location="cpu")
        # if self.conf.model.neus_renderer.n_outside == 0:
        #     print("skip loading nerf")
        # else:
        #     self.nerf_outside.load_state_dict(checkpoint['nerf'])
        if self.conf.train.migrate_ckpt and checkpoint['sdf_network_fine']["lin8.bias"].shape[0] == 257:
            print("enable migration from old checkpoint")
            sdf_state_dict = self.sdf_network.state_dict()
            for k,v in sdf_state_dict.items():
                sdf_state_dict[k] = v.cpu()
            ckpt_bias = checkpoint['sdf_network_fine']["lin8.bias"]
            curr_bias = sdf_state_dict["lin8.bias"]


            checkpoint['sdf_network_fine']["lin8.bias"] = torch.cat(
                [
                    ckpt_bias[0:1], # 1
                    curr_bias[1:10], # 9
                    ckpt_bias[1:257] # 256
                ]
            ) # 266

            ckpt_weight_g = checkpoint["sdf_network_fine"]["lin8.weight_g"]
            curr_weight_g = sdf_state_dict["lin8.weight_g"]

            checkpoint['sdf_network_fine']["lin8.weight_g"] = torch.cat(
                [
                    ckpt_weight_g[0:1], # 1
                    curr_weight_g[1:10], # 9
                    ckpt_weight_g[1:257] # 256
                ]
            ) # 266
            ckpt_weight_v = checkpoint["sdf_network_fine"]["lin8.weight_v"]
            curr_weight_v = sdf_state_dict["lin8.weight_v"]
            checkpoint['sdf_network_fine']["lin8.weight_v"] = torch.cat(
                [
                    ckpt_weight_v[0:1], # 1
                    curr_weight_v[1:10], # 9
                    ckpt_weight_v[1:257] # 256
                ]
            ) # 266

            
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        try:
            self.color_network.load_state_dict(checkpoint['color_network_fine'])
        except:
            traceback.print_exc()
        print("Skipping optimizer state when loading NEUS checkpoint")
        self.iter_step = checkpoint['iter_step']
        if self.rgb_loss_type == 'adpt_l2_log_lin':
            self.loss_func.load_state_dict(checkpoint["loss_func"])
            pass
        
        # load rng stat
        import random
        try:
            random.setstate(checkpoint['rng_state']['random'])
            np.random.set_state(checkpoint['rng_state']['numpy'])
            torch.set_rng_state(checkpoint['rng_state']['torch'])
            torch.cuda.set_rng_state(checkpoint['rng_state']['torch_cuda'])
        except:
            traceback.print_exc()
        logging.info('End')

    def save_checkpoint(self, is_latest):
        import random
        checkpoint = {
            # 'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "torch_cuda": torch.cuda.get_rng_state(),
                "random": random.getstate()
            }
        }
        if self.rgb_loss_type == 'adpt_l2_log_lin':
            checkpoint["loss_func"] = self.loss_func.state_dict()
            pass
        
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        if not is_latest:
            file = 'ckpt_{:0>6d}.pth'.format(self.iter_step)
        else:
            file = 'latest.pth'

        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', file))
        if self.mitsuba_trainer is not None:
            mitsuba_dir = os.path.join(self.base_exp_dir, 'mitsuba')
            os.makedirs(mitsuba_dir, exist_ok=True)
            self.mitsuba_trainer.save_checkpoint(self.iter_step, mitsuba_dir, is_latest=is_latest)
    @torch.no_grad()
    def save_normal_and_depth(self, path):
        normal_maps, depth_maps = [], []

        for idx in tqdm(range(self.dataset.n_images)):
            rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=1)
            depth_distance_ratio = self.dataset.dist_to_depth_map(idx, resolution_level=1).cpu().numpy()
            light_o, light_lum = self.dataset.gen_light_params(idx)
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

            out_normal_fine = []
            out_dist_fine = []

            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

                render_out = self.renderer.render(rays_o_batch,
                                                rays_d_batch,
                                                light_o,
                                                light_lum,
                                                near,
                                                far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                background_rgb=background_rgb)

                def feasible(key): return (key in render_out) and (render_out[key] is not None)

                assert feasible('gradients') and feasible('weights')
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                dists = render_out['z'] * render_out['weights'][:, :n_samples]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                    dists = dists * render_out['inside_sphere'] 
                normals = normals.sum(dim=1).detach().cpu().numpy()
                dists = (dists.sum(dim=1) / render_out['weights'][:, :n_samples].sum(dim=1)).detach().cpu().numpy()
                out_normal_fine.append(normals)
                out_dist_fine.append(dists)
                del render_out

            normal_img = np.concatenate(out_normal_fine, axis=0).reshape(H,W,3)
            normal_img = normal_img / (1e-10 + np.linalg.norm(normal_img, axis=-1, keepdims=True))
            normal_maps.append(normal_img)

            dist_img = np.concatenate(out_dist_fine, axis=0).reshape(H,W)
            depth_img = depth_distance_ratio * dist_img
            depth_maps.append(depth_img)
        
        np.savez(path, depth_maps=np.stack(depth_maps,axis=0), normal_maps=np.stack(normal_maps,axis=0))
    
    @torch.no_grad()
    def validate_mitsuba_all(self, resolution_level=1, mitsuba_repeats=1):
        for i in range(self.dataset.n_images):
            self.validate_mitsuba(i, resolution_level, mitsuba_repeats=mitsuba_repeats)
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def validate_mitsuba(self, idx=-1, resolution_level=-1, mitsuba_repeats=1):

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, color, mask, pixels_x, pixels_y = self.dataset.gen_rays_at(idx, resolution_level=resolution_level, return_color=True, return_mask=True, return_pixels=True)
        light_o, light_lum = self.dataset.gen_light_params(idx)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        color = color.reshape(-1, 3).split(self.batch_size)
        mask = mask.reshape(-1, 3).split(self.batch_size)
        pixels_x = pixels_x.reshape(-1).split(self.batch_size)
        pixels_y = pixels_y.reshape(-1).split(self.batch_size)
        render_outs = []
        albedo_outs = []
        roughness_outs = []
        for batch_idx, (rays_o_batch, rays_d_batch, color_batch, mask_batch, pixels_x_batch, pixels_y_batch) in enumerate(zip(rays_o, rays_d, color, mask, pixels_x, pixels_y)):
            print("batch idx", batch_idx)
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            pts, zs, normals, valid_mask = self.mitsuba_trainer.physical_shader_gi.renderer.cast_ray_torch(rays_o_batch, rays_d_batch)
            dirs = rays_d_batch
            pts = pts.reshape(-1, 3)
            dirs = dirs.reshape(-1, 3)
            zs = zs.reshape(-1)
            normals = normals.reshape(-1, 3)
            render_out = {
                "z": zs,
                "gradients": normals,
                "valid_mask": valid_mask
            }
            # render_out = render_utils.detach_rec(render_out, to_cpu=True)
            light_to_world, mitsuba_light_lumen = self.dataset.gen_light_params_pose(idx)
            light_o = light_to_world[:3, 3]
            img = None
            albedo = None
            roughness = None
            for i in range(mitsuba_repeats):
                print('repeat', i)
                with torch.no_grad():
                    with dr.suspend_grad():            
                        print("step", batch_idx*mitsuba_repeats+i)
                        out = self.mitsuba_trainer.get_required_output(render_out, rays_o_batch, rays_d_batch, near, far, light_to_world, light_o, light_lum, idx, pixels_x_batch, pixels_y_batch, step=batch_idx*mitsuba_repeats+i, geometry_type_name="mesh_bsdf_adjoint")
                if img is None:
                    img = out['gi_rendered']
                else:
                    img += out['gi_rendered']
                if albedo is None:
                    albedo = out['albedo']
                else:
                    albedo += out['albedo']
                if roughness is None:
                    roughness = out['roughness']
                else:
                    roughness += out['roughness']
                
            img = img / mitsuba_repeats
            albedo = albedo / mitsuba_repeats
            roughness = roughness / mitsuba_repeats

            render_outs.append(img.detach().cpu())
            albedo_outs.append(albedo.detach().cpu())
            roughness_outs.append(roughness.detach().cpu())


        img = torch.cat(render_outs, dim=0)
        img = img.reshape([H, W, 3])

        albedo = torch.cat(albedo_outs, dim=0)
        albedo = albedo.reshape([H, W, 3])
        roughness = torch.cat(roughness_outs, dim=0)
        roughness = roughness.reshape([H, W, 3])[:, :, 0]

        img_dir = os.path.join(self.base_exp_dir,
                            'mitsuba_img')
        os.makedirs(img_dir, exist_ok=True)
        img_path_exr = os.path.join(img_dir, '{:0>8d}_{}.exr'.format(self.iter_step, idx))

        cv.imwrite(img_path_exr, img.cpu().numpy()[:, :, ::-1])

        albedo_dir = os.path.join(self.base_exp_dir,
                            'mitsuba_albedo')
        os.makedirs(albedo_dir, exist_ok=True)
        albedo_path_exr = os.path.join(albedo_dir, '{:0>8d}_{}.exr'.format(self.iter_step, idx))
        cv.imwrite(albedo_path_exr, albedo.cpu().numpy()[:, :, ::-1])

        roughness_dir = os.path.join(self.base_exp_dir,
                        'mituba_roughness')
        os.makedirs(roughness_dir, exist_ok=True)
        roughness_path_exr = os.path.join(roughness_dir, '{:0>8d}_{}.exr'.format(self.iter_step, idx))

        cv.imwrite(roughness_path_exr, roughness.cpu().numpy())

        return img, albedo, roughness

    def validate_image(self, idx=-1, resolution_level=-1, log_to_tb=False, printf=print, basic_only=False, mitsuba_repeats=1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        printf('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        print("s is",  1/self.deviation_network(torch.zeros([1,3])))
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, color, mask, pixels_x, pixels_y = self.dataset.gen_rays_at(idx, resolution_level=resolution_level, return_color=True, return_mask=True, return_pixels=True)
        # print("+=============================+ warning: validate_image============================ using first frame light params =============================+")
        light_o, light_lum = self.dataset.gen_light_params(idx)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        # imageio.imwrite("color_test.png", (color*255).detach().numpy().astype(np.uint8))
        color = color.reshape(-1, 3).split(self.batch_size)
        mask = mask.reshape(-1, 3).split(self.batch_size)
        pixels_x = pixels_x.reshape(-1, 1).split(self.batch_size)
        pixels_y = pixels_y.reshape(-1, 1).split(self.batch_size)
        # out_rgb_fine = []
        # out_normal_fine = []
        # out_intersect_normal = []
        # out_s_grad = []
        # out_gt = []
        # out_brdf_params = []
        # out_mitsuba_rgb = []
        render_outs = []
        color_outs = []
        mask_outs = []
        shadow_mask_outs = []
        pixels_x_out = []
        pixels_y_out = []
        val_pts_out = []
        for rays_o_batch, rays_d_batch, color_batch, mask_batch, pixels_x_batch, pixels_y_batch in zip(rays_o, rays_d, color, mask, pixels_x, pixels_y):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = {}
            pts, zs, normals, valid_mask = self.mitsuba_trainer.physical_shader_gi.renderer.cast_ray_torch(rays_o_batch, rays_d_batch)
            dirs = rays_d_batch
            pts = pts.reshape(-1, 3)
            dirs = dirs.reshape(-1, 3)
            zs = zs.reshape(-1)
            normals = normals.reshape(-1, 3)
            render_out["z"] = zs
            render_out["gradients"] = normals
            render_out["valid_mask"] = valid_mask
            render_out = render_utils.detach_rec(render_out, to_cpu=True)
            render_outs.append(render_out)
            color_outs.append(color_batch)
            cap_pixel_val = self.dataset.cap_pixel_val(idx)
            mask_batch = mask_batch * (color_batch < cap_pixel_val) 
            if self.conf.train.use_shadow_mask:
                shadow_mask = self.dataset.gen_shadow_mask_at(idx, pixels_x_batch.long().squeeze(dim=-1), pixels_y_batch.long().squeeze(dim=-1))
                # print("shadow mask", shadow_mask)
                # mask_batch = mask_batch * shadow_mask
                # mask_batch = shadow_mask
                shadow_mask_outs.append(shadow_mask)
            else:
                shadow_mask_outs.append(None)
            pixels_x_out.append(pixels_x_batch)
            pixels_y_out.append(pixels_y_batch)
            mask_outs.append(mask_batch)

        if self.mitsuba_renderer is not None:
            with torch.no_grad():
                light_to_world, mitsuba_light_lumen = self.dataset.gen_light_params_pose(idx)
                light_o = light_to_world[:3, 3]
                out_dir = os.path.join(self.base_exp_dir,
                                    'mitsuba_rgb')
                os.makedirs(out_dir, exist_ok=True)
                
                print("test before val")
                self.mitsuba_trainer.validate_image(color_outs, out_dir, idx, H, W, render_outs, rays_o, rays_d, self.dataset, light_to_world, light_o, mitsuba_light_lumen, step=self.iter_step, max_steps=self.end_iter, light_idx=idx, mask=mask_outs, pixels_x=pixels_x_out, pixels_y=pixels_y_out, basic_only=basic_only, repeat=mitsuba_repeats, shadow_mask=shadow_mask_outs)
                print("test after val")


        # return img
    @torch.no_grad()
    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        rays_o_0, rays_d_0 = self.dataset.gen_rays_between(idx_0, idx_1, 0.0, resolution_level=resolution_level)

        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        light_o, light_lum = self.dataset.gen_light_params_between(idx_0, idx_1, ratio)
        rays_o_0 = rays_o_0.reshape(-1, 3).split(self.batch_size)
        rays_d_0 = rays_d_0.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_o_0_batch, rays_d_batch, rays_d_0_batch in zip(rays_o, rays_o_0, rays_d, rays_d_0):
            near, far = self.dataset.near_far_from_sphere(rays_o_0_batch, rays_d_0_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_0_batch,
                                              rays_d_0_batch,
                                              light_o.cuda(),
                                              light_lum.cuda(),
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)
            render_at_intersect = False
            if render_at_intersect:
                # print("this path (add 0.01 ray_d to pts)")
                # pts_intersect = render_out["pts_intersect"]
                # pts_intersect += rays_d_batch * 0.05
                sdf = render_out['sdf']
                z_vals = render_out["z_vals"]
                mid_z_vals = render_out["mid_z_vals"]
                from models.renderer import locate_intersection
                sdf_intersect, pts_intersect = locate_intersection(sdf, z_vals, mid_z_vals, rays_o_0_batch, rays_d_0_batch)
                pts_intersect = pts_intersect.squeeze(dim=1) # B x 3

                grads = self.sdf_network.gradient(pts_intersect).squeeze()
                sdf_nn_output = self.sdf_network(pts_intersect)
                rgb,_ = self.color_network(pts_intersect, grads, rays_d_batch, light_o.cuda(), light_lum.cuda(), torch.zeros((rays_d_batch.shape[0], 0), device=rays_d_batch.device), sdf_nn_output[:, 1+self.renderer.n_brdf_dim:])
                out_rgb_fine.append(rgb.detach().cpu().numpy())
                pass
            else:
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3])
        return img_fine

    @torch.no_grad()
    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0, simplify=False, bake_texture_maps=False, texture_resolution=2048, log_to_tb=False, save_to=None):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32, device=self.device)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32, device=self.device)

        print(f"Extracting mesh from marching cubes at resolution {resolution}...")
        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, device=self.device)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        mesh = trimesh.Trimesh(vertices, triangles)
        vertices, triangles = mesh.vertices, mesh.faces

        if save_to is None:
            save_to = os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}'.format(self.iter_step))

        if simplify:
            mesh = mesh.as_open3d.simplify_quadric_decimation(131072, (0.5/resolution)**2)
            print(f"Simplified mesh: {vertices.shape[0]} verts, {triangles.shape[0]} faces -> {len(mesh.vertices)} verts, {len(mesh.triangles)} faces.")
            vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        
        
        if bake_texture_maps:
            save_dir = os.path.join('{}_textures'.format(save_to))
            os.makedirs(save_dir, exist_ok=True)
            print(f"Running UV unwraping (this can take a few minutes) ...")

            coords, vertices, triangles, uv = generate_uv_map(vertices, triangles, texture_resolution)

            print(f"Baking textures at resolution {coords.shape[0]}X{coords.shape[1]}.")
            if self.conf["model.brdf_settings.type"] != "diffuse":
                brdf_params = batch_feed(self.renderer.extract_shading_params, coords.reshape(-1,3))
                pass
            else:
                brdf_params = batch_feed(self.renderer.extract_shading_diffuse, coords.reshape(-1,3))
                pass
            
            ks = ['object_normal', 'subsurface', 'metallic', 'specular', 'clearcoat', 'roughness', 'clearcoat_gloss', 'base_color'] if self.conf["model.brdf_settings.type"] != "diffuse" else ['object_normal', 'base_color']
            for k in ks: 
                if k == 'object_normal':
                    texture = (brdf_params[k].reshape(coords.shape[0], coords.shape[1], -1) + 1) * 255 * 0.5
                else:
                    texture = brdf_params[k].reshape(coords.shape[0], coords.shape[1], -1) * 255
                if k == 'object_normal' or k == 'base_color':
                    cv.imwrite(os.path.join(save_dir, f"{k}.png"), texture.clip(0, 255)[...,::-1]) # opencv saves in BGR format
                else:
                    cv.imwrite(os.path.join(save_dir, f"{k}.png"), texture.clip(0, 255)) # gray image
            
            cv.imwrite(os.path.join(save_dir, f"coords.exr"), coords.reshape(coords.shape[0], coords.shape[1], -1).astype(np.float32)[...,::-1])
            

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
        
        output_path = '{}_lowres.ply'.format(save_to)
        if bake_texture_maps:
            xatlas.export('{}.obj'.format(save_to), vertices, triangles, uv)
            print(f"Saved mesh to '{save_to}.obj' and textures under '{save_dir}'.")
        else:
            trimesh.Trimesh(vertices, triangles).export(output_path)
        
        if log_to_tb and self.writer:
            self.writer.add_mesh('shape', torch.from_numpy(vertices).unsqueeze(0), faces=torch.from_numpy(triangles).unsqueeze(0), global_step=self.iter_step)
        logging.info('End')
        # print("WARNING temporarily enable check_mesh")
        # self.check_mesh_face_visibility(output_path, output_path)
    @torch.no_grad()
    def validate_mesh_hires(self, world_space=False, resolution=64, threshold=0.0, simplify=False, bake_texture_maps=False, bake_vert_maps=False, texture_resolution=2048, log_to_tb=False, save_to=None):
        # bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        # bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        bound_min = (-1.0, -1.0, -1.0)
        bound_max = (1.0, 1.0, 1.0)

        print(f"Extracting mesh from marching cubes at resolution {resolution}...")
        mesh =\
            self.renderer.extract_geometry_hires(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        # mesh = trimesh.Trimesh(vertices, triangles)
        vertices, triangles = mesh.vertices, mesh.faces

        if save_to is None:
            save_to = os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}{}'.format(self.iter_step, "_world" if world_space else ""))


        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
        mesh = trimesh.Trimesh(vertices, triangles)
        if bake_vert_maps:
            logging.info('baking vert maps')
            vertices_torch = torch.as_tensor(vertices.reshape(-1,3)).cuda()
            vertices_torch = vertices_torch.to(torch.float32)
            albedo = batch_feed2(self.mitsuba_trainer.material_renderers["albedo"],vertices_torch.reshape(-1,3) )
            roughness = batch_feed2(self.mitsuba_trainer.material_renderers["roughness"], vertices_torch.reshape(-1,3))
            albedo = albedo.cpu().numpy()
            # albedo = utils.lin2srgb(albedo).reshape(-1, 3)
            roughness = roughness.cpu().numpy()
            # roughness = utils.lin2srgb(roughness).reshape(-1, 3)

            vertex_attributes = {
                "albedo_r": albedo[:, 0],
                "albedo_g": albedo[:, 1],
                "albedo_b": albedo[:, 2],
                "roughness_0": roughness[:, 0],
            }
            print("albedo", albedo.reshape(-1,3).shape)
            
            print("roughness", roughness.reshape(-1,3).shape)
            print("vertices", vertices.shape)
            print("triangles", triangles.shape)
            mesh.vertex_attributes = vertex_attributes
            print(mesh.vertex_attributes)
        save_to_full = '{}.ply'.format(save_to)
        
        
        mesh.export(save_to_full, vertex_normal=True, include_attributes=True)
        # trimesh.exchange.ply.export_ply(mesh, vertex_normal=True)
        
        
        logging.info('End')
        
        return save_to_full


    @torch.no_grad()
    def validate_grid(self, world_space=False, resolution=64, threshold=0.0):
        print(f"Extracting sdf from at resolution {resolution}...")
        object_bbox_min = np.array([-1.0, -1.0, -1.0, 1.0])
        object_bbox_max = np.array([ 1.0,  1.0,  1.0, 1.0])

        bound_min = torch.tensor(object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(object_bbox_max, dtype=torch.float32)

        sdf_vals =\
            self.renderer.extract_fields(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'sdfs'), exist_ok=True)
        save_to = os.path.join(self.base_exp_dir, 'sdfs', '{:0>8d}.npy'.format(self.iter_step))
        np.save(save_to, sdf_vals)
        pass
    @torch.no_grad()
    def interpolate_view(self, img_idx_0, img_idx_1, n_frames = 60):
        images = []
        video_dir = os.path.join(self.base_exp_dir, 'render', '{:0>8d}_{}_{}'.format(self.iter_step, img_idx_0, img_idx_1))
        os.makedirs(video_dir, exist_ok=True)

        for i in range(n_frames):
            print(i)
            # images.append(self.render_novel_image(img_idx_0,
            #                                       img_idx_1,
            #                                       np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
            #                                       resolution_level=4))
            image = self.render_novel_image(img_idx_0,
                                            img_idx_1,
                                            np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                                            resolution_level=2)
                                            # resolution_level=2)
            print(os.path.join(video_dir, "{}.png".format(i)))
            imageio.imwrite(os.path.join(video_dir, "{}.png".format(i)), (utils.lin2srgb(image)*256).clip(0, 255).astype(np.uint8))
            
        # for i in range(n_frames):
        #     images.append(images[n_frames - i - 2])

        # fourcc = cv.VideoWriter_fourcc(*'h264')
        # h, w, _ = images[0].shape
        # writer = cv.VideoWriter(os.path.join(video_dir,
        #                                      '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
        #                         fourcc, 30, (w, h))

        # for image in images:
        #     writer.write((np.flip(utils.lin2srgb(image),axis=-1)*256).clip(0, 255).astype(np.uint8))
        # for i, image in enumerate(images):
        #     imageio.imwrite(os.path.join(video_dir, "{}.png".format(i)), (np.flip(utils.lin2srgb(image),axis=-1)*256).clip(0, 255).astype(np.uint8))

        # writer.release()
def simple_kv_parser(txt):
    pairs = txt.split(',')
    ret = {}
    for p in pairs:
        k, v = p.split('=')
        ret[k] = v
    return ret

@hydra.main(version_base=None, config_path="config_hydra", config_name="main")
def main(args: DictConfig):
    print('Starting GLOW material runner')


    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    print(OmegaConf.to_yaml(args))
    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.exp_suffix, args.distill_suffix, args.distill_conf, args.disable_scaler, args.set_s, args.mitsuba_renderer, args.latest_model_name)

    if args.mode.startswith('train'):
        if len(args.mode.split('_')) == 2:
            runner.train(args.mode.split('_')[1])
        else:
            runner.train()
        dr.flush_malloc_cache()
        torch.cuda.empty_cache()

    elif args.mode == 'validate_mesh_hires':
        with torch.no_grad():
            runner.validate_mesh_hires(world_space=args.is_world, resolution=1024, threshold=args.mcube_threshold, simplify=False, bake_texture_maps=not args.disable_bake_texture, bake_vert_maps=True, texture_resolution=4096)

    elif args.mode.startswith('validate_mesh'):
        with torch.no_grad():
            if len(args.mode.split('_')) == 2:
                resolution = 512
            else:
                assert len(args.mode.split('_')) == 3
                resolution = int(args.mode.split('_')[2])
                pass
            
            runner.validate_mesh(world_space=args.is_world, resolution=resolution, threshold=args.mcube_threshold, simplify=False, bake_texture_maps=not args.disable_bake_texture, texture_resolution=4096)

    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        with torch.no_grad():
            _, img_idx_0, img_idx_1 = args.mode.split('_')
            img_idx_0 = int(img_idx_0)
            img_idx_1 = int(img_idx_1)
            runner.interpolate_view(img_idx_0, img_idx_1)
    elif args.mode.startswith('validate_image'):  # Interpolate views given two image indices
        with torch.no_grad():
            if len(args.mode.split('_')) == 2:
                resolution_level = 2
            else:
                assert len(args.mode.split('_')) == 3
                resolution_level = int(args.mode.split('_')[2])

        runner.validate_images(resolution_level, mitsuba_repeats=1)
    elif args.mode.startswith('validate_single_image'):  # Interpolate views given two image indices
        with torch.no_grad():
            if len(args.mode.split('_')) == 4:
                img_idx = int(args.mode.split('_')[3])

            else:
                raise RuntimeError(args.mode.split('_'))
        runner.validate_image(img_idx, mitsuba_repeats=1)

    elif args.mode.startswith("validate_grid"):
        with torch.no_grad():
            if len(args.mode.split('_')) == 2:
                resolution = 1024
            else:
                assert len(args.mode.split('_')) == 3
                resolution = int(args.mode.split('_')[2])

            runner.validate_grid(resolution=resolution)
    elif args.mode == "validate_mitsuba":
        runner.validate_mitsuba_all( mitsuba_repeats=1)
    else:
        raise RuntimeError(f"Unknown mode {args.mode}")
            

if __name__ == '__main__':
    main()
