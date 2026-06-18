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
            self.loss_func.to(self.device)
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
        # print(list(self.color_network.ambient_net.parameters()))
        # print( list(self.color_network.parameters()))
        # exit()

        params = [{'params': params_to_train, "lr": self.learning_rate} ]
        if hasattr(self.color_network, "gamma"):
            params += [{'params': [self.color_network.gamma], "lr": self.learning_rate_flash}]
        
        # if self.mitsuba_renderer is not None:
        #     mitsuba_params = {k:v for k, v in self.mitsuba_trainer.named_parameters() if v.requires_grad}
        #     print("mitsuba_params", mitsuba_params.keys())
        #     # params += [{
        #     #     "params": list(mitsuba_params.values()),
        #     #     "lir": self.learning_rate
        #     # }]
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

            img = img.clip(0, self.dataset.saturation_intensity)
            img_rendered = img_rendered.clip(0, self.dataset.saturation_intensity)

            img[mask < 0.5] = 0
            img_rendered[mask < 0.5] = 0

            psnr.append(peak_signal_noise_ratio(img, img_rendered, data_range=img.max()))
            # ssim.append(structural_similarity(img, img_rendered, data_range=img.max(),multichannel=True,channels_axis=-1))

            cv.imwrite(os.path.join(self.base_exp_dir, 'novel_view', f"{i:02}.exr"), np.concatenate([img_rendered[...,::-1], mask.reshape(img_rendered.shape[:-1] + (-1,))[...,:1]], axis=-1))
            cv.imwrite(os.path.join(self.base_exp_dir, 'gt', f"{i:02}.exr"), img[..., ::-1])

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
        # print("prune invisible faces")
        mesh_path = self.mitsuba_renderer.mesh_path
        was_dummy = mesh_path == "dummy"
        try:
            
            # print(self.iter_step >= 49999, mode != "validate_mesh", mesh_path=="dummy", self.iter_step < self.end_iter, (not self.mitsuba_renderer.config.render.no_bsdf_sample))
            if self.iter_step >= 49999 and mode != "validate_mesh"  and mesh_path=="dummy" and self.iter_step < self.end_iter and (not self.mitsuba_renderer.config.render.no_bsdf_sample):
                mesh_path = self.validate_mesh_hires(world_space=False, resolution=1024, threshold=0, simplify=False, bake_texture_maps=False, texture_resolution=4096)
        except:
            import traceback as tb
            tb.print_exc()
            print("val mesh error!!!!")
        if was_dummy and mesh_path != "dummy" and self.conf.train.prune_outside_cam:
            self.check_mesh_face_visibility(mesh_path, mesh_path)
        scene, integrator, learned_info, ckpt_path = models.giphysicalshader.init_scene(self.mitsuba_renderer.scene_path, mesh_path, self.mitsuba_renderer.out_dir, self.mitsuba_renderer.config)        
        self.mitsuba_trainer = models.giphysicalshader.PhysicalShadingTrainer(self.sdf_network, self.color_network, self.renderer, self.deviation_network, scene, integrator, learned_info, ckpt_path, self.mitsuba_renderer.config, self.dataset).to(self.device)
        pass

    @torch.no_grad
    def check_mesh_face_visibility(self, mesh_file, out_mesh):
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer
        import trimesh
        ply_mesh = trimesh.load(mesh_file)
        V = torch.as_tensor(ply_mesh.vertices, dtype=torch.float32, device=self.device)
        F = torch.as_tensor(ply_mesh.faces, dtype=torch.int64, device=self.device)
        mesh = Meshes(verts=[V], faces=[F])

        # vis_per_cam = []
        n_images = self.dataset.n_images
        # vis_faces = set()
        assert self.dataset.images is not None
        H, W = self.dataset.images.shape[1], self.dataset.images.shape[2]
        H, W = H , W 
        face_mask = torch.zeros(ply_mesh.faces.shape[0], dtype=torch.bool, device=self.device)
        for img_idx in range(n_images):
            print("img_idx", img_idx)
            intrinsics = self.dataset.intrinsics_all[img_idx]
            pose = self.dataset.pose_all_inv[img_idx]

            R = torch.as_tensor(pose[:3, :3], dtype=torch.float32, device=self.device)
            T = torch.as_tensor(pose[:3, 3], dtype=torch.float32, device=self.device)
            # print("principal", torch.tensor([[intrinsics[0, 2], intrinsics[1, 2]]], dtype=torch.float32, device=self.device))
            # print("img_size", torch.tensor([[W, H]], dtype=torch.float32, device=self.device))
            cameras = PerspectiveCameras(
                R=R.T.unsqueeze(0), # col convention
                T=T.unsqueeze(0),
                # K=torch.as_tensor(intrinsics, dtype=torch.float32, device=self.device).unsqueeze(0),
                focal_length=torch.tensor([[intrinsics[0, 0], intrinsics[1, 1]]], dtype=torch.float32, device=self.device),
                principal_point=torch.tensor([[intrinsics[0, 2], intrinsics[1, 2]]], dtype=torch.float32, device=self.device),
                image_size=torch.tensor([[H, W]], dtype=torch.float32, device=self.device),
                in_ndc=False,   # use pixel units for focal/principal point
                device=self.device,
            )
            # print(len(cameras))
            rast = MeshRasterizer(
                cameras=cameras,
                raster_settings=RasterizationSettings(
                    image_size=(H, W),
                    faces_per_pixel=1,     # just need the closest face
                    cull_backfaces=False,   # optional: back-face culling
                    blur_radius=0.0,
                    bin_size=None,
                    max_faces_per_bin=None,
                    max_faces_opengl=100_000_000
                ),
            )
            fragments = rast(mesh)
            print("after res")
            pix_to_face = fragments.pix_to_face[0, ..., 0]
            print("access pix2face")
            print("prepare to assign")
            face_mask[pix_to_face[pix_to_face >= 0]] = True
            print("update face mask")
            pass
        # vis_faces = np.array(vis_faces)
        # assert len(vis_faces) > 0
        
        # face_mask_real = face_mask[1:].cpu().numpy()
        ply_mesh.update_faces(face_mask.cpu().numpy())
        ply_mesh.export(out_mesh)
    
    def train(self, sample_mode="batch"):
        print("s is",  1/self.deviation_network(torch.zeros([1,3])))
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        # if self.conf["model.sdf_type"] != "ngp":
        #     self.update_learning_rate()
        res_step = max(0, self.end_iter - self.iter_step)
        image_perm = self.get_image_perm()
        scaler = torch.cuda.amp.GradScaler()
        validation_set = np.linspace(0, self.dataset.n_images, num=30, dtype=int, endpoint=False)
        # print("self.mitsuba_renderer", self.mitsuba_renderer)
        # if self.mitsuba_trainer is None:
        for iter_i in tqdm(range(res_step)):
            
            print("start iter")
            self.update_learning_rate()
            # for g in self.optimizer.param_groups:
            #     print("real lr in train", g["lr"])
            # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
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
            actual_img_idx = image_perm[img_idx]
            # print("=============================================== forcing img_perm to chosen one, 172")
            # actual_img_idx = 172
            cap_pixel_val = self.dataset.cap_pixel_val(actual_img_idx)

            seed = int(time.time() * 1000) % 1000000 + iter_i

            def closure(mode="rgb", x_shift=0, y_shift=0):
                if sample_mode == "patch":  
                    data = self.dataset.gen_random_rays_patch(actual_img_idx, int(np.sqrt(self.rgb_batch_size)), mode=="rgb", shift=(x_shift,y_shift), seed=seed)
                else:
                    data, pixels = self.dataset.gen_random_rays_at(actual_img_idx, self.rgb_batch_size, mode=="rgb", shift=(x_shift,y_shift), seed=seed)

                light_o, light_lumen = self.dataset.gen_light_params(actual_img_idx)
                #print("light_o", light_o, "light_lumen", light_lumen)
                rays_o, rays_d, true_rgb, mask = data[..., :3], data[..., 3: 6], data[..., 6: 9], data[..., 9: 10]#, data[..., 10:11], data[..., 11:12]
                pixels_x, pixels_y = pixels[..., 0:1], pixels[..., 1:2]
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

                background_rgb = None
                if self.use_white_bkgd:
                    background_rgb = torch.ones([1, 3])

                mask = (mask > 0.5).float()
                if mode == "rgb":
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                        render_out = self.renderer.render(rays_o, rays_d, light_o, light_lumen, near, far, 
                                                          background_rgb=background_rgb,
                                                          cos_anneal_ratio=self.get_cos_anneal_ratio())
                        if self.is_distill:
                            teacher_nn_sdf, student_nn_sdf = self.distiller.sample_student_teacher(rays_o, rays_d, near, far)
                            render_out["teacher_nn_sdf"] = teacher_nn_sdf
                            render_out["student_nn_sdf"] = student_nn_sdf
                elif mode == 'alpha':
                    render_out = self.renderer.render_alpha(rays_o, rays_d, light_o, light_lumen, near, far, 
                                                    background_rgb=background_rgb,
                                                    cos_anneal_ratio=self.get_cos_anneal_ratio())
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1)
                # normals = normals / (1e-10 + torch.linalg.norm(normals, dim=-1, keepdim=True)) # consider remove this
                # normals = render_out['intersect_gradient']
                render_out['normals'] = normals
                # print("hess_err here", render_out['hess_error'])
                return render_out, true_rgb, mask, light_lumen, rays_o, rays_d, light_o, pixels_x, pixels_y

            # def sample_pixels(n_samples):
            #     n_samples = int(np.sqrt(n_samples))
            #     ret = dict()
            #     keys=['color_fine', 's_val', 'cdf_fine', 'gradient_error', 'weight_max', 'weight_sum', 'normals']
            #     if self.is_distill:
            #         keys = keys + ['teacher_nn_sdf', 'student_nn_sdf']
                
            #     for i in np.arange(0.5/n_samples,1,1/n_samples):
            #         for j in np.arange(0.5/n_samples,1,1/n_samples):  
            #             offset_x = i - 0.5
            #             offset_y = j - 0.5

            #             render_out, true_rgb, mask, light_lumen, rays_o, rays_d, light_o = closure("rgb", offset_x, offset_y)
            #             for k in keys:
            #                 ret[k] = (ret.get(k,0) + render_out[k]/(n_samples**2))
            #     assert n_samples == 1
            #     ret['extra_out'] = render_out['extra_out']
            #     ret['weights'] = render_out['weights']
            #     ret['sampled_color'] = render_out['sampled_color']
            #     ret['inside_sphere'] = render_out['inside_sphere']

            #     return ret, true_rgb, mask, light_lumen, rays_o, rays_d, light_o
            
            # t_init_end = time.time()
            # render_out, true_rgb, mask, light_lumen, rays_o, rays_d, light_o = sample_pixels(self.samples_per_pixel)
            render_out, true_rgb, mask, light_lumen, rays_o, rays_d, light_o, pixels_x, pixels_y = closure("rgb", 0, 0)

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']
            weights = render_out['weights']
            sampled_color = render_out['sampled_color']
            inside_sphere = render_out['inside_sphere']
            # back_facing_loss = render_out['back_facing_loss']
            normals = render_out['normals']
            sdf_grad = render_out["sdf_grad"]
            # print("sdf", sdf.requires_grad)
            # sdf.retain_grad()
            z_vals = render_out["z_vals"]
            mid_z_vals = render_out["mid_z_vals"]
            sdf = render_out['sdf']


            t_render_end = time.time()
            # Loss
            valid_pixel_mask = ((color_fine < cap_pixel_val) | (true_rgb < cap_pixel_val)).float()

            if self.mitsuba_renderer is not None and self.mitsuba_renderer.config.render.use_field_occlusion_hint:

                with torch.no_grad():
                    light_to_world, mitsuba_light_lumen = self.dataset.gen_light_params_pose(actual_img_idx)
                    # mitsuba_loss, extra_losses, extra_out = self.mitsuba_trainer()
                    light_o = light_to_world[:3, 3]
                    from models.renderer import locate_intersection
                    sdf_intersect, pts_intersect = locate_intersection(sdf, z_vals, mid_z_vals, rays_o, rays_d)
                    pts_intersect = pts_intersect.squeeze(dim=1) # B x 3
                    light_dir = light_o - pts_intersect # B x 3
                    light_dir_length = torch.norm(light_dir, dim=-1, keepdim=True) # B x 1
                    light_dir = light_dir / light_dir_length # B x 3
                    
                    shadow_near, shadow_far = self.dataset.near_far_from_sphere(pts_intersect, light_dir)
                    shadow_render_out = self.renderer.render_alpha(pts_intersect, light_dir, light_o, light_lumen, shadow_near, light_dir_length, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=1.0)
                    shadow_render_alpha = shadow_render_out["weight_sum"]                
                    em_occlusion = shadow_render_alpha > 0.2

                    valid_pixel_mask = valid_pixel_mask * (~em_occlusion).float()
            #print(valid_pixel_mask)
            if self.conf.train.use_shadow_mask:
                shadow_mask = self.dataset.gen_shadow_mask_at(actual_img_idx, pixels_x.squeeze(dim=-1), pixels_y.squeeze(dim=-1))
                    # combined_mask = combined_mask * shadow_mask
            else:
                shadow_mask = None
            combined_mask =mask * valid_pixel_mask
            if shadow_mask is not None:
                combined_mask = combined_mask * shadow_mask
            color_error = (color_fine - true_rgb) * combined_mask
            # print("color_fine", color_fine)
            # print("true_rgb", true_rgb)
            # import imageio
            # imageio.imwrite("debug_shadow_mask.png", shadow_mask.cpu().numpy())
            # print(shadow_mask)
            # input("write shadow mask")
            mask_sum = (combined_mask).sum() + 1e-5

            # print(color_fine.shape)
            if sample_mode == "patch":  
                color_dssim = loss_dssim(color_fine, true_rgb, mask>0, valid_pixel_mask>0, cap_pixel_val, self.dssim_window_size)
            else:
                color_dssim = 0
            if self.rgb_loss_type == 'l1':
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            elif self.rgb_loss_type == 'l2':
                color_fine_loss = F.mse_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            elif self.rgb_loss_type == 'huber_log_lin':
                color_fine_loss = (F.smooth_l1_loss(color_error/(color_fine.detach()+1e-3), torch.zeros_like(color_error), reduction='sum') / mask_sum)
                
            elif self.rgb_loss_type == 'l2_log_lin':
                color_fine_loss = (F.mse_loss(color_error/(torch.clamp(color_fine.detach(), min=0.01)), torch.zeros_like(color_error), reduction='sum') / mask_sum)
                # color_fine_loss = ((F.mse_loss(color_error, torch.zeros_like(color_error), reduction='none') / (color_fine.detach() + 1e-2)).sum() / mask_sum)
            elif self.rgb_loss_type == 'adpt_l2_log_lin':
                color_fine_loss = self.loss_func(color_fine, color_error).sum() / mask_sum
                # color_fine_loss = ((F.mse_loss(color_error, torch.zeros_like(color_error), reduction='none') / (color_fine.detach() + 1e-2)).sum() / mask_sum)
            elif self.rgb_loss_type == 'orient_l2_log_lin':
                # print("normals", normals.min(), normals.max(), normals.mean())
                # print("normals norm", torch.linalg.norm(normals, dim=-1).min(), torch.linalg.norm(normals, dim=-1).max(), torch.linalg.norm(normals, dim=-1).mean())
                normals_norm = normals / (1e-10 + torch.linalg.norm(normals, dim=-1, keepdim=True)) # consider remove this
                cos = (-normals_norm * rays_d).sum(dim=1) #N
                # print("cos", cos.min(), cos.max(), cos.mean())
                forward_mask = (cos > 0)
                
                angle = torch.acos(torch.clamp(cos, min=0.0, max=1.0))

                enable_mask = (torch.linalg.norm(normals, dim=-1) < 0.3) | ~forward_mask
  
                weight = torch.where(angle < math.pi/4, torch.clamp(torch.cos(2*(angle-math.pi/4)), min=0.0), torch.clamp(torch.cos(self.orient_loss_period*(angle-math.pi/4)), min=0.0))
                weight[enable_mask] = 1.0

                # print("weight", weight.min(), weight.max(), weight.mean())
                color_fine_loss = ((F.mse_loss(color_error/(torch.clamp(color_fine.detach(), min=0.01)), torch.zeros_like(color_error), reduction='none').sum(dim=-1)*weight.detach())).sum() / mask_sum                
                pass
            elif self.rgb_loss_type == 'orient_l1':
                normals_norm = normals / (1e-10 + torch.linalg.norm(normals, dim=-1, keepdim=True)) # consider remove this
                cos = (-normals_norm * rays_d).sum(dim=1) #N
                forward_mask = (cos > 0)
                angle = torch.acos(torch.clamp(cos, min=0.0, max=1.0))
                enable_mask = (torch.linalg.norm(normals, dim=-1) < 0.3) | ~forward_mask
                weight = torch.where(angle < math.pi/4, torch.clamp(torch.cos(2*(angle-math.pi/4)), min=0.0), torch.clamp(torch.cos(self.orient_loss_period*(angle-math.pi/4)), min=0.0))
                weight[enable_mask] = 1.0
                color_fine_loss = (F.l1_loss(color_error, torch.zeros_like(color_error), reduction='none').sum(dim=-1)*weight.detach()).sum() / mask_sum                
            else:
                print(self.rgb_loss_type)
                raise NotImplementedError

            # print("color_fine_loss", color_fine_loss)
            # print("color_error", color_error)
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error
            # print(render_out.keys())
            if self.is_distill:
                teacher_nn_sdf  = render_out["teacher_nn_sdf"]
                student_nn_sdf  = render_out["student_nn_sdf"]
                distill_error =  teacher_nn_sdf - student_nn_sdf
                distill_loss = F.mse_loss(distill_error, torch.zeros_like(distill_error), reduction='mean')
            if self.iter_step % self.conf.train.accum_grad == 0:
                self.optimizer.zero_grad()
            # print("WARNING: color loss is 0.0")
            

            loss = self.color_weight * color_fine_loss +\
                   color_dssim * self.dssim_weight +\
                   eikonal_loss * self.igr_weight
            # loss = 0
            if self.conf.train.hess_error_weight > 0:
                hess_loss =  render_out["hess_error"]
                print("hess_loss", hess_loss)
                self.writer.add_scalar('Loss/hess_loss', hess_loss, self.iter_step)
                loss += self.conf.train.hess_error_weight * hess_loss

            
            if self.is_distill:
                loss += distill_loss
                pass
            if self.conf.train.num_pcd_pts>0:
                pcd_batch = self.dataset.sample_pcd(self.conf.train.num_pcd_pts).cuda()
                sdfs = self.sdf_network.sdf(pcd_batch)[:, 0]
                pcd_loss = torch.abs(sdfs).mean()
                loss += 0.1*pcd_loss

            if self.ncc_weight > 0:
                ncc_loss, gt_var = loss_ncc(color_fine, true_rgb)
                ncc_loss = torch.where(gt_var > 0.015, ncc_loss, torch.zeros_like(ncc_loss))
                ncc_loss = ncc_loss.mean()
            if self.pts_radiance_weight > 0.0:
                pts_3d, pts_2d, gt_color = self.dataset.gen_pts_at(actual_img_idx)
                gt_color = gt_color.cuda()
                img = self.dataset.image_at(actual_img_idx, resolution_level=1)
                sdf_nn_output = self.sdf_network(pts_3d)
                grads = self.sdf_network.gradient(pts_3d).squeeze()
                assert sdf_nn_output.shape[1] == 257
                # print(grads.shape)
                # print(rays_d.shape)
                # print(rays_o.shape)
                # print(rays_o)
                pts_rays_d = pts_3d - rays_o[:1, :]
                pts_rays_d = pts_rays_d / torch.linalg.norm(pts_rays_d, dim=-1, keepdim=True)
                pts_light_o = rays_o[:1, :].repeat(pts_rays_d.shape[0], 1)
                rgb, extra = self.color_network(pts_3d, grads, pts_rays_d, pts_light_o, light_lumen, torch.zeros((rays_d.shape[0], 0), device=rays_d.device), sdf_nn_output[:, 1:1+256])
                pts_err = (rgb - gt_color)
                pts_loss = (F.mse_loss(pts_err/(torch.clamp(rgb.detach(), min=0.01)), torch.zeros_like(rgb), reduction='mean'))
                loss += self.pts_radiance_weight*pts_loss

            t_loss_end = time.time()
            if self.mitsuba_trainer is not None:
                light_to_world, mitsuba_light_lumen = self.dataset.gen_light_params_pose(actual_img_idx)
                light_o = light_to_world[:3, 3]
                self.mitsuba_trainer.init_per_step()
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
                combined_mask = mask * (true_rgb < cap_pixel_val) 
                
                mitsuba_loss, mitsuba_losses, extra_output = self.mitsuba_trainer(render_out, rays_o, rays_d, near, far, light_to_world, light_o, mitsuba_light_lumen, self.iter_step, self.end_iter,true_rgb, actual_img_idx, pixels_x, pixels_y, combined_mask, shadow_mask)
                loss += mitsuba_loss
                self.writer.add_scalar('Loss/mitsuba_loss', mitsuba_loss, self.iter_step)
                for k,v in mitsuba_losses.items():
                    self.writer.add_scalar('Loss/mitsuba_{}'.format(k), v, self.iter_step)
                if "light_conv_factor" in extra_output:
                    self.writer.add_scalar('Loss/mitsuba_light_conv_factor', extra_output["light_conv_factor"], self.iter_step)

                t_mitsuba_end = time.time()
                 # self.writer.add_scalar('Loss/mitsuba_bilateral_loss'.format(k), bilateral_loss, self.iter_step)
                # loss += self.mitsuba_renderer.config.train.reg_weight * bilateral_loss
            # print("before backward")
            if not self.disable_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                # print("albedo mitsuba grad")
                # for name, param in self.mitsuba_trainer.albedo_texture.network.named_parameters():
                #     print(name, param.grad)
            # print("after backward")
                
            t_backward_end = time.time()
            # wait until autograd graph is released in backward
            if self.iter_step > 2500:
                if self.rgb_loss_type == "l1":
                    if color_fine_loss.mean() > 1.0:
                        print("huge loss!", color_fine_loss)
                        self.validate_image(idx=actual_img_idx, log_to_tb=False, resolution_level=4)
            # if self.iter_step > 490000:
            #     if self.rgb_loss_type == "orient_l2_log_lin":
            #         if color_fine_loss.mean() > 1.0:
            #             print("huge loss!", color_fine_loss)
            #             self.validate_image(idx=image_perm[img_idx], log_to_tb=True)
            # if self.mitsuba_trainer is not None:
            #     if mitsuba_loss.mean() > 1.0:
            #         print("mitsuba huge loss!", mitsuba_loss)
            #         self.validate_image(idx=image_perm[img_idx], log_to_tb=True)

            if self.mask_weight > 0:
                render_out_alpha, _, mask_alpha, _, _, _,_,_,_ = closure("alpha")
                # print("alpha weight sum", render_out_alpha['weight_sum'].min(), render_out_alpha['weight_sum'].max(), render_out_alpha['weight_sum'].mean())
                # print("mask alpha", mask_alpha.min(), mask_alpha.max(), mask_alpha.mean())
                if torch.isfinite(render_out_alpha['weight_sum']).all():
                    mask_loss = F.binary_cross_entropy(render_out_alpha['weight_sum'].clip(1e-3, 1.0 - 1e-3), mask_alpha)
                else:
                    print("got nan", render_out_alpha['weight_sum'])
                    print("got nan", mask_alpha)
                (mask_loss * self.mask_weight).backward()

                # loss = loss + mask_loss * self.mask_weight
                pass
            
            #print("flash_gamma grad", self.color_network.gamma.grad)

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

                for name, param in list(self.sdf_network.named_parameters()):
                    # print("sdf network grad", name, param.grad)
                    if param.grad is not None and (not torch.isfinite(param.grad).all()):
                        print("nan gradient found in sdf {}".format(name))
                        has_nan = True
                for name, param in list(self.color_network.named_parameters()):
                    # print(name)
                    if param.grad is not None and (not torch.isfinite(param.grad).all()):
                        print("nan gradient found in sdf {}".format(name) )
                        has_nan = True
                if self.mitsuba_trainer is not None:
                    for name, param in list(self.mitsuba_trainer.named_parameters()):
                        # print(name)
                        if param.grad is not None and (not torch.isfinite(param.grad).all()):
                            print("nan gradient found in sdf {}".format(name) )
                            has_nan = True
                # for name, param in list(self.deviation_network.named_parameters()):
                # self.debug_all_s_grad.append(self.deviation_network.variance.grad.detach().cpu())
                # self.debug_all_get_s_grad.appe
                # print("idx: {} deviation network parameter".format(image_perm[img_idx], name), self.deviation_network.variance.grad, self.debug_all_s_grad, "the mean is", torch.mean(torch.stack(self.debug_all_s_grad)))
                
                t_find_nan_end = time.time()
                if not has_nan:
                    if self.conf.train.sdf_clip_grad_norm is not None:
                        max_grad_norm = self.conf.train.sdf_clip_grad_norm
                        print("clipping sdf grad norm to {}".format(max_grad_norm))
                        torch.nn.utils.clip_grad_norm_(self.sdf_network.parameters(), max_grad_norm)
                    if self.mitsuba_trainer is not None:
                        self.mitsuba_trainer.step()
                    print("optimizer step")
                    if not self.disable_scaler:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        pass
                        self.optimizer.step()
                if not self.conf.train.optimize_geo:
                    for name, param in list(self.sdf_network.named_parameters()):
                        if param.grad is not None:
                            param.grad.detach_()
                            param.grad.zero_()
            print("loss")
            self.iter_step += 1
            t_optimizer_end = time.time()
            #print("optimizer close", t_optimizer_end - t_optimizer_right_before)
            self.writer.add_scalar('Loss/loss', loss, self.iter_step)

            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            if sample_mode == "patch":
                self.writer.add_scalar('Loss/dssim_loss', color_dssim, self.iter_step)
            if self.is_distill:
                self.writer.add_scalar('Loss/distill_loss', distill_loss, self.iter_step)
            if self.mask_weight > 0:
                self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            # self.writer.add_scalar('Loss/back_facing_loss', back_facing_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
            if self.conf.train.num_pcd_pts>0:
                self.writer.add_scalar('Loss/pcd_loss', pcd_loss, self.iter_step)
            # if self.color_network.bsdf_type=="ambient_sep":
            #     self.writer.add_scalar('Loss/light_l1_loss', light_l1_loss, self.iter_step)
                
            # if self.conf["model.sdf_type"] != "ngp":
            #     self.update_learning_rate()
            if self.ncc_weight > 0 and ncc_loss != 0:
                self.writer.add_scalar('Loss/ncc_loss', ncc_loss, self.iter_step)
            
            if self.rgb_loss_type == 'adpt_l2_log_lin':
                self.writer.add_scalar('Statistics/adpt_loss_mean', self.loss_func.mean_loss[0], self.iter_step)
            if self.pts_radiance_weight > 0.0:
                self.writer.add_scalar('Loss/pts_loss', pts_loss, self.iter_step)
            # print("update learning rate")
            # self.update_learning_rate()
            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()
            t_misc_end = time.time()
            if self.mitsuba_renderer is not None and ((self.iter_step + 1) % self.mitsuba_renderer.config.train.restart_freq == 0):
                # self.extract_mesh_and_reinit_mitsuba()
                print("we have to exit to to load new mesh in mitsuba otherwise it will crash")
                self.save_checkpoint(is_latest=True)
                exit(77)
            print("last line")
            # print("main loop", "render", t_render_end-t_init, "loss", t_loss_end - t_render_end, "mitsuba", t_mitsuba_end - t_loss_end, "backward", t_backward_end - t_mitsuba_end, "nan", t_find_nan_end - t_backward_end,  "misc", t_misc_end-t_find_nan_end)
            # print("main loop: t_init", t_init_end - t_init, "render", t_render_end - t_init_end, "loss", t_loss_end - t_render_end, "backward", t_backward_end - t_loss_end, "optimizer", t_optimizer_end - t_backward_end, "misc", t_misc_end - t_optimizer_end, "loop time", t_misc_end - t_init)
            # prof.export_chrome_trace("trace.json")
            # exit()
    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        lr_begin_iter = self.conf.train.lr_begin_iter
        if self.iter_step <= self.warm_up_end:
            learning_factor = (self.iter_step +1 - lr_begin_iter) / (self.warm_up_end -lr_begin_iter)
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        assert learning_factor>=0, (self.iter_step, lr_begin_iter)
        print("current learning rate", self.learning_rate * learning_factor, "progress", (self.iter_step +1 - lr_begin_iter), "total iters", (self.warm_up_end -lr_begin_iter))
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
        for idx, (rays_o_batch, rays_d_batch, color_batch, mask_batch, pixels_x_batch, pixels_y_batch) in enumerate(zip(rays_o, rays_d, color, mask, pixels_x, pixels_y)):
            print("batch idx", idx)
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

            
            light_to_world, mitsuba_light_lumen = self.dataset.gen_light_params_pose(idx)
            light_o = light_to_world[:3, 3]
            light_o, light_lum = self.dataset.gen_light_params(idx)
            img = None
            albedo = None
            roughness = None
            for i in range(mitsuba_repeats):
                print('repeat', i)
                with torch.no_grad():
                    with dr.suspend_grad():            
                        out = self.mitsuba_trainer.get_required_output(render_out, rays_o_batch, rays_d_batch, near, far, light_to_world, light_o, light_lum, i, geometry_type_name="vol_bsdf_direct")
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

            render_outs.append(img)
            albedo_outs.append(albedo)
            roughness_outs.append(roughness)


        img = torch.cat(render_outs, dim=0)
        img = img.reshape([H, W, 3])

        albedo = torch.cat(albedo_outs, dim=0)
        albedo = albedo.reshape([H, W, 3])
        roughness = torch.cat(roughness_outs, dim=0)
        roughness = roughness.reshape([H, W, 1])  

        img_dir = os.path.join(self.base_exp_dir,
                            'mitsuba_img')
        os.makedirs(img_dir, exist_ok=True)
        img_path_exr = os.path.join(img_dir, '{:0>8d}_{}.exr'.format(self.iter_step, idx))

        cv.imwrite(img_path_exr, img.cpu().numpy())

        albedo_dir = os.path.join(self.base_exp_dir,
                            'mitsuba_albedo')
        os.makedirs(albedo_dir, exist_ok=True)
        albedo_path_exr = os.path.join(albedo_dir, '{:0>8d}_{}.exr'.format(self.iter_step, idx))
        cv.imwrite(albedo_path_exr, albedo.cpu().numpy())

        roughness_dir = os.path.join(self.base_exp_dir,
                        'mituba_roughness')
        os.makedirs(roughness_dir, exist_ok=True)
        roughness_path_exr = os.path.join(roughness_dir, '{:0>8d}_{}.exr'.format(self.iter_step, idx))

        cv.imwrite(roughness_path_exr, roughness.cpu().numpy())

        return img, albedo, roughness

    def validate_image(self, idx=-1, resolution_level=-1, log_to_tb=False, printf=print, basic_only=False, mitsuba_repeats=1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        # print("=============================================== forcing validate image to chosen one, 172")
        # idx = 172
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

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              light_o,
                                              light_lum,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)
            # sdf_intersect, pts_intersect = locate_intersection(render_out['sdf'], render_out['z_vals'], render_out['mid_z_vals'], rays_o_batch, rays_d_batch)

            
            if self.mitsuba_renderer is not None and self.mitsuba_renderer.config.render.use_field_occlusion_hint:
                with torch.no_grad():
                    from models.renderer import locate_intersection
                    sdf_intersect, pts_intersect = locate_intersection(render_out['sdf'], render_out['z_vals'], render_out['mid_z_vals'], rays_o_batch, rays_d_batch)
                    pts_intersect = pts_intersect.squeeze(dim=1) # B x 3
                    light_dir = light_o - pts_intersect # B x 3
                    light_dir_length = torch.norm(light_dir, dim=-1, keepdim=True) # B x 1
                    light_dir = light_dir / light_dir_length # B x 3
                    
                    shadow_near, shadow_far = self.dataset.near_far_from_sphere(pts_intersect, light_dir)
                    shadow_render_out = self.renderer.render_alpha(pts_intersect, light_dir, light_o, light_lum, shadow_near, light_dir_length, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=1.0)
                    shadow_render_alpha = shadow_render_out["weight_sum"]                
                    em_occlusion = shadow_render_alpha > 0.3
                    render_out['color_fine'][em_occlusion.repeat(1, 3)] = 0.0
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
        # import pdb
        def feasible(key): return (key in render_outs[0]) and (render_outs[0][key] is not None)
        def calc_every_batch(func):
            outs = []
            for r in render_outs:
                outs.append(func(r))
            return outs
        
        # start fine
        enabled_fine = True
        if enabled_fine:
            render_at_intersect = False
            if render_at_intersect:
                print("this path")
                out_rgb_fine = []
                for rays_o_batch, rays_d_batch, color_batch in zip(rays_o, rays_d, color):
                    pts_intersect = render_out["pts_intersect"]
                    grads = self.sdf_network.gradient(pts_intersect).squeeze()
                    sdf_nn_output = self.sdf_network(pts_intersect)
                    rgb = self.color_network(pts, grads, rays_d_batch, light_o, light_lum, torch.zeros((rays_d_batch.shape[0], 0), device=rays_d_batch.device), sdf_nn_output[:, 1:1+256])
                    out_rgb_fine.append(rgb.detach().cpu().numpy())
                pass
            else:
                if feasible('color_fine'):
                    def f(render_out):
                        return render_out['color_fine'].detach().cpu().numpy()
                    out_rgb_fine = calc_every_batch(f)
                    pass
            
            img_fine = np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3])
            img = img_fine
            img_fine = ( image_utils.lin2srgb(img_fine) * 256).clip(0, 255)
            # img_fine = ( img_fine * 256).clip(0, 255)
            os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir,
                                    'validations_fine',
                                    '{:0>8d}_{}.png'.format(self.iter_step,  idx)),
                        np.concatenate([img_fine,
                                        (image_utils.lin2srgb(self.dataset.image_at(idx, resolution_level=resolution_level, to256=False))*256).clip(0, 255)])[...,::-1])
            # cv.imwrite(os.path.join(self.base_exp_dir,
            #                         'validations_fine',
            #                         '{:0>8d}_{}.png'.format(self.iter_step,  idx)),
            #             np.concatenate([img_fine,
            #                             (self.dataset.image_at(idx, resolution_level=resolution_level, to256=False)*256).clip(0, 255)])[...,::-1])
            # img = img_fine
            # end fine
        
        if feasible('gradients') and feasible('weights'):
            n_samples = self.renderer.n_samples + self.renderer.n_importance
            def f(render_out):
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                return normals
            out_normal_fine = calc_every_batch(f)
            os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3]) * 128 + 128).clip(0, 255)
            cv.imwrite(os.path.join(self.base_exp_dir,
                                    'normals',
                                    '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                        normal_img[...,::-1])
            
        if feasible('brdf_params'):
            def f(render_out):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                brdf_params = F.sigmoid(render_out['brdf_params'])
                brdf_params = brdf_params * render_out['weights'][:, :n_samples, None]
                brdf_params = brdf_params.sum(dim=1)               
                return brdf_params.detach().cpu().numpy()

            out_brdf_params = calc_every_batch(f)
            brdf_params = np.concatenate(out_brdf_params, axis=0).reshape([H, W, -1])
            if brdf_params.shape[-1] != 0:
                assert brdf_params.shape[-1] == 9
                
                albedo = brdf_params[:, :, 6:9]
                roughness = brdf_params[:, :, 4]
                albedo_dir = os.path.join(self.base_exp_dir,
                                'albedo')
                os.makedirs(albedo_dir, exist_ok=True)
                roughness_dir = os.path.join(self.base_exp_dir,
                                'roughness')
                os.makedirs(roughness_dir, exist_ok=True)

                gamma = self.color_network.flash_light_gamma().detach().cpu().numpy().item()
                
                albedo_path = os.path.join(albedo_dir, '{:0>8d}_{}.png'.format(self.iter_step, idx))
                roughness_path = os.path.join(roughness_dir, '{:0>8d}_{}.png'.format(self.iter_step, idx))
                imageio.imwrite(albedo_path, (np.clip(albedo*gamma, 0.0, 1.0)*255).astype(np.uint8))
                imageio.imwrite(roughness_path, (roughness*255).astype(np.uint8))
                
                albedo_dir_npy = os.path.join(self.base_exp_dir,
                                            'albedo_npy')
                os.makedirs(albedo_dir_npy, exist_ok=True)
                roughness_dir_npy = os.path.join(self.base_exp_dir,
                                'roughness_npy')
                os.makedirs(roughness_dir_npy, exist_ok=True)

                albedo_path_npy = os.path.join(albedo_dir_npy, '{:0>8d}_{}.npy'.format(self.iter_step, idx))
                roughness_path_npy = os.path.join(roughness_dir_npy, '{:0>8d}_{}.npy'.format(self.iter_step, idx))
                np.save(albedo_path_npy, albedo*gamma)
                np.save(roughness_path_npy, roughness)

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


        
                

                

            # image_perm = self.get_image_perm()
        return img
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
    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0, simplify=False, bake_texture_maps=False, bake_vert_maps=False, texture_resolution=2048, log_to_tb=False, save_to=None):
        # print("bake_texture_maps", bake_texture_maps)
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
        
        out_file = None
        if bake_texture_maps:
            xatlas.export('{}.obj'.format(save_to), vertices, triangles, uv)
            print(f"Saved mesh to '{save_to}.obj' and textures under '{save_dir}'.")
            out_file = '{}.obj'.format(save_to)
        else:
            mesh = trimesh.Trimesh(vertices, triangles)
            if bake_vert_maps:
                brdf_params = batch_feed(self.renderer.extract_shading_params, vertices.reshape(-1,3))
                albedo = brdf_params["base_color"]
                # albedo = utils.lin2srgb(albedo)
                roughness = brdf_params["roughness"]
                # roughness = utils.lin2srgb(roughness)
                vertex_attributes = {
                    "albedo_r": albedo[:, 0],
                    "albedo_g": albedo[:, 1],
                    "albedo_b": albedo[:, 2],
                    "roughness_0": roughness[:, 0],
                }
                mesh.vertex_attributes = vertex_attributes
            mesh.export('{}_lowres.ply'.format(save_to))
            # out_file = '{}_lowres.ply'.format(save_to)
        if log_to_tb and self.writer:
            self.writer.add_mesh('shape', torch.from_numpy(vertices).unsqueeze(0), faces=torch.from_numpy(triangles).unsqueeze(0), global_step=self.iter_step)
        
        logging.info('End')

    @torch.no_grad()
    def validate_mesh_hires(self, world_space=False, resolution=64, threshold=0.0, simplify=False, bake_texture_maps=False, texture_resolution=2048, log_to_tb=False, save_to=None):
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
        
        save_to_full = '{}.ply'.format(save_to)
        mesh = trimesh.Trimesh(vertices, triangles).export(save_to_full, vertex_normal=True)
        # trimesh.exchange.ply.export_ply(mesh, vertex_normal=True)

        # print("prune invisible faces")
        # self.check_mesh_face_visibility(save_to_full, save_to_full)

        logging.info('End')
        # self.check_mesh_face_visibility(save_to_full, save_to_full)
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
    print('Starting GLOW runner')

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
    elif args.mode == 'validate_mesh_hires':
        with torch.no_grad():
            runner.validate_mesh_hires(world_space=args.is_world, resolution=512, threshold=args.mcube_threshold, simplify=False, bake_texture_maps=not args.disable_bake_texture, texture_resolution=4096)

    elif args.mode.startswith('validate_mesh'):
        with torch.no_grad():
            if len(args.mode.split('_')) == 2:
                resolution = 512
            else:
                assert len(args.mode.split('_')) == 3
                resolution = int(args.mode.split('_')[2])
                pass
            
            runner.validate_mesh(world_space=args.is_world, resolution=resolution, threshold=args.mcube_threshold, simplify=False, bake_texture_maps=not args.disable_bake_texture, texture_resolution=4096, bake_vert_maps=True)

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
    elif args.mode == "basic_validate":
        runner.validate_mitsuba_all( mitsuba_repeats=32)
    else:
        raise RuntimeError(f"Unknown mode {args.mode}")
            

if __name__ == '__main__':
    main()
