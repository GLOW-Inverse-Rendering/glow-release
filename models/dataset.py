import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import imageio
import open3d as o3d
import pickle
import pathlib
def compose_transf(P1, P2):

    R1 = P1[:3,:3]
    T1 = P1[:3,3:4]
    R2 = P2[:3,:3]
    T2 = P2[:3,3:4]

    R = R1@R2
    T = T1 + R1@T2
    return np.concatenate([R,T], axis=-1)

def make4x4(P):
    assert P.shape[-1] == 4
    assert len(P.shape) == 2
    assert P.shape[0] == 3 or P.shape[0] == 4
    ret = np.eye(4)
    ret[:P.shape[0]] = P
    return ret

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
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

def img_id(path):
    return int(os.path.splitext(os.path.basename(path))[0])
def srgb_to_linsrgb (srgb):
    """Convert sRGB values to physically linear ones. The transformation is
       uniform in RGB, so *srgb* can be of any shape.

       *srgb* values should range between 0 and 1, inclusively.

    """
    gamma = ((srgb + 0.055) / 1.055)**2.4
    scale = srgb / 12.92
    return np.where (srgb > 0.04045, gamma, scale)

srgb2lin = srgb_to_linsrgb

def lin2srgb(lin):
    s1 = 1.055 * (np.power(lin, (1.0 / 2.4))) - 0.055
    s2 = 12.92 * lin
    s = np.where(lin > 0.0031308, s1, s2)
    return np.minimum(s, 1.0)
class EpochSampler:
    def __init__(self, reg_arrays):
        assert all([arr.shape[0] == reg_arrays[0].shape[0] for arr in reg_arrays])
        self.reg_arrays = reg_arrays
        self.num_total = reg_arrays[0].shape[0]
        self._rand_perm()
    def _rand_perm(self):
        self.rand_indexes = torch.randperm(self.num_total)
        self.index_in_rand = 0
    def sample(self, num_samples):
        start = self.index_in_rand
        self.index_in_rand += num_samples
        end = self.index_in_rand
        rand_indexes_slice = self.rand_indexes[start:end]

        sampled_arrays = [arr[rand_indexes_slice] for arr in self.reg_arrays]

        # permute all pixels only after traversing all
        if self.index_in_rand >= self.num_total:
            self._rand_perm()

        return sampled_arrays
    
class Dataset:
    def __init__(self, conf, load_images=True):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.ignore_0 = False #conf.ignore_zero_RGB

        self.data_dir = conf.data_dir
        self.render_cameras_name = conf.render_cameras_name
        self.object_cameras_name = conf.object_cameras_name

        # self.camera_outside_sphere = conf.camera_outside_sphere
        # self.scale_mat_scale = conf.scale_mat_scale
        # self.scale_mat_scale = 1.1
        print(os.path.join(self.data_dir, self.render_cameras_name))
        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.light_offset_mat = None
        light_offset_path = os.path.join(self.data_dir, "light_offset.npy")
        if os.path.exists(light_offset_path):
            print("light offset exists")
            light_offset_mat = np.load(light_offset_path)
            self.light_offset_mat = torch.from_numpy(light_offset_mat).float()
        self.apply_gamma = conf.apply_gamma
        self.use_pcd = conf.use_pcd
        print('apply gamma', self.apply_gamma)
        self.pcd_sampler = None
        print("FORCING load images to be true")
        load_images = True
        if load_images:
            if os.path.exists(os.path.join(self.data_dir, 'images.npy')):
                self.images_np = np.load(os.path.join(self.data_dir, 'images.npy'))
                max_intensity = float(camera_dict.get("max_intensity"))
                self.images_np = self.images_np / max_intensity * 5
                self.saturation_intensity = 5
                self.images_id = list(range(len(self.images_np)))
                
                
            elif len(glob(os.path.join(self.data_dir, 'image/*.exr'))) > 0: 
                # exr format
                # print("============WARNING: only loading 10 images===========")
                self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.exr')))#[:10]
                if self.conf.limit_size is not None:
                    self.images_lis = self.images_lis[:self.conf.limit_size]
                    print(f"Loading limited image set: {self.images_lis}")
                # self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.exr')))[:1]
                # 
                # print(self.images_lis)
                # exr_max = np.inf
                exr_max = 1.0
                if exr_max != np.inf:
                    print("Clamping EXR images at {}".format(exr_max))
                    pass
                
                self.images_np = np.stack([np.minimum(cv.imread(im_name, -1)[...,2::-1], exr_max) for im_name in self.images_lis])
                # self.saturation_intensity = float(camera_dict.get("max_intensity", np.inf))
                self.saturation_intensity = float(exr_max)
                self.images_id = [img_id(p) for p in self.images_lis]
            else:
                # png format
                self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))#[:1]
                if self.conf.limit_size is not None:
                    self.images_lis = self.images_lis[:self.conf.limit_size]
                    print(f"Loading limited image set: {self.images_lis}")
                # print("============WARNING: only loading 1 image===========")
                test_img = cv.imread(self.images_lis[0])[...,::-1]
                self.images_np = np.empty([len(self.images_lis)] + list(test_img.shape), dtype=np.float32)
                print("Assuming PNG inputs are already linear")
                for i, im_name in enumerate(self.images_lis):
                    # self.images_np[i] = srgb2lin(cv.imread(im_name)[...,::-1]  / 255.0)
                    img = cv.imread(im_name, flags=cv.IMREAD_UNCHANGED)
                    if img.dtype  == np.uint8:
                        print("using uint8")
                        self.images_np[i] = img[...,::-1]  / 255.0
                    elif img.dtype == np.uint16:
                        print("using uint16", "i:", i)
                        # print("============WARNING========== scaling by 4")
                        self.images_np[i] = np.minimum(img[...,::-1]  / 65535, 1.0)
                    else:
                        raise NotImplementedError(img.dtype)
                # self.images_np = np.stack([cv.imread(im_name)[...,::-1] for im_name in self.images_lis]) / 256.0
                # self.images_np = srgb2lin(self.images_np)
                self.saturation_intensity = float(camera_dict.get("max_intensity", 255 / 255.0 - 0.001))
                self.images_id = [img_id(p) for p in self.images_lis]
                pass
            if self.apply_gamma:
                self.images_np = np.clip(
                    np.power(self.images_np, 1.0 / 2.2),
                    0.0,
                    1.0,
                )
                self.saturation_intensity = np.inf # since this is not physical method
            print("end images")
            print("saturation_intensity", self.saturation_intensity)
            self.n_images = len(self.images_np)
            print("self.n_images", self.n_images)
            
            if conf.get('ignore_mask', False):
                print("ignore mask")
                print("size of images_np", self.images_np.shape)
                self.masks_np = np.ones_like(self.images_np, dtype=np.float32)
                self.no_mask = True
                print("ignore done")
            else:
                print("Loading masks from ", os.path.join(self.data_dir, 'mask/*.png'))
                mask_dic = {img_id(p):p for p in glob(os.path.join(self.data_dir, 'mask/*.png'))}
                print("end loading masks")
                self.masks_lis = [mask_dic[i] for i in self.images_id]
                self.masks_np = (np.stack([cv.imread(im_name) for im_name in self.masks_lis]) == 255).astype(np.float32) 
                self.no_mask = False
            if os.path.exists(os.path.join(self.data_dir, 'shadow_mask')):
                print('loading shadow mask')
                mask_dic = {img_id(p):p for p in glob(os.path.join(self.data_dir, 'shadow_mask/*.png'))}
                print("end loading shadow mask")
                self.shadow_masks_lis = [mask_dic[i] for i in self.images_id]
                self.shadow_masks_np = (np.stack([cv.imread(im_name) for im_name in self.shadow_masks_lis]) == 255) 
            else:
                self.shadow_masks_np = None
            print("mask done")
            
            sam_mask_path = os.path.join(self.data_dir, 'sam_masks')
            if os.path.exists(sam_mask_path):
                # try:
                print(f"Loading sam mask from ", sam_mask_path)
                sam_mask_dic = {img_id(p):p for p in glob(os.path.join(sam_mask_path, '*.png'))}
                print("end loading sam masks")
                self.sam_mask_lis = [sam_mask_dic[i] for i in self.images_id]
                self.sam_mask_np = np.stack([cv.imread(im_name, -1) for im_name in self.sam_mask_lis])
                # except:
                #     print("sam mask loading failed")
                #     self.sam_mask_np = None
            else:
                self.sam_mask_np = None
            
            self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
            self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
            self.H, self.W = self.images.shape[1], self.images.shape[2]
            self.image_pixels = self.H * self.W

            self.images_np = None
            self.masks_np = None
            print(f'Load images: End ({self.n_images} images at {self.H}X{self.W}) ')
        else:
            self.images_id = sorted([int(k.split('_')[-1]) for k in camera_dict if k.startswith('world_mat_')])
            self.n_images = len(self.images_id)
            # example_input = cv.imread(glob(os.path.join(self.data_dir, 'mask/*.png'))[0])
            # self.H, self.W = example_input.shape[0], example_input.shape[1]
        
        print("loading cameras")
        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.images_id]
        print("end loading cameras")
        print("self.images_idx", self.images_id)
        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        
        print("initializing scale mat")
        self.scale_mats_np = [make4x4(camera_dict['scale_mat_%d' % idx]).astype(np.float32) for idx in self.images_id]
        print("end initializing scale mat")

        self.light_energies = [camera_dict.get('light_energy_%d' % idx, np.zeros(3)).astype(np.float32) for idx in self.images_id]
        print(f"{(np.stack(self.light_energies).max(-1) > 0).sum()} out of {len(self.light_energies)} images have flashlight on.")
        self.intrinsics_all = []
        self.pose_all = []
        self.pose_all_no_scale = []

        for idx, (scale_mat, world_mat) in enumerate(zip(self.scale_mats_np, self.world_mats_np)):
            # P = compose_transf(world_mat, scale_mat)
            P = world_mat @ scale_mat
            # print("P", P)
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            # print("intrinsics", intrinsics)
            # print("pose", pose)
            # print("idx:", idx)
            # print("idx", self.images_id[idx], load_K_Rt_from_P(None, world_mat[:3]))
            # print("world_mat", world_mat)
            _, pose_no_scale = load_K_Rt_from_P(None, world_mat[:3, :4])
            if idx < 150:
                print("idx", idx)
                print("imgid", self.images_id[idx])
                print("intrinsics")
                print(intrinsics)
                print("pose")
                print(pose)
                print("pose_no_scale")
                print(pose_no_scale)
            # input("waiting")
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
            self.pose_all_no_scale.append(torch.from_numpy(pose_no_scale).float())
        # exit()
        self.intrinsics_all = torch.stack(self.intrinsics_all)#.to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all)#.to(self.device)  # [n_images, 4, 4]
        # self.pose_all_no_scale = torch.stack(self.pose_all_no_scale).to(self.device)  # [n_images, 4, 4]
        # self.pose_all_no_scale_inv = torch.linalg.inv(self.pose_all_no_scale).to(self.device)
        self.pose_all_inv = torch.linalg.inv(self.pose_all)#.to(self.device)
        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = make4x4(np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0'])
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]
        if self.use_pcd:
            self.o3d_pcd = o3d.io.read_point_cloud(os.path.join(self.data_dir, "model/pcd.ply"))
            self.pcd = torch.from_numpy(np.asarray(self.o3d_pcd.points)).float()
            self.pcd_sampler = EpochSampler([self.pcd])
    
    def sample_pcd(self, num_pts):
        assert self.pcd_sampler is not None
        pcd_batch = self.pcd_sampler.sample(num_pts)[0]
        return pcd_batch
    
    def cap_pixel_val(self, img_idx):
        return self.saturation_intensity

    def dist_to_depth_map(self, img_idx, resolution_level=1):
        '''
        returns ratio of depth over distance, of size (H,W)
        '''
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        return rays_v[:,:,2].transpose(0, 1)

    def gen_rays_at(self, img_idx, resolution_level=1, return_color=False, return_mask=False, return_pixels=False):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        if self.conf.use_0_5_convention:
            tx = torch.linspace(0, self.W - 1, self.W // l)
            ty = torch.linspace(0, self.H - 1, self.H // l)

            tx_ray = torch.linspace(0.5, self.W - 0.5, self.W // l)
            ty_ray = torch.linspace(0.5, self.H - 0.5, self.H // l)
            print("tx, ty", tx_ray, ty_ray)
            print("intrinsic",self.intrinsics_all[img_idx] )
        else:
            tx = torch.linspace(0, self.W - 1, self.W // l)
            ty = torch.linspace(0, self.H - 1, self.H // l)

            tx_ray = tx
            ty_ray = ty
            
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x_ray, pixels_y_ray = torch.meshgrid(tx_ray, ty_ray)
        p = torch.stack([pixels_x_ray, pixels_y_ray, torch.ones_like(pixels_y_ray)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        # print(pixels_y)
        # print(pixels_x)
        # print("rays_o", rays_o)
        result = [rays_o.transpose(0, 1).to(self.device), rays_v.transpose(0, 1).to(self.device)]
        color = self.images[img_idx][(pixels_y.round().long(), pixels_x.round().long())]
        if  return_color:
            result.append(color.transpose(0, 1).to(self.device))
        if return_mask:
            # print(self.masks.shape)
            mask = self.masks[img_idx][(pixels_y.round().long(), pixels_x.round().long())]
            # print(mask.shape)
            result.append(mask.transpose(0,1).to(self.device))
        # print("here before", pixels_x.shape)
        if return_pixels:
            result.append(pixels_x.transpose(0,1).to(self.device).long().unsqueeze(dim=-1))
            result.append(pixels_y.transpose(0,1).to(self.device).long().unsqueeze(dim=-1))
        return tuple(result)
    def gen_light_params(self, img_idx):
        light_o = self.pose_all[img_idx, :3, 3].to(self.device)
        light_lum = torch.from_numpy(self.light_energies[img_idx]).to(self.device)
        return light_o, light_lum
    
    def gen_light_params_pose(self, img_idx):
        light_o = self.pose_all[img_idx].to(self.device)
        # print("WARNING=====fixme======== hard coded light offset for irb 4th floor")
        # light_o = light_o @ torch.tensor([
        #     [1,0,0,-0.02029],
        #     [0,1,0,-0.004683],
        #     [0,0,1,0],
        #     [0,0,0,1]
        # ], device=light_o.device, dtype=torch.float)
        
        light_lum = torch.from_numpy(self.light_energies[img_idx]).to(self.device)
        if self.light_offset_mat is not None:
            light_o = light_o @ self.light_offset_mat.to(light_o.device)
        return light_o, light_lum
    
    def gen_random_rays_patch(self, img_idx, patch_size, foreground_only=True, shift=(0,0), seed=None):
        if seed is not None:
            np.random.seed(seed)
        patch_size = min(patch_size, self.W, self.H)
        assert all((s-0.5)*(s+0.5)<=0 for s in shift)
        if foreground_only:
            for i in range(10):
                patch_origin_x = np.random.randint(0, self.W - patch_size)
                patch_origin_y = np.random.randint(0, self.H - patch_size)
                pixels_y, pixels_x = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size))
                pixels_x = (pixels_x + patch_origin_x).reshape(-1)
                pixels_y = (pixels_y + patch_origin_y).reshape(-1)

                color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
                mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
                if mask.sum() > 0:
                    break
            
            if mask.sum() == 0: # failed too many times, revert to mask sampling
                pixels_y, pixels_x = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
                pixels_x = pixels_x[self.masks[img_idx].mean(-1) > 0.5] 
                pixels_y = pixels_y[self.masks[img_idx].mean(-1) > 0.5]

                choice = torch.from_numpy(np.random.choice(len(pixels_x), 1, replace=False))
                patch_origin_x = min(int(pixels_x[choice[0]]), self.W-patch_size)
                patch_origin_y = min(int(pixels_y[choice[0]]), self.H-patch_size)

                pixels_y, pixels_x = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size))
                pixels_x = (pixels_x + patch_origin_x).reshape(-1)
                pixels_y = (pixels_y + patch_origin_y).reshape(-1)

                color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
                mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        else:
            patch_origin_x = np.random.randint(0, self.W - patch_size)
            patch_origin_y = np.random.randint(0, self.H - patch_size)

            pixels_y, pixels_x = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size))
            pixels_x = (pixels_x + patch_origin_x).reshape(-1)
            pixels_y = (pixels_y + patch_origin_y).reshape(-1)

            color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
            mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3

        p = torch.stack([pixels_x+shift[0], pixels_y+shift[1], torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3

        if self.ignore_0:
            mask = mask[:,:1] * (color.sum(-1) > 0).float()
        else:
            mask = mask[:,:1]

        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask], dim=-1).cuda()

        



    def gen_random_rays_at(self, img_idx, batch_size, foreground_only=True, shift=(0,0), seed=None):
        """
        Generate random rays at world space from one camera.
        """
        if seed is not None:
            np.random.seed(seed)
        assert all((s-0.5)*(s+0.5)<=0 for s in shift)
        l = 1
        # if self.conf.use_0_5_convention:
        #     tx = torch.linspace(0, self.W - 1, self.W // l)
        #     ty = torch.linspace(0, self.H - 1, self.H // l)

        #     tx_ray = torch.linspace(0.5, self.W - 0.5, self.W // l)
        #     ty_ray = torch.linspace(0.5, self.H - 0.5, self.H // l)
        #     # print("tx, ty", tx_ray, ty_ray)
        #     # print("intrinsic",self.intrinsics_all[img_idx] )
        # else:
        #     tx = torch.linspace(0, self.W - 1, self.W // l)
        #     ty = torch.linspace(0, self.H - 1, self.H // l)

            # tx_ray = tx
            # ty_ray = ty
            
        if foreground_only and (not self.no_mask):
            
            pixels_y, pixels_x = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
            pixels_x = pixels_x[self.masks[img_idx].mean(-1) > 0.5] 
            pixels_y = pixels_y[self.masks[img_idx].mean(-1) > 0.5]
            # self.conf.use_0_5_convention:
            #     pixels_x_ray = pixels_x + 0.5
            #     pixels_y_ray = pixels_y_ray[self.masks[img_idx].mean(-1) > 0.5] + 0.5
            
            batch_size = min(len(pixels_x), batch_size)

            choice = torch.from_numpy(np.random.choice(len(pixels_x), batch_size, replace=False))
            pixels_x = pixels_x[choice]
            pixels_y = pixels_y[choice]
            # pixels_x_ray = pixels_x_ray[choice]
            # pixels_y_ray = pixels_y_ray[choice]

        else:
            pixels_x = torch.from_numpy(np.random.randint(low=0, high=self.W, size=[batch_size]))
            pixels_y = torch.from_numpy(np.random.randint(low=0, high=self.H, size=[batch_size]))

        if self.conf.use_0_5_convention:
            pixels_x_ray = pixels_x + 0.5
            pixels_y_ray = pixels_y + 0.5
        else:
            pixels_x_ray = pixels_x
            pixels_y_ray = pixels_y

        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x_ray+shift[0], pixels_y_ray+shift[1], torch.ones_like(pixels_y_ray)], dim=-1).float()  # batch_size, 3
        # print(self.intrinsics_all_inv.device, pixels_x.device)
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3

        if self.ignore_0:
            mask = mask[:,:1] * (color.sum(-1) > 0).float()
        else:
            mask = mask[:,:1]
        return torch.cat([rays_o, rays_v, color, mask], dim=-1).to(self.device), torch.cat([pixels_x[:, None], pixels_y[:, None]], dim=-1).long().to(self.device)    # batch_size, 10
    
    def gen_sam_mask_at(self, img_idx, pixels_x, pixels_y):
        img_mask = self.sam_mask_np[img_idx]
        img_mask = torch.from_numpy(img_mask)
        pixel_mask = img_mask.long()[(pixels_y.cpu(), pixels_x.cpu())]
        width = self.sam_mask_np.shape[2]
        height = self.sam_mask_np.shape[1]
        return pixel_mask.to(self.device), width, height

    def gen_shadow_mask_at(self, img_idx, pixels_x, pixels_y):
        img_mask = self.shadow_masks_np[img_idx]
        # imageio.imwrite("test.png", (img_mask.astype(np.uint8)*255))
        # print(img_idx, pixels_x, pixels_y)
        # input("written test")
        img_mask = torch.from_numpy(img_mask)
        img_mask = ~img_mask
        # print("pixels_y", pixels_y)
        # print("pixels_x", pixels_x)
        
        pixel_mask = img_mask.float()[(pixels_y.cpu(), pixels_x.cpu())]
        return pixel_mask.to(self.device)


    def gen_light_params_between(self, idx_0, idx_1, ratio):
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3])#.cuda()
        trans = torch.from_numpy(pose[:3, 3])#.cuda()
        light_o = trans[:3]#.cuda()
        light_lum_0 = torch.from_numpy(self.light_energies[idx_0])#.cuda()
        light_lum_1 = torch.from_numpy(self.light_energies[idx_1])#.cuda()
        light_lum = light_lum_0 * (1.0 - ratio) + light_lum_1 * ratio
        return light_o, light_lum

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        print(self.images_lis[idx_0])
        print(self.images_lis[idx_1])
        # imageio.imwrite("test1.png", (lin2srgb(self.images[idx_0])*255).astype(np.uint8))
        # imageio.imwrite("test2.png", (lin2srgb(self.images[idx_1])*255).astype(np.uint8))
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3])#.cuda()
        trans = torch.from_numpy(pose[:3, 3])#.cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1).to(self.device), rays_v.transpose(0, 1).to(self.device)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near.clip(min=0), far

    def image_at(self, idx, resolution_level, to256=True):
        if to256:
            ratio = 256
        else:
            ratio = 1
        img = self.images[idx].numpy() * ratio
        img = (cv.resize(img, (self.W // resolution_level, self.H // resolution_level)))
        if to256:
            img = img.clip(0, 255)
        return img
    
    def mask_at(self, idx, resolution_level, to256=True):
        mask = self.masks[idx].numpy()
        mask = (cv.resize(mask, (self.W // resolution_level, self.H // resolution_level)))

        if to256:
            mask = (mask * 256).clip(0, 255)
        
        return mask


    def export_as_pickle(self, filename):
        data = {
            "images": self.images,
            "intrinsics_all": self.intrinsics_all,
            "pose_all": self.pose_all,
            "pose_all_no_scale": self.pose_all_no_scale,
            
        }
        if self.masks is not None:
            data["masks"] = self.masks
            
        with open(pathlib.Path(filename), 'wb') as f:
            pickle.dump(data, f)
