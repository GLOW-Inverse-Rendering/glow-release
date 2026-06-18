from typing import Optional
import sys
from pathlib import Path as _PathForNerad
NERAD_ROOT = _PathForNerad(__file__).resolve().parents[1] / 'integrations' / 'inverse-neural-radiosity'
sys.path.insert(0, str(NERAD_ROOT)) # folded NeRAD integration root in this release tree

import common
import logging
from pathlib import Path
import torch.nn
import enum
import subprocess
import pickle
import base64
import os.path
import drjit as dr
import mitsuba as mi
logger = logging.getLogger(__name__)

def mitsuba_pre_1_0():
    if mi.__version__ == "3.5.2": 
        return True
    elif mi.__version__ == "3.6.2":
        return False
    else:
        raise NotImplementedError(mi.__version__)

logger.info("Configuring Global Drjit Flags")
dr.set_flag(dr.JitFlag.LoopRecord, False)
dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.VCallOptimize, False)

def load_config(scene_path, mesh_path, out_dir, config):
    amsgrad=config.train.amsgrad
    bsdf=config.render.bsdf
    use_flashlight=config.render.use_flashlight
    freeze_flashlight_intensity=config.render.freeze_flashlight_intensity
    load_integrator_only=config.train.load_integrator_only
    flashlight_type =config.render.flashlight_type
    print("load_integrator_only: ", load_integrator_only)
    # print("============== warning force bsdf to be same")
    # bsdf = "gt"
    flashlight_tmpl = """
        "dataset.collocated_flashlight=true",
        "dataset.flashlight_energy=1.0  ",
        "dataset.flashlight_type={flashlight_type} ",
        "learn_flashlight_intensity=true ",
        "freeze_flashlight_intensity={freeze_flashlight_intensity} ",
    """.format(flashlight_type=flashlight_type, freeze_flashlight_intensity="true" if freeze_flashlight_intensity else "false")
    # print("freeze flashlight")
    # 
    flashlight_str = flashlight_tmpl if use_flashlight else ""
    # rendering = "nerad_transfer_dir_em_cond_no_albedo_no_scene" if use_flashlight else "nerad"
    # print("including scene properties for debugging")
    # "nerad_transfer_dir_em_cond_no_albedo_no_scene"
    if use_flashlight:
        if config.render.ambient_light:
            rendering = "nerad_transfer_dir_multi_field_ambient"
        else:
            rendering = "nerad_transfer_dir_multi_field"
    else:
        rendering = "nerad"
    load_integrator_only_str = "load_integrator_only=true" if load_integrator_only else "load_integrator_only=false"
    mesh_path = mesh_path if mesh_path is not None else "null"
    cmd = f'''
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf
import base64
import pickle
import pathlib
NERAD_ROOT = pathlib.Path({str(NERAD_ROOT)!r})
with initialize_config_dir(version_base=None, config_dir=str(NERAD_ROOT / 'config'), job_name="wildlight"):
    overrides=[
        "dataset.scene={scene_path}", 
        "bsdf={bsdf} ",
        {flashlight_str}
        "rendering={rendering} ",
        "out_root={out_dir}",
        "learning_rate=5e-4",
        "amsgrad={amsgrad}",
        "{load_integrator_only_str}"
        ]

    if len("{mesh_path}") != 0:
        overrides += ["dataset.custom_mesh={mesh_path}"]
    cfg = compose(config_name="train", 
        overrides=overrides
    )
    OmegaConf.resolve(cfg)
    cfg.out_root = pathlib.Path(cfg.out_root)
    print(base64.encodebytes(pickle.dumps(cfg)).decode("ascii"))
'''
    #si_cull.prim_index.numpy().min()
    try:
        cfg_pickle = subprocess.run(['python3', '-c', cmd], capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise
    cfg = pickle.loads(base64.decodebytes(cfg_pickle))
    print(cfg)
    return cfg


def find_latest_ckpt(folder: Path) -> Optional[Path]:
    if os.path.isfile(folder / "latest.ckpt"):
        return folder / "latest.ckpt"

    files = list(folder.glob("*.ckpt"))
    if len(files) == 0:
        return None

    files = sorted(
        [[int(file.stem), file] for file in files],
        key=lambda a: a[0]
    )
    return files[-1][1]
def load_scene(cfg):
    device = "cuda:0"
    scene, learned_modules = common.load_scene_from_cfg(cfg.dataset, cfg.bsdf, device)
    
    recons_loss_function = common.create_loss_function(cfg.recons_loss, cfg.n_steps)
    loss_functions = [recons_loss_function]
    logger.info(f"recons_loss_function: {recons_loss_function}")
    is_nerad = True

    if is_nerad:
        LHS_recons_loss_function = common.create_loss_function(cfg.LHS_recons_loss, cfg.n_steps)
        loss_functions.append(LHS_recons_loss_function)
        logger.info(f"LHS_recons_loss_function: {LHS_recons_loss_function}")

    integrator_injection = {}
    if is_nerad:
        residual_loss_function = common.create_loss_function(cfg.residual_loss, cfg.n_steps)
        loss_functions.append(residual_loss_function)
        integrator_injection["residual_function"] = residual_loss_function


    integrator_function_injection = {"device": device}
    integrator = common.create_integrator(
        cfg.rendering,
        scene,
        post_init_injection=integrator_injection,
        kwargs_injection=integrator_function_injection,
    )
    # print("bsdf ckpt", find_latest_ckpt(cfg.out_root / "checkpoints") if cfg.resume else None)
    # print("radiance cache", find_latest_ckpt(Path(cfg.radiance_cache) / "checkpoints") if cfg.radiance_cache is not None else None)
    # exit()
    # print(cfg.out_root)
    # exit()
    ckpt_path = find_latest_ckpt(cfg.out_root / "checkpoints") if cfg.resume else None
    learned_info = common.prepare_learned_objects(
        scene,
        integrator,
        learned_modules,
        cfg.envmap,
        cfg,
        ckpt_path,
        find_latest_ckpt(Path(cfg.radiance_cache) / "checkpoints") if cfg.radiance_cache is not None else None,
        device,
    )
    return scene, integrator, learned_info, ckpt_path

class CacheWrapper(torch.nn.Module):
    def __init__(self, network):
        self.cache = None
        self.network = network
    
    def forward(self, x):
        if self.cache is None or (x != self.cache).any():
            self.cache = self.network(x)
        return self.cache

class ColorNetworkWrapperMode(enum.Enum):
    ALBEDO = "albedo"
    ROUGHNESS = "roughness"

class ColorNetworkWrapper(torch.nn.Module):
    def __init__(self, color_network, mode: ColorNetworkWrapperMode):
        self.color_network = CacheWrapper(color_network)
        self.mode = mode
        
    def forward(self, *args, **kwargs):
        result = self.color_network(*args, **kwargs)
        albedo = result[..., :3]
        roughness = result[..., 3:6] # 3 channels for compatibility with nerad
        if self.mode is ColorNetworkWrapperMode.ALBEDO:
            return albedo
        elif self.mode is ColorNetworkWrapperMode.ROUGHNESS:
            return roughness
        else:
            raise ValueError(f"Unknown mode {self.mode}")
def find_bsdf(scene):
    bsdf = None
    for s in scene.shapes():
        bsdf = s.bsdf()
        if 'my-bsdf' in bsdf.id():
            break
    return bsdf
def wrap_bsdf(cfg, sdf_net, color_net):
    if cfg.is_sdf:
        shape_cfg_name = 'dummy_sdf,{s},{x},{y},{z}'.format(s=cfg.scale, x=cfg.trans_x, y=cfg.trans_y, z=cfg.trans_z)
    elif cfg.custom_mesh is not None:
        shape_cfg_name = 'custom_mesh,{}'.format(cfg.custom_mesh)
    else:
        shape_cfg_name = 'same'
    bsdf_cfg = cfg.bsdf
    cached_sdf = CacheWrapper(sdf_net)
    bsdf_kwargs = {
        "post_init": {},
        "texture": {
            "base_color": {
                "post_init": {
                    "function": "reflectance_net_coupled_external_net",
                    "kwargs": {
                        "network": ColorNetworkWrapper(color_net, mode=ColorNetworkWrapperMode.ALBEDO),
                        "sdf": cached_sdf,
                        "feature_size": 265,
                        "scene_min": [-1.0, -1.0, -1.0],
                        "scene_max": [1.0, 1.0, 1.0],
                    }
                }
            },
            "roughness": {
                "post_init": {
                    "function": "reflectance_net_coupled_external_net",
                    "kwargs": {
                        "network": ColorNetworkWrapper(color_net, mode=ColorNetworkWrapperMode.ROUGHNESS),
                        "sdf": cached_sdf,
                        "feature_size": 265,
                        "scene_min": [-1.0, -1.0, -1.0],
                        "scene_max": [1.0, 1.0, 1.0],
                    }
                }
            }
        }
    }
    brdf_dict = {"name": bsdf_cfg.name, "bsdf_kwargs": bsdf_kwargs}
    scene_2 = common.load_scene_with_edits(cfg.scene, shape_cfg_name, brdf_dict, emitter_mode='same')
    bsdf = find_bsdf(scene_2)
    assert bsdf is not None, "Could not find the bsdf"
    
    return bsdf
