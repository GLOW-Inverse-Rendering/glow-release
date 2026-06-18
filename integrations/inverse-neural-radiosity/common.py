import logging
import random
import collections
from pathlib import Path
from typing import Any, Optional, Union
import itertools

import drjit as dr
import mitsuba as mi
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from mytorch.exp_lr import ExpLR, ExpLRScheduler

from nerad.bsdf import registered_bsdfs
from nerad.integrator import registered_integrators
from nerad.loss import loss_registry
from nerad.model.config import (BsdfConfig, ComputeConfig, DatasetConfig, EnvmapConfig,
                                ObjectConfig, RenderingConfig, TrainConfig)
from nerad.texture import registered_textures
from nerad.utils.dict_utils import inject_dict
from nerad.utils.io_utils import glob_sorted
from nerad.utils.json_utils import read_json
from nerad.utils.metric_utils import compute_metrics
from nerad.utils.mitsuba_utils import (builtin_bsdf_required_textures,
                                       load_scene_with_edits)
from nerad.utils.wildlight_utils import WildLightCamerasUtil
logger = logging.getLogger(__name__)


def configure_compute(cfg: ComputeConfig):
    logger.info(f"Set drjit flags to {cfg.dr_optimization_flags}")
    dr.set_flag(dr.JitFlag.LoopRecord, cfg.dr_optimization_flags)
    dr.set_flag(dr.JitFlag.VCallRecord, cfg.dr_optimization_flags)
    dr.set_flag(dr.JitFlag.VCallOptimize, cfg.dr_optimization_flags)

    logger.info(f"Set torch detech anomaly to {cfg.torch_detect_anomaly}")
    torch.autograd.set_detect_anomaly(cfg.torch_detect_anomaly)

    logger.info(f"Seed everything with {cfg.seed}")
    seed_everything(cfg.seed)

    log_mitsuba_registration()


def create_integrator(
    cfg: RenderingConfig,
    scene: mi.Scene,
    extra_config: dict[str, Any] = None,
    post_init_injection: dict[str, Any] = None,
    kwargs_injection: dict[str, Any] = None,
):
    mi_dict = {
        "type": cfg.integrator,
    }
    if extra_config is not None:
        mi_dict.update(extra_config)

    integrator_config = OmegaConf.to_container(cfg.config, resolve=True)
    if len(integrator_config) > 0:
        if cfg.integrator in registered_integrators:
            mi_dict["config"] = {
                "type": "dict",
                **integrator_config,
            }
        else:
            mi_dict.update(integrator_config)

    logger.info(f"Integrator dict: {mi_dict}")
    integrator = mi.load_dict(mi_dict)

    # if "enable_scale_mat" in cfg and cfg.enable_scale_mat:
    #     scene_min, scene_max = wildlight_infer_scene_size(cfg.sdf_cameras)
    #     kwargs_injection["scene_min"] = scene_min
    #     kwargs_injection["scene_max"] = scene_max
    # elif "post_init" in cfg and "sdf_kwargs" in cfg.post_init:
    print("WARNING: forcing a standard sized scene min & scene max")
    if kwargs_injection is not None:
        kwargs_injection["scene_min"] = mi.ScalarPoint3f(-1.0, -1.0, -1.0)
        kwargs_injection["scene_max"] = mi.ScalarPoint3f(1.0, 1.0, 1.0)

    _mitsuba_post_init(cfg.post_init, integrator, scene, post_init_injection, kwargs_injection)
    logger.info(f"Integrator: {integrator}")
    return integrator

def create_regularization_integrator(scene):
    mi_dict = {
        "type": "regularization",
    }
    # if extra_config is not None:
    #     mi_dict.update(extra_config)

    # integrator_config = OmegaConf.to_container(cfg.config, resolve=True)
    # if len(integrator_config) > 0:
    #     if cfg.integrator in registered_integrators:
    #         mi_dict["config"] = {
    #             "type": "dict",
    #             **integrator_config,
    #         }
    #     else:
    #         mi_dict.update(integrator_config)
    key = "my-bsdf.brdf_0.roughness.texture" # FIXME: change to a config
    params = mi.traverse(scene)
    texture = params[key]

    logger.info(f"Regularization Integrator dict: {mi_dict}")
    integrator = mi.load_dict(mi_dict)
    integrator.post_init(texture=texture)

    logger.info(f"Regularization Integrator: {integrator}")

    return integrator

def wildlight_infer_scene_size(sdf_cameras):
    cam_util = WildLightCamerasUtil(sdf_cameras)
    scene_min, scene_max =  cam_util.get_min_max()
    return scene_min, scene_max

def load_scene_from_cfg(cfg: DatasetConfig, bsdf_cfg: BsdfConfig, device: str):
    bsdf_name = bsdf_cfg.name
    texture_cfg = bsdf_cfg.get("texture")
    if texture_cfg is None:
        texture_cfg = {}
    learned_modules: dict[str, nn.Module] = {}

    if bsdf_name != "gt" or not cfg.collocated_flashlight or not cfg.is_sdf or cfg.custom_mesh is not None:
        # three cases: (1) built-in bsdf with fixed texture (2) built-in bsdf with custom texture, (3) custom bsdf
        is_custom_bsdf = False

        # for (1), check if custom textures presents
        required_textures = builtin_bsdf_required_textures.get(bsdf_name)
        if required_textures is not None:
            required_textures = sorted(required_textures)
            provided_textures = sorted(texture_cfg.keys())
            assert (required_textures == []) or (required_textures == provided_textures), \
                f"BSDF '{bsdf_cfg.name}' requires {required_textures} but got {provided_textures}"

        elif bsdf_name == "gt":
            pass
        else:
            assert bsdf_name in registered_bsdfs, f"BSDF '{bsdf_name}' is neither built-in nor custom"
            assert len(texture_cfg) == 0, "Custom BSDF does not support custom texture"
            is_custom_bsdf = True
        emitter_cfg_name = '{}/{}'.format(cfg.flashlight_type, cfg.flashlight_energy) if cfg.collocated_flashlight else 'same'
        if cfg.is_sdf:
            shape_cfg_name = 'dummy_sdf,{s},{x},{y},{z}'.format(s=cfg.scale, x=cfg.trans_x, y=cfg.trans_y, z=cfg.trans_z)
        elif cfg.custom_mesh == "dummy":
            shape_cfg_name = "dummy_sdf,0.0,-1000,-1000,-1000"
        elif cfg.custom_mesh is not None:
            shape_cfg_name = 'custom_mesh,{}'.format(cfg.custom_mesh)
        else:
            shape_cfg_name = 'same'
        # print("shape cfg name here", shape_cfg_name, cfg.custom_mesh, cfg.custom_mesh is not None, type(cfg.custom_mesh))
        brdf_dict = {"name": bsdf_cfg.name, "bsdf_kwargs": bsdf_cfg.bsdf_kwargs if "bsdf_kwargs" in bsdf_cfg else {}}
        scene = load_scene_with_edits(cfg.scene, shape_cfg_name, brdf_dict, emitter_cfg_name)
        params = mi.traverse(scene)
        for key in params.keys():
            if not key.startswith("my-bsdf.") or not key.endswith(".texture"):
                continue
            if is_custom_bsdf:
                name = "bsdf"
                post_init_cfg = bsdf_cfg.post_init
            else:
                # key looks like: my-bsdf.brdf_0.reflectance.texture
                # we use the name "reflectance"
                name = key.split(".")[-2]
                post_init_cfg = texture_cfg[name].post_init

            obj = params.get(key)
            assert isinstance(obj, nn.Module)
            kwargs_injection = {"device": device}

            if cfg.enable_scale_mat:
                scene_min, scene_max = wildlight_infer_scene_size(cfg.sdf_cameras)
                kwargs_injection["scene_min"] = scene_min
                kwargs_injection["scene_max"] = scene_max
                print("WARNING: using configured scene_min/scene_max override")
                kwargs_injection["scene_min"] = np.array([-2.81555003, -0.016453  , -2.82326005])
                kwargs_injection["scene_max"] = np.array([-0.81554997,  1.983547  , -0.82325995])

            else:
                kwargs_injection["scene_min"] = mi.ScalarPoint3f(-1.0, -1.0, -1.0)
                kwargs_injection["scene_max"] = mi.ScalarPoint3f(1.0, 1.0, 1.0)



            _mitsuba_post_init(post_init_cfg, obj, scene, kwargs_injection=kwargs_injection)
            learned_modules[name] = obj
    else:
        scene = mi.load_file(cfg.scene)

    return scene, learned_modules
def load_dataset(cfg: DatasetConfig, bsdf_cfg: BsdfConfig, device: str, skip_img=False):
    scene, learned_modules = load_scene_from_cfg(cfg, bsdf_cfg, device)

    transforms = read_json(cfg.cameras)
    n_views = min(cfg.n_views, len(transforms)) if cfg.n_views > 0 else len(transforms)

    logger.info(f"Load {n_views} views from {len(transforms)} views")
    if not skip_img:
        if cfg.is_exr:
            images = load_img_files(Path(cfg.cameras).parent / "exr", n_views, cfg.is_exr)
        else:
            images = load_img_files(Path(cfg.cameras).parent / "png", n_views, cfg.is_exr)
        assert len(images) == n_views, (len(images), n_views)
    else:
        images = None
    # Handle old training
    cfg = OmegaConf.to_container(cfg, resolve=True)
    albedo_path = cfg.get("albedo")
    roughness_path = cfg.get("roughness")

    gt_albedo = None
    if albedo_path is not None:
        gt_albedo = load_img_files(albedo_path, n_views, cfg.get("is_exr"))
        # print(n_views)
        assert len(gt_albedo) == n_views

    gt_roughness = None
    if roughness_path is not None:
        gt_roughness = load_img_files(roughness_path, n_views, cfg.get("is_exr"))
        assert len(gt_roughness) == n_views

    extra_data = {}
    sam_masks = None
    sam_masks_path = Path(cfg["cameras"]).parent / "sam_masks"
    if sam_masks_path.exists():
        sam_masks = load_sam_masks(sam_masks_path, n_views)
        assert len(sam_masks) == n_views

    extra_data["sam_masks"] = sam_masks
    return scene, transforms, images, learned_modules, gt_albedo, gt_roughness, extra_data




def load_img_files(folder: Path, limit: int = 0, is_exr: bool = True):
    if is_exr:
        files = glob_sorted(folder, "*.exr")
    else:
        files = glob_sorted(folder, "*.png")
    logger.info(f"Loading files from {folder}:\n" + ", ".join([f.name for f in files]))

    if limit <= 0:
        limit = len(files)
    # for file in files:
    #     print(file)
    #     print(mi.Bitmap(str(file)))
    loaded= [
        mi.Bitmap(str(file)) for file in files[:limit]
    ]
    # if resize_frac is not None:
    #     loaded = [
    #         m.resample(


    #         )
    #         for m in loaded
    #     ]
    # print("before", np.array(loaded[0]))
    if not is_exr:
        # loaded = [
        #     m.convert(
        #         pixel_format=mi.Bitmap.PixelFormat.RGBA,
        #         component_format=mi.Struct.Type.Float32,
        #         srgb_gamma=False

        #     )
        #     for m in loaded
        # ]
        print("====== WARNING ======: we are assuming png files are already in linear format")
        loaded = [
            m.convert(
                pixel_format=mi.Bitmap.PixelFormat.RGBA,
                component_format=mi.Struct.Type.Float32,
            )
            for m in loaded
        ]


    # print(np.array(loaded[0]))
    loaded = [
        m.convert(pixel_format=mi.Bitmap.PixelFormat.RGBA)
        for m in loaded if m.pixel_format != mi.Bitmap.PixelFormat.RGBA
    ]
    # print("after", np.array(loaded[0]))
    # print(loaded[0])
    return loaded

def load_sam_masks(folder: Path, limit: int = 0):
    files = glob_sorted(folder, "*.png")
    logger.info(f"Loading files from {folder}:\n" + ", ".join([f.name for f in files]))

    if limit <= 0:
        limit = len(files)
    loaded= [
        mi.Bitmap(str(file)) for file in files[:limit]
    ]
    return loaded



def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_mitsuba_registration():
    logger.info(f"Registered integrators: {' '.join(registered_integrators)}")
    logger.info(f"Registered bsdf: {' '.join(registered_bsdfs)}")
    logger.info(f"Registered texture: {' '.join(registered_textures)}")


def create_loss_function(config: ObjectConfig, n_steps: int):
    return loss_registry.build(
        config.name,
        inject_dict(config.config, {"n_steps": n_steps})
    )


def _mitsuba_post_init(cfg: Union[dict, DictConfig], obj: Any, scene: mi.Scene, injection: dict[str, Any] = None, kwargs_injection: dict[str, Any] = None):
    # NOTE: for unknown reasons, torch module creation must
    # happen after mitsuba object contruction (not during).
    # Therefore, initialization is finalized in post_init.

    if not hasattr(obj, "post_init"):
        return

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg, dict)

    if "kwargs" in cfg:
        bbox = scene.bbox()
        # print('==========WARNING FIXME!!!! ========= forcing bbox to be -1.0 and 1.0')
        # bbox = mi.ScalarBoundingBox3f(mi.ScalarPoint3f(-1.0, -1.0, -1.0), mi.ScalarPoint3f(1.0, 1.0, 1.0))
        kwargs_injection = kwargs_injection or {}
        assert ("scene_min" in kwargs_injection and "scene_max" in kwargs_injection) == ("scene_min" in kwargs_injection or "scene_max" in kwargs_injection)
        if "scene_min" not in kwargs_injection:
            kwargs_injection.update({
                "scene_min": bbox.min,
                "scene_max": bbox.max,
            })

        inject_dict(cfg["kwargs"], kwargs_injection)
    injection = injection or {}
    inject_dict(cfg, injection)
    if "scene" in cfg:
        cfg["scene"] = scene
    obj.post_init(**cfg)


def prepare_learned_objects(
    scene: mi.Scene,
    integrator: mi.Integrator,
    learned_modules: dict[str, nn.Module],
    envmap_cfg: EnvmapConfig,
    train_cfg: Optional[TrainConfig],
    ckpt_file: Optional[str],
    radiance_cache_ckpt: Optional[str],
    device: str,
) -> dict[str, Any]:
    result = {}

    # Add params to mi for gradient flow
    params = mi.traverse(scene)

    # NOTE: we should not add new keys to params, instead, we should have created the integrator
    # within the scene hierarchy
    # However, the current integration traverses the integrator separately.
    integrator_params = mi.traverse(integrator)

    # Register learned integrator
    if isinstance(integrator, nn.Module):
        learned_modules["integrator"] = integrator

    # Remove learned modules without parameter
    learned_modules = {
        k: v for k, v in learned_modules.items() if len(list(v.parameters())) > 0
    }

    logger.info(
        "Learned modules:\n" + ",\n".join([f"{name}: {obj}" for name, obj in learned_modules.items()])
    )

    # Create Mitsuba optimizer

    # Hard-coded: all possible key suffixes that requires dr.jit and mitsuba handling
    dr_grad_keys = [
        ".grad_activator",
        ".tensor",
    ]
    mi_optimized_keys = [
        ".mi_texture",
    ]

    # Envmap learning currently uses a Mitsuba bitmap parameter.
    assert envmap_cfg.name in {"gt", "mitsuba"}
    learn_envmap = envmap_cfg.name == "mitsuba"
    result["learn_envmap"] = learn_envmap
    if learn_envmap:
        logger.info("Enable envmap training")
        envmap_key = find_envmap_param_key(params, scene)
        mi_optimized_keys.append(envmap_key)

        gt_envmap: mi.TensorXf = params[envmap_key]

        envmap_shape = [
            envmap_cfg.config.get("height") or gt_envmap.shape[0],
            envmap_cfg.config.get("width") or gt_envmap.shape[1],
            gt_envmap.shape[2],
        ]
        logger.info(f"Learned envmap shape: {envmap_shape}")
        params[envmap_key] = dr.full(mi.TensorXf, 1, envmap_shape)
        params.update()

        result.update({
            "gt_envmap": gt_envmap,
            "envmap_key": envmap_key,
        })

    learn_flashlight_intensity = train_cfg["learn_flashlight_intensity"] if train_cfg is not None else False

    if learn_flashlight_intensity:
        logger.info("Enable flashlight intensity training")
        envmap_key = "flashlight.intensity.value"
        mi_optimized_keys.append(envmap_key)

    mi_optimized_params = {}
    mi_optimized_sdf_params = {}

    for key, obj in params.items():
        # print(key)
        if key == "dummy_grad_.bsdf.reflectance.value":
            logger.info(f"found shape grad activator. dr.enable_grad: {key}")
            dr.enable_grad(obj)
            continue
        if any((key.endswith(s) for s in dr_grad_keys)):
            logger.info(f"dr.enable_grad: {key}")
            dr.enable_grad(obj)
            continue

        if any((key.endswith(s) for s in mi_optimized_keys)):
            logger.info(f"Trained with Mitsuba: {key}")
            dr.enable_grad(obj)
            mi_optimized_params[key] = obj
            continue
        if train_cfg is not None and train_cfg.optimize_geometry:
            if key == "custom_mesh_.vertex_positions" or key == "custom_mesh_.faces":
                logger.info(f"Trained with Mitsuba: {key}")
                # if  key == "custom_mesh_.vertex_positions":
                #     dr.enable_grad(obj)
                mi_optimized_sdf_params[key] = obj

    for key, obj in integrator_params.items():
        if "grad_activator" in key:
            logger.info(f"dr.enable_grad: integrator {key}")
            dr.enable_grad(obj)
            continue
        if key == "mi_texture":
            logger.info(f"Trained with Mitsuba: integrator {key}")
            dr.enable_grad(obj)
            mi_optimized_params[key] = obj
            continue
        if key.startswith("sdf") and not isinstance(obj, torch.nn.Module): # find a better way to do this
            logger.info(f"Trained with Mitsuba: integrator {key}")
            dr.enable_grad(obj)
            mi_optimized_sdf_params[key] = obj

    learn_flashlight_global_offset = train_cfg["learn_flashlight_global_offset"] if train_cfg else False

    if learn_flashlight_global_offset:
        logger.info("Enable flashlight offset training")
        key = "flashlight_global_offset"
        flash_light_global_offset = mi.Vector3f(0.0)
        dr.make_opaque(flash_light_global_offset)
        mi_optimized_params[key] = flash_light_global_offset

    mi_optim = None
    mi_optim_geo = None
    if train_cfg is not None and (len(mi_optimized_params) > 0 or len(mi_optimized_sdf_params) > 0):
        lr = {}
        for key, obj in mi_optimized_params.items():
            lr[key] = train_cfg.learning_rate
        mi_optim = mi.ad.Adam(lr=train_cfg.learning_rate, beta_1=train_cfg.beta_1, beta_2=train_cfg.beta_2)
        mi_optim.set_learning_rate(lr)
        for key, obj in mi_optimized_params.items():
            mi_optim[key] = obj
            pass

        if train_cfg.optimize_geometry:
            lr_geo = {}
            if train_cfg.dataset.is_sdf:
                for key, obj in mi_optimized_sdf_params.items():
                    assert train_cfg.lr_decay_start < 0
                    # lr[key] = train_cfg.learning_rate# * 0.02
                    lr_geo[key] = train_cfg.learning_rate# * 10.0
                    logger.info("{} learning_rate: {}".format(key, lr_geo[key]))
                    pass
                mi_optim_geo = mi.ad.Adam(lr=train_cfg.learning_rate, beta_1=train_cfg.beta_1, beta_2=train_cfg.beta_2)
                mi_optim_geo.set_learning_rate(lr_geo)
                for key, obj in mi_optimized_sdf_params.items():
                    mi_optim_geo[key] = obj
                    pass
                pass
            else:
                # for key, obj in mi_optimized_sdf_params.items():
                assert len(list(mi_optimized_sdf_params.keys()))==2
                assert train_cfg.lr_decay_start < 0
                vertex = mi_optimized_sdf_params["custom_mesh_.vertex_positions"]
                faces = mi_optimized_sdf_params["custom_mesh_.faces"]
                del mi_optimized_sdf_params["custom_mesh_.faces"]
                print("Force 20 times lr for geo")
                mi_optim_geo = mi.ad.Adam(lr=train_cfg.learning_rate*20, beta_1=train_cfg.beta_1, beta_2=train_cfg.beta_2, uniform=True)
                pass


        # for key, obj in mi_optimized_sdf_params.items():


        params.update(mi_optim)
        params.update(mi_optim_geo)
        integrator_params.update(mi_optim)
        integrator_params.update(mi_optim_geo)

    # Create PyTorch optimizer

    torch_optimized_params = []
    torch_sdf_params = []
    for key, obj in learned_modules.items():
        logger.info(f"Trained with PyTorch: {key}")
        obj.to(device)
        if key != "integrator":
            torch_optimized_params += list(obj.parameters())
        else:
            print("integrator")
            if hasattr(obj, "network"):
                torch_optimized_params += list(obj.network.parameters())
            if hasattr(obj, "sdf") and isinstance(obj.sdf, torch.nn.Module):
                torch_sdf_params += list(obj.sdf.parameters())
                pass
            pass

    torch_optim = None
    if train_cfg is not None and (len(torch_optimized_params) > 0 or len(torch_sdf_params) > 0):
        # print("torch optimized params:", list(torch_optimized_params))
        torch_optim = torch.optim.Adam([{'params': torch_optimized_params}, {'params': torch_sdf_params, 'lr': train_cfg.learning_rate*0.02}], lr=train_cfg.learning_rate,
                                       betas=(train_cfg.beta_1, train_cfg.beta_2), amsgrad=train_cfg.amsgrad)

    logger.info(
        "Optimizer summary:\n"
        f"Mitsuba: {mi_optim is not None} ({len(mi_optimized_params)} {len(mi_optimized_sdf_params)})\n"
        f"PyTorch: {torch_optim is not None} ({len(torch_optimized_params)} ({len(torch_sdf_params)}))"
    )

    # LR scheduling
    mi_scheduler = None
    torch_scheduler = None
    if train_cfg is not None and train_cfg.lr_decay_start >= 0:
        lr_scheduler_args = (train_cfg.lr_decay_start, train_cfg.lr_decay_rate,
                             train_cfg.lr_decay_steps, train_cfg.lr_decay_min_rate)
        if mi_optim is not None:
            mi_scheduler = ExpLR(*lr_scheduler_args)
        if torch_optim is not None:
            torch_scheduler = ExpLRScheduler(torch_optim, *lr_scheduler_args)

    # Resume training
    result["step"] = 0
    if ckpt_file is not None:
        logger.info(f"Load checkpoint {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        last_step = ckpt["step"]
        result["step"] = last_step
        logger.info(f"Checkpoint step is {last_step}")

        if torch_optim is not None:
            logger.info("Load torch optim")
            # print(list(ckpt["optim"]["param_groups"]))
            # print(list(torch_optim.state_dict()["param_groups"].keys()))
            try:
                torch_optim.load_state_dict(ckpt["optim"])
            except:
                logger.warn("optim state dict load failed")
            # print("WARNING ============== skipping loading optimzier state for debug")
        for name, obj in learned_modules.items():
            if train_cfg is not None and train_cfg.load_integrator_only and name != "integrator":
                print("skipping non integrator")
                continue
            logger.info(f"Load torch module {name}")
            try:
                obj.load_state_dict(ckpt["modules"][name])
            except:
                import traceback as tb
                tb.print_exc()
                logger.warn(f"module {name} state dict load failed")
        if torch_scheduler is not None:
            if "scheduler" in ckpt:
                torch_scheduler.load_state_dict(ckpt["scheduler"])
            else:
                torch_scheduler.last_epoch = last_step - 1

        if mi_optim is not None:
            logger.info("Load mi optim")
            try:
                mi_optim.state.update({
                    k: tuple(mi.TensorXf(v.to(device)) for v in s) for k, s in ckpt["mi_optim"].items()
                })
            except:
                print("loading mi optim failed!!!")
        if mi_optim_geo is not None:
            logger.info("Load mi geo optim")
            try:
                mi_optim_geo.state.update({
                    k: tuple(mi.TensorXf(v.to(device)) for v in s) for k, s in ckpt["mi_optim_geo"].items()
                })
            except:
                logging.warn("mi optim geo restore failed")


        # print(list(ckpt["mi_params"].keys()))
        for name in itertools.chain(mi_optimized_params, mi_optimized_sdf_params):
            logger.info(f"Load mi param {name}")
            # data = mi.TensorXf(ckpt["mi_params"][name].to(device))
            if "mi_params" not in ckpt or name not in ckpt["mi_params"]:
                logging.warn("skipping loading {}".format(name))
                continue
            data = ckpt["mi_params"][name].to(device)
            if name in params:
                params[name] = data
                pass
            elif name in integrator_params:
                # print(integrator_params[name])
                # print(type(integrator_params[name]))
                # print(data)
                # print(name)
                if len(integrator_params[name]) != 1 and len(data) == 1:
                    data = data.permute(1, 0)
                integrator_params[name] = data
            # else:
            #     raise RuntimeError(name)
            # print(mi_optimized_params.keys())
            # print(name)
            if name in mi_optimized_params:
                if name in params:
                    mi_optimized_params[name] = params[name]
                    if mi_optim is not None:
                        mi_optim[name] = params[name]
                elif name in integrator_params:
                    mi_optimized_params[name] = integrator_params[name]
                    if mi_optim is not None:
                        mi_optim[name] = integrator_params[name]
                else:
                    # raise RuntimeError(name)
                    data = mi.TensorXf(data)
                    # print(data)
                    # print(mi_optim[name], type(mi_optim[name]))
                    # print(data.array)
                    data = dr.unravel(type(mi_optimized_params[name]), data.array)
                    mi_optimized_params[name] = data
                    if mi_optim is not None:
                        mi_optim[name] = data

            elif name in mi_optimized_sdf_params:
                if name in params:
                    mi_optimized_sdf_params[name] = params[name]
                    if mi_optim_geo is not None:
                        mi_optim_geo[name] = params[name]
                elif name in integrator_params:
                    mi_optimized_sdf_params[name] = integrator_params[name]
                    if mi_optim_geo is not None:
                        mi_optim_geo[name] = integrator_params[name]
                else:
                    raise RuntimeError(name)
            else:
                raise RuntimeError(name)

        if len(mi_optimized_params) > 0:
            params.update(mi_optim)
            integrator_params.update(mi_optim)

        if mi_scheduler is not None:
            mi_optim.set_learning_rate(mi_scheduler.get_lr_rate(last_step - 1) * train_cfg.learning_rate)



    if train_cfg is not None and train_cfg.optimize_geometry and not train_cfg.dataset.is_sdf:
        lambda_ = 15.0
        print("lambda", lambda_)
        ls = mi.ad.LargeSteps(vertex, faces, lambda_)
        mi_optim_geo['u'] = ls.to_differential(vertex)
        params["custom_mesh_.vertex_positions"] = ls.from_differential(mi_optim_geo['u'])
        params.update()
        pass

    if train_cfg is not None and train_cfg.learn_flashlight_intensity and train_cfg.freeze_flashlight_intensity:
        del mi_optim[envmap_key]
        dr.disable_grad(params[envmap_key])
        # del mi_optimized_params[env_key] # do not delete to keep it in checkpoint

    if radiance_cache_ckpt is not None:
        #load a pre-trained radiance cache, that is most likely trained using GT truth data directly, similar to Zhang et. al
        radiance_ckpt = torch.load(radiance_cache_ckpt, map_location="cpu")
        name = "integrator"
        logger.info(f"Load radiance cache for torch module {name}")
        learned_modules[name].load_state_dict(radiance_ckpt["modules"][name])
    result.update({
        "params": params,
        "integrator_params": integrator_params,
        "learned_modules": learned_modules,
        "mi_optimized_params": mi_optimized_params | mi_optimized_sdf_params,
        "mi_optim": mi_optim,
        "mi_optim_geo": mi_optim_geo,
        "torch_optim": torch_optim,
        "mi_scheduler": mi_scheduler,
        "torch_scheduler": torch_scheduler,
        "torch_optimized_params": torch_optimized_params,
        "ls": ls if  train_cfg is not None and (not train_cfg.dataset.is_sdf) and train_cfg.optimize_geometry else None
    })

    return result


def find_envmap_param_key(params: mi.SceneParameters, scene: mi.Scene) -> str:
    env_emitters = [em for em in scene.emitters() if em.is_environment()]
    if len(env_emitters) != 1:
        raise ValueError(f"Expecting 1 environment map in the scene, found {len(env_emitters)}")

    data = mi.traverse(env_emitters[0])["data"]
    for key, obj in params:
        if data is obj:
            return key

    raise ValueError("Couldn't find environment map data in scene params")


def compute_output_metrics(
    name: str,
    outputs: list[mi.Bitmap],
    integrator: str,
    gt: dict[str, mi.Bitmap],
):
    gt = gt.get(name)
    if gt is None:
        return {}

    is_image = name == "image"
    names = [name]
    if is_image and integrator.startswith("nerad"):
        names = ["lhs", "rhs"]
    assert len(names) == len(outputs)

    results = {}
    for name, pred in zip(names, outputs):
        metrics = compute_metrics(pred, gt)
        for key, value in metrics.items():
            results[f"{name}_{key}"] = value

    return results
