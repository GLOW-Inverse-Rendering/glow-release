import torch
import torch.nn
import torch.nn.functional as F
import drjit as dr
import mitsuba as mi
import models.nerad_wrapper
from nerad.utils.render_utils import mis_weight
from nerad.integrator.nerad import Nerad, NeradMixin, MyPathTracerMixin
from nerad.utils.mitsuba_utils import vec_to_tens_safe, float_to_tens_safe
import models.physicalshader
import models.fields
import models.renderer
import os.path
import cv2
import numpy as np
from wildlightutils import render_utils
from wildlightutils import image_utils
from wildlightutils import reg_utils
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from nerad.hook.save_checkpoint import SaveCheckpointHook
import logging
import pathlib
import time
import copy
import tinycudann as tcnn

logger = logging.getLogger(__name__)

class MyPathTracerMixin:
    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               si,
               ray,
               medium: mi.Medium,
               active: mi.Bool):

        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)
        eta = mi.Float(1)
        result = mi.Spectrum(0)
        throughput = mi.Spectrum(1)
        if models.nerad_wrapper.mitsuba_pre_1_0:
            valid_ray = mi.Mask((~mi.Bool(self.hide_emitters))
                                & dr.neq(scene.environment(), None))
        else:
            valid_ray = mi.Mask((~mi.Bool(self.hide_emitters))
                                & (scene.environment() != None))

        active = mi.Bool(active)                      # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        bsdf_ctx = mi.BSDFContext()


        # copied main loop
        # Compute a surface interaction that tracks derivatives arising
        # from differentiable shape parameters (position, normals, etc.)
        # In primal mode, this is just an ordinary ray tracing operation.

        # si = scene.ray_intersect(ray,
        #                             ray_flags=mi.RayFlags.All,
        #                             coherent=dr.eq(depth, 0))

        # Get the BSDF, potentially computes texture-space differentials
        bsdf = si.bsdf(ray)

        # ---------------------- Direct emission ----------------------

        em_hit_result = self.emitter_hit(
            scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si)
        result += em_hit_result
        # ---------------------- Emitter sampling ----------------------

        # Should we continue tracing to reach one more vertex?
        active_next = (depth + 1 < self.max_depth) & si.is_valid()
        em_sample_result = self.sample_emitter(
            scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next)
        result += em_sample_result

        # ------------------ Detached BSDF sampling -------------------
        
        bsdf_sample, bsdf_weight, ray = self.bsdf_sample(
            sampler, active, bsdf_ctx, si, bsdf, active_next)

        # ------ Update loop variables based on current interaction ------

        throughput *= bsdf_weight
        eta *= bsdf_sample.eta
        valid_ray |= active & si.is_valid() & ~mi.has_flag(
            bsdf_sample.sampled_type, mi.BSDFFlags.Null)

        # Information about the current vertex needed by the next iteration
        prev_si = si
        prev_bsdf_pdf = bsdf_sample.pdf
        prev_bsdf_delta = mi.has_flag(
            bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

        # -------------------- Stopping criterion ---------------------

        depth[si.is_valid()] += 1
        # Don't run another iteration if the throughput has reached zero
        throughput_max = dr.max(throughput)
        rr_prob = dr.minimum(throughput_max * eta**2, self.rr_prob)
        rr_active = depth >= self.rr_depth
        rr_continue = sampler.next_1d() < rr_prob
        throughput[rr_active] *= dr.rcp(dr.detach(rr_prob))
        if models.nerad_wrapper.mitsuba_pre_1_0:
            active = active_next & (
                ~rr_active | rr_continue) & dr.neq(throughput_max, 0)
        else:
            active = active_next & (
                ~rr_active | rr_continue) & (throughput_max != 0)
        # copied main loop

        # Record the following loop in its entirety
        loop = mi.Loop(name="MyPathTracer",
                       state=lambda: (sampler, ray, throughput, result,
                                      eta, depth, valid_ray, prev_si, prev_bsdf_pdf,
                                      prev_bsdf_delta, active))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)
        with dr.suspend_grad(self.detach_higher_order):
            while loop(active):
                # Compute a surface interaction that tracks derivatives arising
                # from differentiable shape parameters (position, normals, etc.)
                # In primal mode, this is just an ordinary ray tracing operation.
                if models.nerad_wrapper.mitsuba_pre_1_0():
                    si = scene.ray_intersect(ray,
                                            ray_flags=mi.RayFlags.All,
                                            coherent=dr.eq(depth, 0))
                else:
                    si = scene.ray_intersect(ray,
                                            ray_flags=mi.RayFlags.All,
                                            coherent=(depth != 0))
                # Get the BSDF, potentially computes texture-space differentials
                bsdf = si.bsdf(ray)

                # ---------------------- Direct emission ----------------------

                em_hit_result = self.emitter_hit(
                    scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si)
                result += em_hit_result
                # ---------------------- Emitter sampling ----------------------

                # Should we continue tracing to reach one more vertex?
                active_next = (depth + 1 < self.max_depth) & si.is_valid()

                em_sample_result = self.sample_emitter(
                    scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next)
                result += em_sample_result

                # ------------------ Detached BSDF sampling -------------------

                bsdf_sample, bsdf_weight, ray = self.bsdf_sample(
                    sampler, active, bsdf_ctx, si, bsdf, active_next)

                # ------ Update loop variables based on current interaction ------

                throughput *= bsdf_weight
                eta *= bsdf_sample.eta
                valid_ray |= active & si.is_valid() & ~mi.has_flag(
                    bsdf_sample.sampled_type, mi.BSDFFlags.Null)

                # Information about the current vertex needed by the next iteration
                prev_si = si
                prev_bsdf_pdf = bsdf_sample.pdf
                prev_bsdf_delta = mi.has_flag(
                    bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

                # -------------------- Stopping criterion ---------------------

                depth[si.is_valid()] += 1
                # Don't run another iteration if the throughput has reached zero
                throughput_max = dr.max(throughput)
                rr_prob = dr.minimum(throughput_max * eta**2, self.rr_prob)
                rr_active = depth >= self.rr_depth
                rr_continue = sampler.next_1d() < rr_prob
                throughput[rr_active] *= dr.rcp(dr.detach(rr_prob))
                if models.nerad_wrapper.mitsuba_pre_1_0:
                    active = active_next & (
                        ~rr_active | rr_continue) & dr.neq(throughput_max, 0)
                else:
                    active = active_next & (
                        ~rr_active | rr_continue) & (throughput_max != 0)
            aov = [depth] if self.return_depth else []
            # if counter_profiler.enabled:
            #     counter_profiler.record("integrator.depth", np.array(depth).tolist())
        # print("grad_enabled", dr.grad_enabled(result))
        if not dr.grad_enabled(result):
            dr.enable_grad(result)
        return dr.select(valid_ray, result, 0), valid_ray, aov

    def aov_names(self):
        return ['depth'] if self.return_depth else []

    def bsdf_sample(self, sampler, active, bsdf_ctx, si, bsdf, active_next):

        bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                               sampler.next_1d(),
                                               sampler.next_2d(),
                                               active_next)
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        # When the path tracer is differentiated, we must be careful that
        #   the generated Monte Carlo samples are detached (i.e. don't track
        #   derivatives) to avoid bias resulting from the combination of moving
        #   samples and discontinuous visibility. We need to re-evaluate the
        #   BSDF differentiably with the detached sample in that case. */
        if (dr.grad_enabled(ray)):
            ray = dr.detach(ray)

            # Recompute 'wo' to propagate derivatives to cosine term
            wo = si.to_local(ray.d)
            # print("bsdf_sample, all", dr.all([active, active_next]))
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, dr.all([active, active_next]))
            bsdf_weight[bsdf_pdf > 0] = bsdf_val / dr.detach(bsdf_pdf)
        return bsdf_sample, bsdf_weight, ray

    def emitter_hit(self, scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si):

        # Compute MIS weight for emitter sample from previous bounce
        ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

        mis = mis_weight(
            prev_bsdf_pdf,
            scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
        )

        em_hit_result = throughput * mis * ds.emitter.eval(si)
        # print(em_hit_result)
        return em_hit_result

    def sample_emitter(self, scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next):
        # Is emitter sampling even possible on the current vertex?
        active_em = active_next & mi.has_flag(
            bsdf.flags(), mi.BSDFFlags.Smooth)

        # If so, randomly sample an emitter without derivative tracking.
        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active_em)
        if models.nerad_wrapper.mitsuba_pre_1_0:    
            active_em &= dr.neq(ds.pdf, 0.0)
        else:
            active_em &= (ds.pdf != 0.0)

        if (dr.grad_enabled(si.p)):
            # Given the detached emitter sample, *recompute* its
            # contribution with AD to enable light source optimization
            ds.d = dr.normalize(ds.p - si.p)
            em_val = scene.eval_emitter_direction(si, ds, active_em)
            if models.nerad_wrapper.mitsuba_pre_1_0():
                em_weight = dr.select(dr.neq(ds.pdf, 0), em_val / ds.pdf, 0)
            else:
                em_weight = dr.select((ds.pdf != 0), em_val / ds.pdf, 0)
        # Evaluate BSDF * cos(theta) differentiably
        wo = si.to_local(ds.d)
        # print(si.t)
        # print("before eval_pdf")
        bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        # print("bsdf_value_em_shape", np.array(bsdf_value_em).shape)

        # print("bsdf_value_em", bsdf_value_em, dr.all(dr.all(dr.isfinite(bsdf_value_em))))
        mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
        em_sample_result = throughput * mis_em * bsdf_value_em * em_weight

        return em_sample_result



class MyPathTracer(MyPathTracerMixin):
    def __init__(self):
        super().__init__()
        # self.scene = scene
        self._init(hide_emitters=True, return_depth=False, max_depth=22, rr_depth=5, detach_higher_order=True)
        self.sampler = mi.load_dict({
            'type': 'independent',
            'sample_count': 1
        })
    
    def _init(
        self,
        hide_emitters: bool,
        return_depth: bool,
        max_depth: int,
        rr_depth: int,
        rr_prob: float = 0.95,
        **kwargs
    ):
        self.hide_emitters = hide_emitters
        self.return_depth = return_depth
        self.rr_prob = rr_prob
        self.detach_higher_order = kwargs.get("detach_higher_order", False)

        # max depth
        if max_depth < 0 and max_depth != -1:
            raise Exception(
                "\"max_depth\" must be set to -1 (infinite) or a value >= 0")

        # Map -1 (infinity) to 2^32-1 bounces
        self.max_depth = max_depth if max_depth != -1 else 0xffffffff

        if rr_depth <= 0:
            raise Exception(
                "\"rr_depth\" must be set to a value greater than zero!")
        self.rr_depth = rr_depth


    def to_string(self):
        return (
            "MyPathTracer[\n"
            f"    max_depth={self.max_depth},\n"
            f"    rr_depth={self.rr_depth},\n"
            "]"
        )
class DummyMitsubaWrapper(torch.nn.Module):
    def __init__(self, name, scale=False):
        super().__init__()
        self.name = name
        self.scale=scale
        pass
    def set_pytorch_output(self, output):
        self.output = output
    def eval(self, si, active=True):
        # print(dr.grad_enabled(si)) # si grad is enabled here
        result = self.output
        # print(dr.shape(si))
        # print(result)
        shape = dr.shape(result)
        result = result.array
        # print("before shape is", shape)
        # exit()
        if len(shape) == 1 or (len(shape) == 2 and shape[-1] == 1):
            result = mi.Vector3f(result, result, result)
        elif len(shape) == 2 and shape[-1] == 3:
            # print("m", result[:, 0], result[:, 1], result[:, 2], result[0])
            # print("resutl before ", result)
            result = dr.unravel(mi.Vector3f, result, order="F")
            # print("result after", result)
            # result = mi.Vector3f(result[:, 0], result[:, 1], result[:, 2])
            # exit()
        else:
            raise RuntimeError(shape)
        # result = dr.clamp(result, 0, 1)
        result = 1/(1+dr.exp(-result)) # this is sigmoid
        # print(self.name, "result", result, dr.shape(result))
        if self.scale :
            result = result + 1
        return result

    def traverse(self, callback):
        # callback.put_parameter("network", self.network, mi.ParamFlags.Differentiable)
        pass
class PhysicalShaderGIRenderer(NeradMixin, MyPathTracerMixin):
    def __init__(self, scene, sampler, integrator, config, *, neus_renderer=None):
        # torch.nn.Module.__init__(self)
        # myProps = mi.Properties("PhysicalShaderGIRenderer")

        # mi.SamplingIntegrator.__init__(self, myProps)
        self.config = config
        
        self.scene = scene
        # self.bsdf = bsdf # mitsuba bsdf with texture represented with neus color network
        self.sampler = sampler

        self.hide_emitters = True
        self.params = mi.traverse(scene)
        self.network = integrator.network
        if self.config.use_frozen_radiance:
            self.freeze_network = integrator.network.clone()
            self.freeze_network.to("cuda")
        else:
            self.freeze_network = integrator.network
            self.freeze_network.to("cuda")

        if self.config.scale_input is not None:
            scale_input = self.config.scale_input
            self.freeze_network.set_input_scale(self.config.scale_input)
            self.network.set_input_scale(self.config.scale_input)
        if self.config.ambient_light:
            self.env_map = models.physicalshader.EnvironmentNetwork().to("cuda")
        else:
            self.env_map = None
        self.builtin_renderer = MyPathTracer()
        # self.network.reset_field(1) # reset occlusion net
        # print("PhysicalShaderGIRenderer.__init__: resetting self.network")
        self.debug_mirror_sdf = mi.load_dict({
            'type': 'conductor',
            'material': 'none'
        })
       
        self.neus_renderer = neus_renderer

        params =  mi.traverse(self.scene)
        if self.config.bsdf == "principledmy":
            self.optimizing_texture_dict = {
                "albedo": "brdf_0.base_color.texture",
                "roughness": "brdf_0.roughness.texture",
                "eta": "brdf_0.eta.texture",
                # "clearcoat": "my-bsdf.brdf_0.clearcoat.texture",
                # "clearcoat_gloss": "my-bsdf.brdf_0.clearcoat_gloss.texture"
            }
            self.optimizing_param_dict = {
                "albedo": "my-bsdf.brdf_0.base_color.network",
                "roughness": "my-bsdf.brdf_0.roughness.network",
                "eta": "my-bsdf.brdf_0.eta.network",
                # "clearcoat": "my-bsdf.brdf_0.clearcoat.network",
                # "clearcoat_gloss": "my-bsdf.brdf_0.clearcoat_gloss.network"
            }
        elif self.config.bsdf == "principled":
            self.optimizing_texture_dict = {
                "albedo": "brdf_0.base_color.texture",
                "roughness": "brdf_0.roughness.texture",
                # "eta": "my-bsdf.brdf_0.eta.texture",
                # "clearcoat": "my-bsdf.brdf_0.clearcoat.texture",
                # "clearcoat_gloss": "my-bsdf.brdf_0.clearcoat_gloss.texture"
            }
            self.optimizing_param_dict = {
                "albedo": "my-bsdf.brdf_0.base_color.network",
                "roughness": "my-bsdf.brdf_0.roughness.network",
                # "eta": "my-bsdf.brdf_0.eta.network",
                # "clearcoat": "my-bsdf.brdf_0.clearcoat.network",
                # "clearcoat_gloss": "my-bsdf.brdf_0.clearcoat_gloss.network"
            }
        else:
            raise RuntimeError(f"Unknown bsdf {self.config.bsdf}")




        self.flash_cache = {}
        self.flash_to_world_cache_key = None
        self.flash_mode = None
        if 'flashlight.position' in params.keys():
            self.flash_mode = "pos"
        elif 'flashlight.to_world' in params.keys():
            self.flash_mode = "dir"
        else:
            raise RuntimeError(f"unknown flash mode")
        # print("aux_params", aux_params)
        # exit()
    def get_scene(self):
        return self.scene

    def update_flashlight_to_world(self, to_world):
        self.flash_to_world_cache_key = to_world
        to_world = mi.Transform4f(to_world)
        dr.make_opaque(to_world)
        self.params['flashlight.to_world'] = to_world
        
        
        
    def update_flashlight_intensity(self, intensity):
        assert intensity[0] == intensity[1] == intensity[2] # only support gray scale intensity for now
        intensity = mi.Float(intensity.mean().item())
        dr.make_opaque(intensity)
        # intensity = dr.ravel(mi.Color3f, intensity)
        # print(self.params.keys())
        # print("flashlight values", self.params['flashlight.intensity.value'])
        self.params['flashlight.intensity.value'] = intensity

    def cull_occ_mask(self, scene, si, ds, active_em):
        si_cull = dr.detach(si)
        ray_cull = si_cull.spawn_ray_to(dr.detach(ds.p)) # detach to prevent drjit from complaining
        si_cull = scene.ray_intersect(ray_cull, active_em)
        occ_mask = si_cull.is_valid() & (dr.dot(si_cull.sh_frame.n, ray_cull.d) < 0)

        ray_cull_active = si_cull.is_valid() & (~(dr.dot(si_cull.sh_frame.n, ray_cull.d) < 0))
        cull_counter = mi.Float(0)
        cull_max_iter_limit = 1
        def body(ray_cull_active, si_cull, ray_cull, cull_counter, occ_mask):
            ray_cull = si_cull.spawn_ray_to(dr.detach(ds.p))
            si_tmp = scene.ray_intersect(ray_cull, ray_flags=mi.RayFlags.All, coherent=False, active=ray_cull_active)
            si_cull[ray_cull_active] = si_tmp
            occ_mask[ray_cull_active] |= si_cull.is_valid() & (dr.dot(si_cull.sh_frame.n, ray_cull.d) < 0)
            ray_cull_active &= si_cull.is_valid() & (~(dr.dot(si_cull.n, ray_cull.d) < 0))
            cull_counter +=1
            return ray_cull_active, si_cull, ray_cull, cull_counter, occ_mask
        
        def cond(ray_cull_active, si_cull, ray_cull, cull_counter, occ_mask):
            return ray_cull_active & (cull_counter < cull_max_iter_limit)
        with dr.suspend_grad():
            if models.nerad_wrapper.mitsuba_pre_1_0():
                loop = mi.Loop("back culling", lambda: (ray_cull_active, si_cull, ray_cull, cull_counter, occ_mask))
                while loop(cond(ray_cull_active, si_cull, ray_cull, cull_counter, occ_mask)): 
                    ray_cull_active, si_cull, ray_cull, cull_counter, occ_mask = body(ray_cull_active, si_cull, ray_cull, cull_counter, occ_mask)
            else:
                ray_cull_active, si_cull, ray_cull, cull_counter, occ_mask = dr.while_loop(
                    state=(ray_cull_active, si_cull, ray_cull, cull_counter, occ_mask),
                    cond=cond,
                    body=body
                )

        return occ_mask

    def sample_emitter(self, scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next, use_mis):
        # bsdf = self.aux_bsdf
        # print("using aux bsdf", bsdf)
        # Is emitter sampling even possible on the current vertex?
        active_em = active_next & mi.has_flag(
            bsdf.flags(), mi.BSDFFlags.Smooth)
        # print("active_em_init", active_em)
        # If so, randomly sample an emitter without derivative tracking.
        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), False, active_em)
        occ_mask = self.cull_occ_mask(scene, si, ds, active_em)

        if models.nerad_wrapper.mitsuba_pre_1_0():
           active_em &= dr.neq(ds.pdf, 0.0)
        else:
            active_em &= (ds.pdf!=0.0)
        # print("WARNING no active_em")

        # if (dr.grad_enabled(si.p)):
        #     # Given the detached emitter sample, *recompute* its
        #     # contribution with AD to enable light source optimization
        #     ds.d = dr.normalize(ds.p - si.p)
        #     em_val = scene.eval_emitter_direction(si, ds, active_em)
        #     em_weight = dr.select(dr.neq(ds.pdf, 0), em_val / ds.pdf, 0)

        # Evaluate BSDF * cos(theta) differentiably
        wo = si.to_local(ds.d)
        # print(si.t)
        # print("before eval_pdf")
        # print("WARNING we are not using active for bsdf eval + negative wo")
        # bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, -wo, active_em)
        bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        # bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo)




        # print("==========================================sample emitter======")
        # print("grad enabled", dr.grad_enabled(bsdf_value_em))
        # print("bsdf_value_em_shape", np.array(bsdf_value_em).shape)
        
        # print("bsdf_value_em", bsdf_value_em, dr.all(dr.all(dr.isfinite(bsdf_value_em))))
        mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
        # print("em weight here", em_weight)
        if use_mis:
            # print("do mis")
            em_sample_result = throughput * mis_em * bsdf_value_em * em_weight
        else:
            # print("no mis")
            em_sample_result = throughput * bsdf_value_em * em_weight
        # print("WARNING: using bsdf_value_em as em_sample_result")
        # em_sample_result = bsdf_value_em
        # em_sample_result = (si.p + 1) / 2
        # em_sample_result = (ds.d+1)/2
        return em_sample_result, occ_mask
        # return ds.p, occ_mask

    @dr.wrap_ad(source="drjit", target="torch")
    def render_alpha_neus_torch(self, rays_o, rays_d, light_o, light_lum, near, far, step):

        near = torch.zeros_like(near)
        far = torch.zeros_like(far) + 2.0
        rays_o.requires_grad_(True)
        
        near_batch = near.shape[0]
        far_batch = far.shape[0]
        assert  near_batch == far_batch, (near_batch, far_batch)
        mult = rays_o.shape[0] // near_batch
        # print('mult', mult)
        assert mult * near_batch == rays_o.shape[0], (mult, near_batch, rays_o.shape[0])
        near = near.repeat_interleave(mult, dim=0)
        far = far.repeat_interleave(mult, dim=0)
        # print("emitter sample: render alpha")
        # print("occlusion", rays_o, rays_d)
        render_out = self.neus_renderer.render_alpha(rays_o, rays_d, light_o, light_lum, near, far, cos_anneal_ratio=1.0)        
        return render_out["weight_sum"]
    

    def render_alpha_neus(self, rays, light_o, light_lum, near, far, step):
        # print(dr.grad_enabled(rays.o), dr.grad_enabled(rays.d), dr.grad_enabled(light_o), dr.grad_enabled(light_lum), dr.grad_enabled(near), dr.grad_enabled(far))
        rays_o = vec_to_tens_safe(rays.o)
        rays_d = vec_to_tens_safe(rays.d)
        weight_sum = self.render_alpha_neus_torch(rays_o, dr.detach(rays_d), dr.detach(light_o), dr.detach(light_lum), dr.detach(near), dr.detach(far), dr.detach(step))
        dr.make_opaque(weight_sum)
        # weight_sum = dr.detach(weight_sum)
        # print("weight_sum", dr.grad_enabled(weight_sum))
        return weight_sum

    def sample_emitter_neus(self, scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next, light_o, light_lum, near, far, step, use_mis):
        # Is emitter sampling even possible on the current vertex?

        active_em = active_next & mi.has_flag(
            bsdf.flags(), mi.BSDFFlags.Smooth)
        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), False, active_em)
        # print("si.p", si.p)
        # ray= si.spawn_ray_to(dr.detach(ds.p)) # detach to prevent drjit from complaining
        # occlusion = self.render_alpha_neus(ray, light_o, light_lum, near, far, step).array
        # occlusion = dr.select(occlusion<=1.0, occlusion, 1.0)
        if models.nerad_wrapper.mitsuba_pre_1_0():
            active_em &= dr.neq(ds.pdf, 0.0)
        else:
            active_em &= (ds.pdf != 0.0)
        wo = si.to_local(ds.d)
        bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
        # print(occlusion)
        if use_mis:
            # print("do mis", mis_em)
            em_sample_result = throughput * mis_em * bsdf_value_em * em_weight #* (1- occlusion)
        else:
            # print("no mis", mis_em)
            em_sample_result = throughput * bsdf_value_em * em_weight #* (1- occlusion)
        # print(1-occlusion, em_sample_result)
        # em_sample_result = throughput*occlusion
        # print(em_sample_result)
        # print(occlusion_mask)
        # em_sample_result[occlusion_mask] = 0.0

        return em_sample_result
    
    def bsdf_sample(self, sampler, active, bsdf_ctx, si, bsdf, active_next):

        bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                               sampler.next_1d(),
                                               sampler.next_2d(),
                                               active_next)
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        # When the path tracer is differentiated, we must be careful that
        #   the generated Monte Carlo samples are detached (i.e. don't track
        #   derivatives) to avoid bias resulting from the combination of moving
        #   samples and discontinuous visibility. We need to re-evaluate the
        #   BSDF differentiably with the detached sample in that case. */
        if (dr.grad_enabled(ray)):
            ray = dr.detach(ray)

            # Recompute 'wo' to propagate derivatives to cosine term
            wo = si.to_local(ray.d)
            # print("bsdf_sample, all", dr.all([active, active_next]))
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, dr.all([active, active_next]))
            bsdf_weight[bsdf_pdf > 0] = bsdf_val / dr.detach(bsdf_pdf)
        return bsdf_sample, bsdf_weight, ray

    def emitter_hit(self, scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si):

        # Compute MIS weight for emitter sample from previous bounce
        ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

        mis = mis_weight(
            prev_bsdf_pdf,
            scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
        )

        em_hit_result = throughput * mis * ds.emitter.eval(si)
        # print(em_hit_result)
        return em_hit_result
    @dr.wrap_ad(source="drjit", target="torch")
    def eval_env_map_torch(self, dirs, active):
        active = active == 1.0
        out = torch.zeros_like(dirs)
        out[active] = self.env_map(dirs[active])
        env_map_finite_mask = torch.isfinite(out).all(dim=-1)
        # print("active dirs", dirs[active], "is finite", torch.isfinite(dirs[active]).all())
        # print("env_map output all, dir", dirs, dirs[~env_map_finite_mask], env_map_finite_mask.all(), out, out[~env_map_finite_mask])
        return out

    def eval_env_map(self, pts, dirs, normals, a, point_light_pos, point_light_dir, em_weight, active):
        active = dr.select(active, 1.0, 0.0)
        
        env_map_result = self.eval_env_map_torch(vec_to_tens_safe(dirs), float_to_tens_safe(active))
        env_map_result = dr.unravel(mi.Vector3f, env_map_result.array)
        return env_map_result

    def RHS_net(self, ray, scene, depth, sampler, active, step):
        t_bsdf_sample = time.time()
        if models.nerad_wrapper.mitsuba_pre_1_0():
            si = scene.ray_intersect(ray,
                            ray_flags=mi.RayFlags.All,
                            coherent=dr.eq(depth, 0))
        else:
            si = scene.ray_intersect(ray,
                            ray_flags=mi.RayFlags.All,
                            coherent=(depth != 0))
        si = dr.detach(si) # because the culling does not handle gradient properly
        # si = scene.ray_intersect(ray)
        # implement back culling manually
        # print("si here", si)


        ray_cull_active = si.is_valid() & (~(dr.dot(si.sh_frame.n, ray.d) < 0))
        ray_hit = si.is_valid() & ((dr.dot(si.sh_frame.n, ray.d) < 0))
        # mask = dr.zeros(mi.Bool)
        cull_counter = mi.Float(0)
        cull_max_iter_limit = 1
        def body(ray_cull_active, si, ray, cull_counter, ray_hit):
            ray = si.spawn_ray(dr.detach(ray.d))
            if models.nerad_wrapper.mitsuba_pre_1_0():  
                si_tmp = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0), active=ray_cull_active)
            else:
                si_tmp = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=(depth!=0), active=ray_cull_active)
            si[ray_cull_active] = si_tmp
            #     si = si_tmp
            ray_hit[ray_cull_active] |= si.is_valid() & ((dr.dot(si.n, ray.d) < 0))
            ray_cull_active &= si.is_valid() & (~(dr.dot(si.n, ray.d) < 0))
            cull_counter +=1
            return ray_cull_active, si, ray, cull_counter, ray_hit
        def cond(ray_cull_active, si, ray, cull_counter, ray_hit):
            return ray_cull_active & (cull_counter < cull_max_iter_limit)
        with dr.suspend_grad():
            if models.nerad_wrapper.mitsuba_pre_1_0():
                loop = mi.Loop("back culling", lambda: (ray_cull_active, si, ray, cull_counter, ray_hit))
            
                while loop(cond(ray_cull_active, si, ray, cull_counter, ray_hit)): 
                    ray_cull_active, si, ray, cull_counter, ray_hit = body(ray_cull_active, si, ray, cull_counter, ray_hit)
            else:
                ray_cull_active, si, ray, cull_counter, ray_hit = dr.while_loop(
                    state=(ray_cull_active, si, ray, cull_counter, ray_hit),
                    cond=cond,
                    body=body
                )
            
        active &= ray_hit

        t_bsdf_si_t  = time.time()

        # mask |= culled_mask
        #  dr.max(cull_counter), dr.mean(ray_cull_active), 
        # print("cull vs first inter",first_int_end_t - first_int_t, t_bsdf_si_t - first_int_end_t)
        
        # The BSDF lookup below computes texture-space differentials but is not used in this path.
        # bsdf = si.bsdf(ray)
        # bsdf = self.debug_mirror_sdf
        # print("warning debug bsdf")

        # ---------------------- Direct emission ----------------------
        # This does not handle backculling properly. Plus we do not need this for point light
        # bsdf_sample_result = self.emitter_hit(
        #     scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si)
        active = active & si.is_valid()
        # ---------------------- Eval RHS ----------------------
        RHS_eval_doesnt_need_grad = True
        with dr.suspend_grad(when=RHS_eval_doesnt_need_grad):
            with torch.set_grad_enabled(not RHS_eval_doesnt_need_grad):
                # print("before net")
                if step < self.config.bsdf_sample_skip_iters:
                    # print("change me!!!! I hard code 50000 to be dark field")
                    print("in the hard coded path")
                    net_ins = self.extract_inputs(si, scene, sampler, override_all_occ_field_idx=2)
                else:
                    net_ins = self.extract_inputs(si, scene, sampler)
                # print("after exact")
                # print("before net", net_ins[6])
                # print("net_ins here", net_ins)
                t_before_net = time.time()
                network_out =  self.freeze_network.eval(*net_ins)  * self.params["flashlight.intensity.value"]
                # print("network_out, active, is_valid", network_out, active, si.is_valid()) & si.is_valid()
                # print("network out", si.p, network_out, active)
                RHS_net = dr.select(active, network_out, mi.Vector3f(0))
                t_rhs = time.time()
                # print("RHS_net here", RHS_net)
                # RHS_net[ray_cull_active] = 0.0
        if self.config.ambient_light:
            env_map_valid = ~si.is_valid()
            # print(si)
            # print("net_ins, ray", net_ins[1], ray.d)
            # print("si", si)
            ray_valid = dr.all(dr.isfinite(ray.d))
            env_map_valid = env_map_valid & ray_valid
            net_ins_env = list(net_ins)
            net_ins_env[1] = -ray.d # stay compatible with extract inputs
            env_map = self.eval_env_map(*net_ins_env, env_map_valid)
            RHS_env = dr.select(env_map_valid, env_map, mi.Vector3f(0))
            with dr.suspend_grad(when=RHS_eval_doesnt_need_grad):
                with torch.set_grad_enabled(not RHS_eval_doesnt_need_grad):
                    # print("before net")
                    net_ins_ambient = list(net_ins)
                    net_ins_ambient[-1] = mi.Float(4) + 0 * net_ins_ambient[0]  # 4 is hard coded to be ambient light
                    # print("net ins ambient", net_ins_ambient)
                    network_out_ambient =  self.freeze_network.eval(*net_ins_ambient) 
                    RHS_net_ambient = dr.select(active, network_out_ambient, mi.Vector3f(0))
        else:
            RHS_net_ambient = None
            RHS_env = None
        return RHS_net, si, RHS_env, RHS_net_ambient
    def RHS_builtin_renderer(self, ray, scene, depth, sampler, active, step):
        ray_orig = ray
        # only to satisfy si requirement
        if models.nerad_wrapper.mitsuba_pre_1_0():  
            si = scene.ray_intersect(ray,
                    ray_flags=mi.RayFlags.All,
                    coherent=dr.eq(depth, 0))
        else:
            si = scene.ray_intersect(ray,
                    ray_flags=mi.RayFlags.All,
                    coherent=(depth != 0))
        si = dr.detach(si) # because the culling does not handle gradient properly
        ray_cull_active = si.is_valid() & (~(dr.dot(si.sh_frame.n, ray.d) < 0))
        ray_hit = si.is_valid() & ((dr.dot(si.sh_frame.n, ray.d) < 0))
        # mask = dr.zeros(mi.Bool)
        cull_counter = mi.Float(0)
        cull_max_iter_limit = 1
        loop = mi.Loop("back culling", lambda: (ray_cull_active, si, ray, cull_counter, ray_hit))
        
        while loop(ray_cull_active & (cull_counter < cull_max_iter_limit)): 
            ray = si.spawn_ray(dr.detach(ray.d))
            if models.nerad_wrapper.mitsuba_pre_1_0():
                si_tmp = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0), active=ray_cull_active)
            else:
                si_tmp = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=(depth!= 0), active=ray_cull_active)
            si[ray_cull_active] = si_tmp
            #     si = si_tmp
            ray_hit[ray_cull_active] |= si.is_valid() & ((dr.dot(si.n, ray.d) < 0))
            ray_cull_active &= si.is_valid() & (~(dr.dot(si.n, ray.d) < 0))
            cull_counter +=1
        active &= ray_hit
        with dr.suspend_grad():
            with torch.set_grad_enabled(False):
                rgb, valid_ray, aov = self.builtin_renderer.sample(scene,
                    sampler,
                    si,
                    ray,
                    None,
                    active)
        # only to satisfy si requirement
        return rgb, si
    @dr.wrap_ad(source="drjit", target="torch")
    def RHS_net_neus_renderer_torch(self, rays_o, rays_d, light_o, light_lum, near, far, step):
        assert self.neus_renderer is not None
        # print("rays_o shape", rays_o.shape)
        # light_o_batch = light_o.shape[0]
        # light_lum_batch = light_lum.shape[0]
        near_batch = near.shape[0]
        far_batch = far.shape[0]
        assert  near_batch == far_batch, (near_batch, far_batch)
        mult = rays_o.shape[0] // near_batch
        # print('mult', mult)
        assert mult * near_batch == rays_o.shape[0], (mult, near_batch, rays_o.shape[0])
        near = near.repeat_interleave(mult, dim=0)
        far = far.repeat_interleave(mult, dim=0)
        result = self.neus_renderer.render(rays_o, rays_d, light_o, light_lum, near, far, cos_anneal_ratio=1.0)
        return result['color_fine'].detach(), result["weight_sum"].detach()

    def RHS_net_neus_renderer(self, ray, light_o, light_lum, near, far, step, scene, depth, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta):
        assert self.neus_renderer is not None
        rays_o = ray.o
        rays_d = ray.d
        if models.nerad_wrapper.mitsuba_pre_1_0():
            si = scene.ray_intersect(ray,
                    ray_flags=mi.RayFlags.All,
                    coherent=dr.eq(depth, 0))
        else:
            si = scene.ray_intersect(ray,
                    ray_flags=mi.RayFlags.All,
                    coherent=(depth != 0))
        si = dr.detach(si) # because the culling does not handle gradient properly

        # implement back culling manually
        bsdf_sample_result = self.emitter_hit(
            scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si)
        
        color, weight_sum = self.RHS_net_neus_renderer_torch(vec_to_tens_safe(rays_o), vec_to_tens_safe(rays_d), light_o, light_lum, near, far, step)
        dr.make_opaque(color)
        dr.make_opaque(weight_sum)
        
        color = dr.unravel(mi.Color3f, color)
        weight_sum = weight_sum.array
        occlusion = 1 - weight_sum
        return color, occlusion * bsdf_sample_result

    def render_lhs(self, scene, si, sampler, override_em_weight=None, active=True):
        net_ins = self.extract_inputs(si, scene, sampler, override_em_weight)
        # if override_em_weight is not None:
        #     print(dr.shape(override_em_weight))
        #     print(dr.shape(net_ins[-1]))
        # print("render_lhs active", dr.shape(active), active)

        net_ins_active = list(net_ins) + [active]
        LHS = self.network.eval(*net_ins_active) * dr.detach(self.params["flashlight.intensity.value"])
        LHS = dr.select(si.is_valid(), LHS, mi.Vector3f(0))
        
        # print("grad enabled", torch.is_grad_enabled())
        # print("debug lhs")
        # print("LHS grad enabled", dr.grad_enabled(LHS))
        return LHS
    
    def render(self, scene, ray, si, bsdf, sampler, step, repeat_bsdf_sample=None, active=True, *, neus_light_o=None, neus_light_lum=None, neus_near=None, neus_far=None, override_material=None, override_em_occlusion=None, use_shadow=True, use_ambient=False):
        t_init = time.time()
        si_orig = si
        depth = mi.UInt32(0)
        eta = mi.Float(1)
        throughput = mi.Spectrum(1)
        valid_ray = mi.Mask((~mi.Bool(self.hide_emitters))
                            & (scene.environment() is not None))

        active = mi.Bool(active)                      # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        bsdf_ctx = mi.BSDFContext()

        # ---------------------- Direct emission ----------------------

        # E = self.emitter_hit(scene, throughput, prev_si,
        #                      prev_bsdf_pdf, prev_bsdf_delta, si) # we will never hit emitter
        t_emitter_hit = time.time()
        # ---------------------- Emitter sampling ----------------------

        active_next = si.is_valid()
        # print("albedo in render", override_albedo, override_roughness)
        # if override_albedo is not None:
        #     # self.albedo_texture.network.set_override_output(override_albedo)
        #     self.albedo_texture.network = DummyMitsubaWrapper("albedo")
        #     self.albedo_texture.network.set_pytorch_output(override_albedo)

        #     self.aux_albedo_texture.network = DummyMitsubaWrapper("albedo")
        #     self.aux_albedo_texture.network.set_pytorch_output(override_albedo)

        # if override_roughness is not None:
        #     # self.roughness_texture.network.set_override_output(override_roughness)
        #     self.roughness_texture.network = DummyMitsubaWrapper("roughness")
        #     self.roughness_texture.network.set_pytorch_output(override_roughness)

        #     self.aux_roughness_texture.network = DummyMitsubaWrapper("roughness")
        #     self.aux_roughness_texture.network.set_pytorch_output(override_roughness)
        texture_dict = {}
        bsdf_params = mi.traverse(bsdf)
        # print(list(bsdf_params.keys()))
        for k,v in self.optimizing_texture_dict.items():
            texture_dict[k] = bsdf_params[v]


        if override_material is not None:
            for key, value in texture_dict.items():
                if override_material[key] is not None:
                    if key == "eta":
                        value.network =  DummyMitsubaWrapper(key, scale=True)
                    else:
                        value.network =  DummyMitsubaWrapper(key)
                    value.network.set_pytorch_output(override_material[key])
        # bsdf = self.debug_mirror_sdf # debugging
        # print("========WARNING debug bsdf")
        do_bsdf = not self.config.no_bsdf_sample and step > self.config.bsdf_sample_skip_iters
        # if step == self.config.bsdf_sample_skip_iters:
            # self.network.reset_field(1)
            # print("resettting field")
        if self.neus_renderer is not None:
            em_sample_result = self.sample_emitter_neus(scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next, neus_light_o, neus_light_lum, neus_near, neus_far, step, use_mis=do_bsdf)
        else:
            em_sample_result_raw, occ_mask = self.sample_emitter(
                scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next, use_mis=do_bsdf)
        if override_em_occlusion is not None:
            em_sample_result = (1-mi.Color3f(override_em_occlusion.array, override_em_occlusion.array, override_em_occlusion.array)) * em_sample_result_raw 
        else:
            if use_shadow:  
                em_sample_result = dr.select(occ_mask, 0.0, em_sample_result_raw)
            else:
                em_sample_result = em_sample_result_raw

        t_sample_emitter = time.time()

        if not self.config.no_bsdf_sample : # we always compute the secondary residual
            # print("do bsdf sample", step, self.config.bsdf_sample_skip_iters)
            bsdf_sample, bsdf_weight, ray = self.bsdf_sample(
                sampler, active, bsdf_ctx, si, bsdf, active_next)            
            # t_bsdf_sample = time.time()
            # ------ Update loop variables based on current interaction ------

            throughput *= bsdf_weight
            eta *= bsdf_sample.eta
            valid_ray |= active & si.is_valid() & ~mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Null)

            # Information about the current vertex needed by the next iteration
            prev_si = si
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            depth[si.is_valid()] += 1
            # Don't run another iteration if the throughput has reached zero
            active = active_next
            t_bsdf_sample = time.time()
 
            stop_grad_radiance = dr.detach
            if self.config.renderer_type == "neus":
                # print("using neus renderer")
                RHS_net, bsdf_sample_result = self.RHS_net_neus_renderer(ray, neus_light_o, neus_light_lum, neus_near, neus_far, step,  scene, depth, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta)
            elif self.config.renderer_type == "nerad":
                # print("using rhs net")
                RHS_net, si, RHS_env, RHS_net_ambient = self.RHS_net(ray, scene, depth, sampler, active, step)
            elif self.config.renderer_type == "builtin":
                # print("using builtin")
                RHS_net, si = self.RHS_builtin_renderer(ray, scene, depth, sampler, active, step)
            else:
                raise RuntimeError("using unknown renderer")
            RHS_net = dr.select(active & si.is_valid(), RHS_net, mi.Vector3f(0))
            
            if do_bsdf:
                RHS =  throughput * stop_grad_radiance(RHS_net)  + em_sample_result
            else:
                RHS = em_sample_result
            if self.config.ambient_light  :
                RHS = RHS * use_ambient.array
                print("RHS here", RHS, use_ambient.array )
                # print("use ambient inside", use_ambient.array)
                if self.config.detach_ambient_env_throughput:                
                    RHS = RHS + dr.detach(throughput) * RHS_env
                else:
                    RHS = RHS + throughput * RHS_env
                print("RHS_env here", RHS_env)
                if do_bsdf:
                    if self.config.detach_ambient_net_throughput:
                        RHS = RHS + dr.detach(throughput) * RHS_net_ambient
                    else:
                        RHS = RHS + throughput * RHS_net_ambient 
                print("RHS_ambient here", RHS_net_ambient)
            
        else:
            # print("no bsdf sample")
            valid_ray |= active & si.is_valid() 
            RHS = em_sample_result
            
        t_end = time.time()
        rgb = dr.select(valid_ray, RHS, 0)
        sec_ray_o = ray.o
        sec_ray_d = ray.d
        if self.config.no_bsdf_sample:
            bsdf_result = None
            ambient_bsdf_result = None
        else:
            bsdf_result = vec_to_tens_safe(throughput*stop_grad_radiance(RHS_net))   
            if self.config.ambient_light:
                ambient_bsdf_result = vec_to_tens_safe(throughput * stop_grad_radiance(RHS_env + RHS_net_ambient))
            else:
                ambient_bsdf_result = None
        return vec_to_tens_safe(rgb), vec_to_tens_safe(sec_ray_o), vec_to_tens_safe(sec_ray_d), vec_to_tens_safe(em_sample_result_raw), bsdf_result, float_to_tens_safe(dr.select(occ_mask, 1.0, 0.0)), ambient_bsdf_result
    
    def extract_inputs(self, si, scene, sampler, override_em_weight=None, override_all_occ_field_idx=None):
        init_t = time.time()
        pts = si.p
        dirs = si.to_world(si.wi)
        normals = si.sh_frame.n
        # normals = dr.select(dr.dot(dirs, normals)<0, -normals, normals)
        # albedo = dr.detach(self.get_albedo_detached(si))
        three_params_t = time.time()
        params = self.scene_params(scene)
        point_light_pos = None
        point_light_dir = None

        scene_params_t = time.time()
        if override_em_weight is None:
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), False, True)

            # occluded = scene.ray_test(shadow_ray)
            # occ_mask = dr.zeros(mi.Bool)
            # si_cull = dr.detach(si)
            # ray_cull = si_cull.spawn_ray_to(dr.detach(ds.p)) # detach to prevent drjit from complaining
            # si_cull = scene.ray_intersect(ray_cull)
            # culled_mask = si_cull.is_valid() & (dr.dot(si_cull.sh_frame.n, ray_cull.d) < 0)
            # occ_mask |= culled_mask
            occ_mask = self.cull_occ_mask(scene, si, ds, True)
        # print("WARNING no visibility testing used in extract inputs")
        # wo = si.to_local(ds.d)
        # bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, True)
        emitter_t = time.time()
        with dr.suspend_grad():
            if self.flash_mode=="pos":
                point_light_pos = params['flashlight.position'] + si.p*0
                point_light_dir = None
            elif self.flash_mode == "dir":
                # point_light_pos = params['flashlight.to_world'].translation() + si.p*0
                # point_light_dir = params['flashlight.to_world'].translation() + si.p*0
                # mat_key = tuple(params["flashlight.to_world"].matrix.numpy().flatten().tolist())
                mat_key = tuple(self.flash_to_world_cache_key.flatten().tolist())
                if mat_key not in self.flash_cache:
                    # print("not hit cache", mat_key)
                    scale, quat, trans = dr.transform_decompose(params["flashlight.to_world"].matrix)
                    # point_light_pos = trans + si.p*0
                    point_light_pos = trans
                    #point_light_dir = dr.unravel(mi.Vector3f, mi.Float(trans)) + si.p*0
                    rot_mat = dr.quat_to_matrix(quat)
                                
                    rot_transform = mi.Transform4f(rot_mat)
                    result = rot_transform @ mi.Vector3f([0.0, 0.0, 1.0])
                    # point_light_dir =  result + si.p*0
                    point_light_dir =  result
                    # print("point_light_dir", point_light_dir)
                    dr.eval(point_light_pos,point_light_dir )
                    self.flash_cache[mat_key] = (trans, result)
                else:
                    # print("hit cache", mat_key)
                    point_light_pos, point_light_dir = self.flash_cache[mat_key]
        flashlight_t = time.time()
        
        # print("em_weight", em_weight)
        if override_em_weight is None:
            em_weight = mi.Float(0) + 0 * pts
            if self.config.use_separate_emitter_bsdf_cache:
                all_occ_field = 1 if override_all_occ_field_idx is None else override_all_occ_field_idx
                em_weight = dr.select(occ_mask, mi.Float(all_occ_field), mi.Float(0)) + 0 * pts
            # print("extract_pts no override", occluded, em_weight)
            # print("forcing 0th radiance cache")
        else:
            em_weight = override_em_weight
        em_weight_t = time.time()
        # print("extract", three_params_t - init_t, scene_params_t - three_params_t, emitter_t - scene_params_t, flashlight_t - emitter_t, em_weight_t - flashlight_t)
        point_light_pos = point_light_pos + 0 * pts # broadcast shape
        point_light_dir = point_light_dir + 0 * pts
        return dr.detach(pts), dr.detach(dirs), dr.detach(normals), None, dr.detach(point_light_pos), dr.detach(point_light_dir), dr.detach(em_weight)
    
    def prepare_ray(self, origins, dirs):
        dr.make_opaque(origins)
        dr.make_opaque(dirs)
        origins = dr.unravel(mi.Point3f, origins.array)
        dirs = dr.unravel(mi.Vector3f, dirs.array)

        ray = mi.Ray3f(origins, dirs)
        return ray
    
    def cast_ray(self, ray):
        si = self.scene.ray_intersect(ray)
        return si
    
    @dr.wrap_ad(source='torch', target='drjit')
    def cast_ray_torch(self, origins, dirs):
        ray = self.prepare_ray(origins, dirs)
        si = self.cast_ray(ray)
        valid_mask = dr.select(si.is_valid(), 1.0, 0.0)
        return vec_to_tens_safe(si.p), float_to_tens_safe(si.t), vec_to_tens_safe(si.n), float_to_tens_safe(valid_mask)
    
    def prepare(self, zs, origins, dirs, normals): # zs == inf means invalid ray, bsdf is provided by us
        # print("========= physical GI renderer forward", torch.is_grad_enabled())
        # print("dr.shape(dirs)", dr.shape(dirs))
        ray = self.prepare_ray(origins, dirs)
        dr.make_opaque(zs)
       

        zs = zs.array

        # print("normals here", normals)
        
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.t = zs
        si.p = ray(zs)
        # print(dr.shape(normals))

        if normals is not None:
            dr.make_opaque(normals)
            normals = dr.normalize(dr.unravel(mi.Normal3f, normals.array))
            si.sh_frame.n = normals
            si.initialize_sh_frame()
            si.n = si.sh_frame.n
        
        # print(dr.shape(si.is_valid()))
        # print(dr.shape(ray.d))
        # print(si.to_local(-ray.d))
        si.wi = dr.select(si.is_valid(), si.to_local(-ray.d), -ray.d)
        si.wavelengths = ray.wavelengths
        si.dp_du = si.sh_frame.s
        si.dp_dv = si.sh_frame.t
        # self.sampler.seed(seed, dr.shape(zs)[0])
        # dr.make_opaque(si)
        # dr.make_opaque(bsdf)
        return ray, si
    
    @dr.wrap_ad(source='torch', target='drjit')
    def forward(self, bsdf, zs, origins, dirs, normals, step, neus_light_o=None, neus_light_lum=None, neus_near=None, neus_far=None, override_eta=None, override_clearcoat=None, override_roughness=None, override_clearcoat_gloss=None, override_albedo=None, override_em_occlusion=None, use_shadow=True, use_ambient=0.0):
        # print("in mitsuba", time.time())
        override_material_dict = {
            "eta": override_eta,
            "clearcoat": override_clearcoat,
            "roughness": override_roughness,
            "clearcoat_gloss": override_clearcoat_gloss,
            "albedo": override_albedo,
        }
        init_t = time.time()
        torch.set_grad_enabled(True)
        ray, si = self.prepare(zs, origins, dirs, normals)
        self.sampler.seed(step, dr.shape(zs)[0])
        prepare_t = time.time()
        output= self.render(self.scene, ray, si, bsdf, self.sampler, step, neus_light_o=neus_light_o, neus_light_lum=neus_light_lum, neus_near=neus_near, neus_far=neus_far, override_material=override_material_dict, override_em_occlusion=override_em_occlusion, use_shadow=use_shadow, use_ambient=use_ambient)
        output_t = time.time()
        # print("time", output_t - prepare_t, prepare_t - init_t)
        return output


    @dr.wrap_ad(source='torch', target='drjit')
    def forward_lhs(self, zs, origins, dirs, normals, step, override_em_weight=None, active=True):
        torch.set_grad_enabled(True)
        # active = dr.eq(active, 0.0)
        ray, si = self.prepare(zs, origins, dirs, normals)
        if override_em_weight is not None:
            override_em_weight = dr.unravel(mi.Vector3i, override_em_weight.array)
        output= vec_to_tens_safe(self.render_lhs(self.scene, si, self.sampler, override_em_weight, active))
        # print("dr.grad_enabled(output)", dr.grad_enabled(output))
        return output
    @dr.wrap_ad(source="torch", target="drjit")
    def forward_emitter(self, zs, origins, dirs, normals, bsdf, step):
        ray, si = self.prepare(zs, origins, dirs, normals)
        self.sampler.seed(step, dr.shape(zs)[0])
        throughput = mi.Spectrum(1)
        # valid_ray = mi.Mask((~mi.Bool(self.hide_emitters))
        #                     & dr.neq(scene.environment(), None))

        bsdf_ctx = mi.BSDFContext()
        ds, em_weight = self.scene.sample_emitter_direction(
            si, self.sampler.next_2d(), False, True)
        occ_mask = self.cull_occ_mask(self.scene, si, ds, True)
        mask_val = dr.select(occ_mask, 1.0, 0.0)
        return float_to_tens_safe(mask_val)
    @classmethod
    def init_renderer(cls, sdf, scene, integrator, sample_count, config, *, neus_renderer=None):
       
        # bsdf = models.nerad_wrapper.wrap_bsdf(sdf, color_net)
        
        sampler = mi.load_dict({
            'type': 'independent',
            'sample_count': 1
        })
        # sampler = mi.load_dict({
        #     'type': 'stratified',
        #     'sample_count': sample_count
        # })
        # sampler.set_samples_per_wavefront(sample_count)
        return cls(scene, sampler, integrator, config, neus_renderer=neus_renderer)

class MaterialRenderer(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        
        self.network = network
    def forward(self, pts):
        reflect = self.network(pts)
        return reflect
    pass
    
def init_scene(scene_path, mesh_path, out_dir, config):
    cfg = models.nerad_wrapper.load_config(scene_path, mesh_path, out_dir, config=config)
    scene, integrator, learned_info, ckpt_path = models.nerad_wrapper.load_scene(cfg)
    #saving_hooks = [SaveCheckpointHook(save_cfg) for save_cfg in cfg.saving.values()]
    return scene, integrator, learned_info, ckpt_path
        
class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.set_materialize_grads(False)
        ctx.scale = scale
        return x
    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None
        return g * ctx.scale, None
def scale_grad_no_scalar(x, scale):
    return ScaleGrad.apply(x, scale)

def scale_grad(x, scale, mask=None): # borrowed from neilf++ https://github.com/apple/ml-neilfpp/blob/main/code/utils/general.py#L117
    # if scale == 0:
    #     return x.detach()
    if scale == 1:
        if mask is None:
            return x
        else:
            return torch.where(mask, x, x.detach())
    else:
        if mask is None:
            return ScaleGrad.apply(x, scale)
        else:
            return torch.where(mask, ScaleGrad.apply(x, scale), x.detach())

class PhysicalShaderGI(torch.nn.Module):
    def __init__(self, sdf, scene, integrator, learned_info, config, *, neus_renderer=None): # we need extra color network for physical materials
        super().__init__()
 
        # from nerad.mitsuba_wrapper.reflectance_net import ReflectanceMlp()
        # self.color_net = models.fields.RenderingNetwork(**mitsuba_scene_cfg.material_network)
        self.scene, self.integrator, self.learned_info = scene, integrator, learned_info
        self.config = config
        # roughness_key = "my-bsdf.brdf_0.roughness.network"
        params =  mi.traverse(self.scene)
        self.renderer = PhysicalShaderGIRenderer.init_renderer(sdf, self.scene, self.integrator, 64, config.render)

        self.params = self.learned_info["params"]
        self.mi_optim = self.learned_info["mi_optim"]
        self.torch_optim = self.learned_info["torch_optim"]
        self.bsdf = models.nerad_wrapper.find_bsdf(self.scene)
    
    def update_mi_params(self):
        print("optimize params")
        if self.mi_optim is not None:
            print("inside optimize params")
            if "flashlight.intensity.value" in self.mi_optim.variables:
                print("inside flash intensity update")
                grad = dr.grad(self.mi_optim["flashlight.intensity.value"]).torch()
                new_flashlight = torch.clamp(self.mi_optim["flashlight.intensity.value"].torch() - 5e-4 * grad, 0.0, 100.0)
                print("grad", grad, self.mi_optim["flashlight.intensity.value"].torch(), new_flashlight)
                # print(type())
                
                old_flashlight = self.mi_optim.variables["flashlight.intensity.value"]
                # print("old new", old_flashlight, new_flashlight, grad)
                if isinstance(old_flashlight, mi.Color3f):
                    new_flashlight =  mi.Color3f(new_flashlight[0,0].item(), new_flashlight[0,1].item(), new_flashlight[0,2].item())
                    # new_flashlight = dr.unravel(mi.Color3f, new_flashlight)
                    # print(type(new_flashlight), new_flashlight)
                    # exit()
                else:
                    new_flashlight =  mi.Float(new_flashlight)
                dr.make_opaque(new_flashlight)
                self.mi_optim["flashlight.intensity.value"] = new_flashlight
                # print("flashlight update",self.mi_optim["flashlight.intensity.value"] )
            # print(dr.grad(self.mi_optim["flashlight.intensity.value"]).torch())
            # print(dr.grad(self.params["flashlight.intensity.value"]).torch())
            self.mi_optim.step()
            # print("post step",  self.mi_optim["flashlight.intensity.value"], id(self.mi_optim["flashlight.intensity.value"]))
            # print("post step", params["flashlight.intensity.value"], id(params["flashlight.intensity.value"]))
            # print("post step", self.params["flashlight.intensity.value"], id(self.params["flashlight.intensity.value"]))
            # print(self.mi_optim["flashlight.intensity.value"])
            # print("after mi step", self.mi_optim["flashlight.intensity.value"], dr.grad(self.mi_optim["flashlight.intensity.value"]), dr.grad(self.params["flashlight.intensity.value"]))
            self.params.update(self.mi_optim)
            # print("post update",  self.mi_optim["flashlight.intensity.value"], id(self.mi_optim["flashlight.intensity.value"]))
            # print("post update", params["flashlight.intensity.value"], id(params["flashlight.intensity.value"]))
            # print("post update", self.params["flashlight.intensity.value"], id(self.params["flashlight.intensity.value"]))
        pass
    
    def zero_torch_grad(self):
        self.torch_optim.zero_grad()

    def update_torch_params(self):
        # for name, param in self.albedo_network.named_parameters():
        #     print(name, param.grad)
        # for name, param in self.roughness_network.named_parameters():
        #     print(name, param.grad)
        self.torch_optim.step()

    
    def gen_adjoint_pairs(self, zs, normals, rays_o, rays_d, light_to_world, light_illum, step, mode, weights=None, *, neus_input={}, use_shadow=True, use_ambient=0.0, **kwargs):
        assert zs.shape[0] == rays_o.shape[0] == rays_d.shape[0] == normals.shape[0], (zs.shape, rays_o.shape, rays_d.shape, normals.shape)
        zs.requires_grad_()
        normals.requires_grad_()
        orig_zs = zs
        orig_normals = normals
        zs = zs.repeat(2)
        rays_o = rays_o.repeat(2, 1)
        rays_d = rays_d.repeat(2, 1)
        normals = normals.repeat(2, 1)
        # print("zs", zs.shape, zs)
        if 'neus_near' in neus_input:
            neus_input['neus_near'] = neus_input['neus_near'].repeat(2, 1)
        if 'neus_far' in neus_input:
            neus_input['neus_far'] = neus_input['neus_far'].repeat(2, 1)
        # print(list(kwargs.keys()))
        for k in kwargs.keys():
            if kwargs[k] is None:
                continue
            if k == "override_em_occlusion":
                kwargs[k] = kwargs[k].repeat(2)
            else:
                kwargs[k] = kwargs[k].repeat(2, 1)
        result, sec_ray_o, sec_ray_d, em_sample_result, bsdf_result, occ_mask, bsdf_result_ambient = self(zs, normals, rays_o, rays_d, light_to_world, light_illum, step=step, neus_input=neus_input, use_shadow=use_shadow, use_ambient=use_ambient, **kwargs)
        # print("result in gen_adjoint_pairs", result)
        # if mode == "vol":
        # print("weights requires grad", weights.requires_grad)
        # print("bsdf result", bsdf_result)
        assert weights is not None
        weights_orig = weights
        weights_orig.requires_grad_() # One of the differentiated Tensors does not require grad
        
        weights = weights.repeat(2, 1)
        # print(weights.shape, result.shape)
        # result = 
        result = result.reshape(weights.shape[0], weights.shape[1], 3)
        # print(result.shape)
        split_point = result.shape[0] // 2
        assert split_point * 2 == result.shape[0]
        result_output = (weights[:, :, None] *  result).sum(dim=1) #/ weights[:, :, None].sum(dim=1).detach()
        # LHS = LHS.reshape(-1, weights.shape[1], 3)
        # LHS_output = (weights[:, :, None] *  LHS).sum(dim=1) / weights[:, :, None].sum(dim=1).detach()

        # ctx.save_for_backward(result_output[split_point:], orig_zs, orig_normals, weights_orig)
        # ctx.split_point = split_point
        adjoint_sample = result_output[split_point:] # first half
        adjoint_sample_all_pts = result[split_point:] # second half
        # if grad_scale_mask is not None:
        #     adjoint_sample = scale_grad_no_scalar(adjoint_sample, grad_scale_mask)
        result_output = result_output[:split_point] # second half
        # LHS_output = LHS_output[:split_point]
        result_output_all_pts = result[:split_point] # second half
        sec_ray_o = sec_ray_o.reshape(weights.shape[0], weights.shape[1], 3)
        sec_ray_d = sec_ray_d.reshape(weights.shape[0], weights.shape[1], 3)
        sec_ray_o = sec_ray_o[:split_point] # second half
        sec_ray_d = sec_ray_d[:split_point] # second half
        em_sample_result = em_sample_result.reshape(weights.shape[0], weights.shape[1], 3)
        em_sample_result = em_sample_result[:split_point] # second half
        if bsdf_result is not None:
            bsdf_result = bsdf_result.reshape(weights.shape[0], weights.shape[1], 3)
            bsdf_result = bsdf_result[:split_point] # second half
        if bsdf_result_ambient is not None:
            bsdf_result_ambient = bsdf_result_ambient.reshape(weights.shape[0], weights.shape[1], 3)
            bsdf_result_ambient = bsdf_result_ambient[:split_point]
        if occ_mask is not None:
            occ_mask = occ_mask.reshape(weights.shape[0], weights.shape[1])
            occ_mask = occ_mask[:split_point] # second half
        return  result_output, adjoint_sample, result_output_all_pts.detach(), adjoint_sample_all_pts, sec_ray_o, sec_ray_d, em_sample_result, bsdf_result, occ_mask, bsdf_result_ambient

    def forward(self, zs, normals, rays_o, rays_d, light_to_world, light_illum, step, *, neus_input={}, override_material=None, override_em_occlusion=None, use_shadow=True, use_ambient=0.0): # currently we are not differentiable against light_lum light_orign
        if self.config.render.use_flashlight:
            self.renderer.update_flashlight_to_world(light_to_world.detach().cpu().numpy())
        if override_material is not None:
            subsurface, metallic, specular, clearcoat, roughness, clearcoat_gloss, base_color = torch.split(override_material, [1,1,1,1,1,1,3], dim=-1)
        else:
            subsurface, metallic, specular, clearcoat, roughness, clearcoat_gloss, base_color = None, None, None, None, None, None, None
        RHS, sec_ray_o, sec_ray_d, em_sample_result, bsdf_result, occ_mask, bsdf_result_ambient= self.renderer.forward(self.bsdf, 
                        zs, 
                        rays_o, 
                        rays_d, 
                        normals, 
                        step,
                        neus_input['neus_light_o'],
                        neus_input['neus_light_lum'],
                        neus_input['neus_near'],
                        neus_input['neus_far'],
                        specular, # WARNING: this is actually eta
                        clearcoat,
                        roughness,
                        clearcoat_gloss,
                        base_color,
                        override_em_occlusion,
                        use_shadow=use_shadow,
                        use_ambient=use_ambient)
        # else:
        #     RHS, LHS= self.renderer.forward_adjoint(self.bsdf, 
        #             zs, 
        #             rays_o, 
        #             rays_d, 
        #             normals, 
        #             seed)

        
        # print(torch.linalg.norm(output, dim=-1))
        # RHS = (RHS + 1) / 2
        return RHS, sec_ray_o, sec_ray_d, em_sample_result, bsdf_result, occ_mask, bsdf_result_ambient
    
    def forward_emitter(self, zs, origins, dirs, normals, step):
        return self.renderer.forward_emitter(zs, origins, dirs, normals, self.bsdf, step)


def calc_neus_weight(rays_d, gradients_primal, sdf_primal, deviation_network, dists, batch_size, n_samples):
    sdf_primal = sdf_primal.reshape(-1).unsqueeze(dim=1)
    true_cos = (rays_d * gradients_primal).sum(-1, keepdim=True)

    # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
    # the cos value "not dead" at the beginning training iterations, for better convergence.
    iter_cos = -(F.relu(-true_cos))  # always non-positive

    # Estimate signed distances at section points
    # print(sdf_primal.shape, iter_cos.shape, dists.reshape(-1, 1).shape)
    estimated_next_sdf = sdf_primal + iter_cos * dists.reshape(-1, 1) * 0.5
    estimated_prev_sdf = sdf_primal - iter_cos * dists.reshape(-1, 1) * 0.5

    inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
    # inv_s = inv_s * 0 + 100000
    inv_s = inv_s.expand(batch_size * n_samples, 1)
    # print(estimated_next_sdf.shape, inv_s.shape)
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

    weights_primal = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device=dists.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

    return weights_primal

def sample_top_K(weights, zs, normals, sdf, deviation_network, top_k, rays_d, dists):
    device = weights.device
    batch_size = weights.shape[0]
    total_n_samples = weights.shape[1]
    _, weights_idx = torch.sort(weights, dim=-1, descending=True)
    
    xx = torch.arange(weights.shape[0], device=device)[:, None].expand(weights.shape[0], total_n_samples)[:, :top_k]
    weights_idx = weights_idx[:, :top_k]
    weights_idx,_ = torch.sort(weights_idx, dim=-1)
    # print(weights_idx)

    sdf = sdf[xx, weights_idx]
    normals = normals[xx, weights_idx]
    # inv_s = inv_s[xx, weights_idx]
    dists = dists[xx, weights_idx]

    weights = weights[xx, weights_idx]
    zs = zs[xx, weights_idx]
    return weights, zs, sdf, normals, xx, weights_idx

def sample_weight_top_k(weights, zs, normals, sdf, top_k):
    device = weights.device
    batch_size = weights.shape[0]
    total_n_samples = weights.shape[1]    
    assert len(weights.shape)==2, weights.shape
    xx = torch.arange(weights.shape[0], device=device)[:, None].expand(weights.shape[0], total_n_samples)[:, :top_k]
    yy = torch.multinomial(weights, num_samples=top_k, replacement=True)
    yy,_ = torch.sort(yy, dim=-1)


    sdf = sdf[xx, yy]
    normals = normals[xx, yy]
    weights = torch.ones_like(sdf) / top_k
    zs = zs[xx, yy]
    return weights, zs, sdf, normals, xx, yy

class ImageGridShadowCache(torch.nn.Module):
    def __init__(self,
                 width,
                 height,
                 d_out,
                 d_hidden,
                 n_layers,
                 point_encoding,
                 num_lights=1000,
                 skip_in=()):
        super().__init__()
        dims = [2] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.width = width
        self.height = height
        self.embed_fn_points = torch.nn.ModuleList()
        for i in range(num_lights):
            self.embed_fn_points.append(tcnn.Encoding(
                n_input_dims=2,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": point_encoding["n_levels"],
                    "n_features_per_level": point_encoding["n_features_per_level"],
                    "log2_hashmap_size": point_encoding["log2_hashmap_size"],
                    "base_resolution": point_encoding["base_resolution"],
                    "per_level_scale": point_encoding["per_level_scale"],
                },
                # dtype=torch.float32
            ))
        
        dims[0] += (self.embed_fn_points[0].n_output_dims - 2)
        self.num_layers = len(dims)
        self.skip_in = skip_in
        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                dims[l] += dims[0]
        
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            # out_dim = dims[l + 1]
            lin = torch.nn.Linear(dims[l], out_dim)

            setattr(self, "lin" + str(l), lin)
        self.num_lights = num_lights
        self.relu = torch.nn.ReLU()
    def forward(self, points, light_idx): # points must be long image space coordinates
        size_vec = torch.tensor([self.width, self.height]).to(points.device)
        points = points / size_vec
        points = self.embed_fn_points[light_idx](points)
        x = points.float()
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
        # print(mask_val.shape)
        # print(x.shape)
        x = x.squeeze()
        return x

def locate_n_th_intersection(sdfs, z_vals, mid_z_vals, rays_o, rays_d, n_th): # From geo-neus
    batch_size, n_samples = z_vals.shape
    
    sdf_d = sdfs.reshape(batch_size, n_samples)
    prev_sdf, next_sdf = sdf_d[:, :-1], sdf_d[:, 1:]
    sign = prev_sdf * next_sdf
    sign = torch.where(sign <= 0, torch.ones_like(sign), torch.zeros_like(sign))
    idx = reversed(torch.Tensor(range(1, n_samples)).cuda())
    tmp = torch.einsum("ab,b->ab", (sign, idx))
    vals, indices = torch.sort(tmp, dim=1, descending=True)

    sdf1 = torch.gather(sdf_d, 1, indices[:, n_th])
    sdf2 = torch.gather(sdf_d, 1, indices[:, n_th])
    
    return z_vals_sdf0, pts_sdf0
class PhysicalShadingTrainer(torch.nn.Module):
    def __init__(self, sdf_network, color_network, neus_renderer, deviation_network, scene, integrator, learning_info, ckpt_path, config, dataset):
        super().__init__()
        self.sdf_network = sdf_network
        self.color_network = color_network
        self.deviation_network = deviation_network
        self.neus_renderer = neus_renderer
        self.dataset = dataset

        self.scene, self.integrator, self.learned_info = scene, integrator, learning_info
        self.config = config
        self.physical_shader_gi = PhysicalShaderGI(sdf_network, scene, integrator, learning_info, config, neus_renderer=neus_renderer if config.render.use_neus_rhs else None)
        params =  mi.traverse(self.scene)
        optimizing_param_dict = self.physical_shader_gi.renderer.optimizing_param_dict
        self.material_renderers = {
            k: MaterialRenderer(params[v])
            for k, v in optimizing_param_dict.items()
        }

        self.light_conv_factor = torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        if self.config.model.shadow_visibility_cache.type == "embedding":
            self.shadow_visibility_cache = models.physicalshader.RadianceCacheOcclusion(**config.model.shadow_visibility_cache.config)
        elif self.config.model.shadow_visibility_cache.type == "image_grid":
            self.shadow_visibility_cache = ImageGridShadowCache(**config.model.shadow_visibility_cache.config)
        elif self.config.model.shadow_visibility_cache.type == "none":
            self.shadow_visibility_cache = None
        else:
            raise RuntimeError("unknown visibility cache type", self.config.model.shadow_visibility_cache.type)
        #params_to_train = [self.light_conv_factor]
        params_to_train = []
        # print("not training light conv factor")

        if self.shadow_visibility_cache is not None:
            params_to_train.extend(self.shadow_visibility_cache.parameters())
            # exit()
        if self.physical_shader_gi.renderer.env_map is not None:
            params_to_train.extend(self.physical_shader_gi.renderer.env_map.parameters())
        if len(params_to_train) > 0:
            self.optim = torch.optim.Adam(params_to_train, lr=self.config.train.learning_rate)
        else:
            self.optim = None
        self.restore_trainer_modules(ckpt_path)

        # self.params = self.learned_info["params"]
        # self.mi_optim = self.learned_info["mi_optim"]
        # self.torch_optim = self.learned_info["torch_optim"]
        # self.learn
    

    def get_vol_geometry(self, render_out, rays_o, rays_d, top_k, repeat_samples):
        device = rays_o.device
        
        weights = render_out['weights'] # B x N
        # print("===================weights", weights, torch.isfinite(weights).all())
        batch_size = weights.shape[0]
        total_n_samples = weights.shape[1]

        # weights = weights[xx, weights_idx]
        # print("total weight", weights.sum(dim=1))

        sdf = render_out['sdf']
        normals = render_out['gradients']
        # inv_s = render_out['inv_s']
        dists = render_out['dists']
        zs = render_out['z'].detach()

        weights, zs, sdf, normals, xx, weights_idx = sample_top_K(weights, zs, normals, sdf, self.deviation_network, top_k, rays_d, dists)

        
        zs = torch.repeat_interleave(zs.reshape(-1), repeat_samples, dim=0)
        normals = normals.reshape(-1, 3)
        normals = torch.repeat_interleave(normals, repeat_samples, dim=0)
        normals = F.normalize(normals,dim=-1)
        # print("normals", normals)
        # normals = F.normalize(normals,dim=-1)
        # print("normals", normals, normals.min(), normals.max())
        rays_o = torch.repeat_interleave(rays_o, top_k*repeat_samples, dim=0)
        rays_d = torch.repeat_interleave(rays_d, top_k*repeat_samples, dim=0)

        output_color_all_pts = render_out["sampled_color"]
        output_color_all_pts = output_color_all_pts[xx, weights_idx]
        output_color = (weights[:, :, None] * output_color_all_pts.reshape(-1, top_k, 3)).sum(dim=1) #/ weights[:, :, None].sum(dim=1)

        output_color_all_pts = torch.repeat_interleave(output_color_all_pts, repeat_samples, dim=1)

        valid_mask = torch.ones((weights.shape[0]*weights.shape[1]*repeat_samples), dtype=torch.bool, device=weights.device).bool()

        weights = torch.repeat_interleave(weights, repeat_samples, dim=1)
        return zs, normals, weights, rays_o, rays_d, output_color_all_pts, output_color, valid_mask, xx, weights_idx

    def get_mesh_geometry(self, render_out, rays_o, rays_d, repeat_samples):
        device = rays_o.device
        
        normals = render_out['gradients']
        zs = render_out['z'].detach()

        zs = torch.repeat_interleave(zs.reshape(-1), repeat_samples, dim=0)
        normals = normals.reshape(-1, 3)
        normals = torch.repeat_interleave(normals, repeat_samples, dim=0)
        normals = F.normalize(normals,dim=-1)
        batch_size = rays_o.shape[0]

        rays_o = torch.repeat_interleave(rays_o, repeat_samples, dim=0)
        rays_d = torch.repeat_interleave(rays_d, repeat_samples, dim=0)

        valid_mask = render_out["valid_mask"].bool()
        # print("valid mask inside", valid_mask.shape)
        valid_mask = torch.repeat_interleave(valid_mask, repeat_samples, dim=0)
        # weights = torch.repeat_interleave(weights, repeat_samples, dim=1)
        weights = torch.ones((batch_size, repeat_samples), dtype=rays_o.dtype, device=rays_o.device) / repeat_samples
        return zs, normals, weights, rays_o, rays_d, None, None, valid_mask
    def get_vol_sample_pts_geometry(self, render_out, rays_o, rays_d, light_o, light_lum):
        device = rays_o.device
        top_k =  self.config.geometry_type.options.top_k
        weights = render_out['weights'] # B x N
        # print("===================weights", weights, torch.isfinite(weights).all())
        batch_size = weights.shape[0]
        total_n_samples = weights.shape[1]

        # weights = weights[xx, weights_idx]
        # print("total weight", weights.sum(dim=1))

        sdf = render_out['sdf']
        normals = render_out['gradients']
        inv_s = render_out['inv_s']
        dists = render_out['dists']
        zs = render_out['z'].detach()
        # print("before", zs.shape, zs[0], weights[0], rays_o[0], rays_d[0])

        weights, zs, sdf, normals, xx, weights_idx = sample_weight_top_k(weights, zs, normals, sdf, top_k)
        # print(weights.shape, zs.shape, sdf.shape, normals.shape)
        # print(rays_o.shape, rays_d.shape, zs.shape)
        pts = rays_o[:, None] + rays_d[:, None] * zs[:, :, None]# 512 x 3
        repeat_samples = self.config.geometry_type.options.repeat_bsdf_sample

        rays_o = torch.repeat_interleave(rays_o, top_k, dim=0)
        rays_d = torch.repeat_interleave(rays_d, top_k, dim=0)
        output_color_all_pts, extra_out = self.color_network(pts.reshape(-1,3).detach(), normals.reshape(-1, 3).detach(), rays_d.detach(), light_o, light_lum, None, None)
        output_color_all_pts = output_color_all_pts.reshape(*zs.shape[:2], 3) # B x N x 3
        zs = torch.repeat_interleave(zs.reshape(-1), repeat_samples, dim=0)
        normals = normals.reshape(-1, 3)
        normals = torch.repeat_interleave(normals, repeat_samples, dim=0)
        # normals = F.normalize(normals,dim=-1)
        # print("normals", normals, normals.min(), normals.max())
        rays_o = torch.repeat_interleave(rays_o, repeat_samples, dim=0)
        rays_d = torch.repeat_interleave(rays_d, repeat_samples, dim=0)

        output_color = (weights[:, :, None] * output_color_all_pts.reshape(-1, top_k, 3)).sum(dim=1) / weights[:, :, None].sum(dim=1).detach()

        output_color_all_pts = torch.repeat_interleave(output_color_all_pts, repeat_samples, dim=1)

        valid_mask = torch.ones((weights.shape[0]), dtype=torch.bool, device=weights.device).bool()
        valid_mask = torch.repeat_interleave(valid_mask, repeat_samples, dim=0)
        weights = torch.repeat_interleave(weights, repeat_samples, dim=1)
        # print("after", zs.shape, zs[:10], weights[0], rays_o[:10], rays_d[:10] )
        return zs, normals, weights, rays_o, rays_d, output_color_all_pts, output_color, valid_mask
    def get_pts_geometry(self, render_out, rays_o, rays_d, light_o, light_lum):
        device = rays_o.device
        n_samples = 64
        n_importance = 64
        # top_k =  32
        weights = render_out['weights'] # B x N
        # _, weights_idx = torch.sort(weights, dim=-1, descending=True)
        # xx = torch.arange(rays_o.shape[0], device=device)[:, None].expand(rays_o.shape[0], n_samples + n_importance)[:, :top_k]
        # weights_idx = weights_idx[:, :top_k]
        # weights = weights[xx, weights_idx]
        # print("total weight", weights.sum(dim=1))
        # zs = (render_out['z'] * weights).sum(dim=-1) / weights.sum(dim=-1)
        zs = render_out['pts_intersect'].squeeze(dim=-1)
        # print("original zs", zs)
        # print("zs is 0", zs == 0)
        valid_mask = torch.isfinite(zs) & (zs > 1e-3)
        zs[~valid_mask] = 1.0
        pts = rays_o + rays_d * zs[:, None] # 512 x 3
        # print(pts.shape, rays_o.shape, rays_d.shape, zs.shape)
        dirs = rays_d
        # print("valid mask", valid_mask)
        # print("pts", pts)
        # print("dirs", dirs)
        # assert torch.isfinite(pts).all()
        sdf_nn_output = self.sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        brdf_params = sdf_nn_output[:, 1:1]
        feature_vector = sdf_nn_output[:, 1:]
        gradients = self.sdf_network.gradient(pts).squeeze()
        # assert torch.isfinite(pts).all()
        # assert torch.isfinite(gradients).all()
        # assert torch.isfinite(dirs).all()
        # assert torch.isfinite(light_o).all()
        # assert torch.isfinite(light_lum).all()
        # assert torch.isfinite(feature_vector).all()
        # print("rays_o", rays_o, "rays_d", rays_d, "pts", pts, "valid_mask", valid_mask)
        sampled_color, extra_out = self.color_network(pts.detach(), gradients.detach(), dirs.detach(), light_o, light_lum, brdf_params.detach(), feature_vector)
        # weights = torch.ones((rays_o.shape[0],1), device=rays_o.device)
        # print( gradients, sampled_color)
        # assert torch.isfinite(sampled_color).all()
        output_color_all_pts = sampled_color.reshape(-1, 1, 3)
        repeat_samples = self.config.geometry_type.options.repeat_bsdf_sample

        zs = torch.repeat_interleave(zs, repeat_samples, dim=0)
        weights = torch.ones((rays_o.shape[0], repeat_samples), device=rays_o.device) / repeat_samples
        rays_o = torch.repeat_interleave(rays_o, repeat_samples, dim=0)
        rays_d = torch.repeat_interleave(rays_d, repeat_samples, dim=0)
        gradients = torch.repeat_interleave(gradients, repeat_samples, dim=0)
        return zs, gradients, weights, rays_o, rays_d, output_color_all_pts, sampled_color, valid_mask




    class AdjointRender(torch.autograd.Function):
        @staticmethod
        def forward(ctx, renderer, zs, normals, rays_o, rays_d, light_to_world, light_illum, step, mode, weights=None, neus_input=None, override_material=None, override_em_occlusion=None, use_shadow=True, use_ambient=0.0):
            with torch.enable_grad():
                # result_output, result_output_all_pts, adjoint_sample, sec_ray_o, sec_ray_d, em_sample_result, bsdf_result = renderer.gen_adjoint_pairs(zs, normals, rays_o, rays_d, light_to_world, light_illum, step, mode, weights, neus_input=neus_input, override_albedo=override_albedo, override_roughness=override_roughness, override_em_occlusion=override_em_occlusion, grad_scale_mask=grad_scale_mask)
                result_output, adjoint_sample, result_output_all_pts, adjoint_sample_all_pts, sec_ray_o, sec_ray_d, em_sample_result, bsdf_result, occ_mask, bsdf_result_ambient = renderer.gen_adjoint_pairs(zs, normals, rays_o, rays_d, light_to_world, light_illum, step, mode, weights, neus_input=neus_input, override_material=override_material, override_em_occlusion=override_em_occlusion, use_shadow=use_shadow, use_ambient=use_ambient)
                ctx.save_for_backward(adjoint_sample, adjoint_sample_all_pts, zs, normals, weights, override_material, override_em_occlusion)
                # print(result_output.shape, adjoint_sample.shape, weights.shape)
                return result_output, result_output_all_pts, sec_ray_o, sec_ray_d, em_sample_result, bsdf_result, occ_mask, bsdf_result_ambient

        @staticmethod
        def backward(ctx, grad_result_output, grad_output_all_pts, grad_sec_ray_o, grad_sec_ray_d, grad_em_sample_result, grad_bsdf_result, grad_occ_mask, grad_bsdf_result_ambient):
            rerender, rerender_all_pts, zs, normals, weights, override_material, override_em_occlusion = ctx.saved_tensors
            # split_point = ctx.split_point
            # print(rerender, zs, normals)
            # print("grad outputs shape", grad_output.shape)
            #zs, normals, weights, albedo_grad, roughness_grad = torch.autograd.grad((rerender, lhs_output_all_pts), (zs, normals, weights, override_albedo, override_roughness), grad_outputs=(grad_output, grad_LHS_all_pts))
            # print("override material", override_material, "override_em_occlusion", override_em_occlusion)
            if override_material is not None:
                if override_em_occlusion is None:
                    # print("before torch.grad adjointrender")
                    zs, weights, normals, material_grad = torch.autograd.grad((rerender,), (zs, weights, normals, override_material), grad_outputs=(grad_result_output,))
                    # print("after torch.grad adjointrender")
                    occ_grad = None
                else:
                    # print("in this path", occ_grad)
                    zs, weights, normals, material_grad, occ_grad = torch.autograd.grad((rerender,), (zs, weights, normals, override_material, override_em_occlusion), grad_outputs=(grad_result_output,))
            else:
                if override_em_occlusion is None:
                    zs, weights, normals = torch.autograd.grad((rerender,), (zs, weights, normals), grad_outputs=(grad_result_output,))
                    material_grad, occ_grad = None, None
                else:
                    zs, weights, normals, occ_grad  = torch.autograd.grad((rerender,), (zs, weights, normals, override_em_occlusion), grad_outputs=(grad_result_output,))
                    material_grad = None
            print("occ_grad", occ_grad)
            return (None, zs, normals, None, None, None, None, None, None, weights, None, material_grad, occ_grad, None, None)

    class AdjointRenderSampleWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx, renderer, zs, normals, rays_o, rays_d, light_to_world, light_illum, step, mode, top_k, repeat_samples, sdf, weights_orig, zs_orig, normals_orig, weights=None, neus_input=None):
            with torch.enable_grad():
                # weights, zs, sdf, normals, xx, weights_idx = sample_weight_top_k(weights, zs, normals, sdf, self.deviation_network, top_k, rays_d, dists)
                weights_primal, zs_primal, sdf_primal, gradients_primal, xx_primal, weights_idx_primal = sample_weight_top_k(weights_orig, zs_orig, normals_orig, sdf, top_k)

                # result_output, LHS_output, result_output_all_pts, LHS_output_all_pts, adjoint_sample = renderer.gen_adjoint_pairs(zs, normals, rays_o, rays_d, light_to_world, light_illum, step, mode, weights, neus_input=neus_input)
                # ctx.save_for_backward(adjoint_sample, zs, normals, weights)
                zs_primal = torch.repeat_interleave(zs_primal.reshape(-1), repeat_samples, dim=0)
                gradients_primal = torch.repeat_interleave(gradients_primal.reshape(-1, 3), repeat_samples, dim=0)
                weights_primal = torch.repeat_interleave(weights_primal, repeat_samples, dim=1)

                with torch.enable_grad():
                    assert zs.shape[0] == rays_o.shape[0] == rays_d.shape[0] == normals.shape[0], (zs.shape, rays_o.shape, rays_d.shape, normals.shape)
                    assert zs_primal.shape[0] == zs.shape[0] == gradients_primal.shape[0] == normals.shape[0], (zs_primal.shape, gradients_primal.shape, weights_primal.shape, zs.shape)
                    orig_zs = zs
                    orig_normals = normals
                    orig_zs.requires_grad_()
                    orig_normals.requires_grad_()
                    # print("orig zs shape", zs.shape)
                    zs = torch.cat([zs_primal, zs], dim=0)
                    normals = torch.cat([gradients_primal, normals], dim=0)
                    
                    # print("zs.shape", zs.shape, normals.shape, rays_o.shape, rays_d.shape)
                    # zs = zs.repeat(2, 1)
                    rays_o = rays_o.repeat(2, 1)
                    rays_d = rays_d.repeat(2, 1)
                    # normals = normals.repeat(2, 1)
                    result, LHS = renderer(zs, normals, rays_o, rays_d, light_to_world, light_illum, step=step, neus_input=neus_input)
                                        
                    assert weights is not None
                    orig_weights = weights
                    orig_weights.requires_grad_()
                    weights = torch.cat([weights_primal, weights])
                    
                    
                    result = result.reshape(weights.shape[0], weights.shape[1], 3)
                    split_point = result.shape[0] // 2
                    assert split_point * 2 == result.shape[0]
                    result_output = (weights[:, :, None] *  result).sum(dim=1) / weights[:, :, None].sum(dim=1)
                    LHS = LHS.reshape(-1, weights.shape[1], 3)
                    LHS_output = (weights[:, :, None] *  LHS).sum(dim=1) / weights[:, :, None].sum(dim=1)

                    adjoint_sample = result_output[split_point:]
                    result_output = result_output[:split_point]
                    LHS_output = LHS_output[:split_point]
                    result_output_all_pts, LHS_output_all_pts = result[:split_point], LHS[:split_point]

                    ctx.save_for_backward(adjoint_sample, orig_zs, orig_normals, orig_weights)
                return result_output, LHS_output, result_output_all_pts, LHS_output_all_pts
                # return result_output, LHS_output, result_output_all_pts, LHS_output_all_pts

        @staticmethod
        def backward(ctx, grad_output, grad_LHS, grad_output_all_pts, grad_LHS_all_pts):
            rerender, zs, normals, weights = ctx.saved_tensors
            # split_point = ctx.split_point
            # print(rerender, zs, normals)
            # print("grad outputs shape", grad_output.shape)
            zs, normals, weights = torch.autograd.grad(rerender, (zs, normals, weights), grad_outputs=grad_output)
            # print("zs shape normals shape", zs.shape, normals.shape)

            # zs, normals = zs[split_point:], normals[split_point:]
            # print("zs", zs)
            # print("normals", zs, normals, weights)
            # final_all_input_grads = [grad_output.mm(grad) for grad in all_input_grads]
            return (None, zs, normals, None, None, None, None, None, None, None, None, None, None, None, None, weights, None)
        
    class AdjointRenderVisGeo(torch.autograd.Function):
        @staticmethod
        def forward(ctx, neus_renderer, renderer, sdf_network, deviation_network, zs, normals, rays_o, rays_d, near, far, light_to_world, light_illum, step, mode, weights, top_k, repeat_samples):
            n_samples = weights.shape[1]
            rays_o_no_repeat = rays_o.reshape(weights.shape[0], weights.shape[1],3)[:, 0]
            rays_d_no_repeat = rays_d.reshape(weights.shape[0], weights.shape[1],3)[:, 0]
            batch_size, n_samples, zs_primal, z_vals_outside_primal, sample_dist_primal = neus_renderer.sample_z(rays_o_no_repeat, rays_d_no_repeat, near, far) # these sample will be used for primal
            dists = zs_primal[..., 1:] - zs_primal[..., :-1]
            dists = torch.cat([dists, torch.tensor([sample_dist_primal], device=dists.device).expand(dists[..., :1].shape)], -1)
            zs_primal = zs_primal + dists * 0.5
            # Section midpoints
            pts_primal = rays_o_no_repeat[:, None, :] + rays_d_no_repeat[:, None, :] * zs_primal[..., :, None]  # n_rays, n_samples, 3
            # print("pts_primal", pts_primal.shape)
            # rays_o = rays_o[:, None, :].expand(pts_primal.shape)
            # rays_d = rays_d[:, None, :].expand(pts_primal.shape)
            
            zs_primal = zs_primal.reshape(-1)
            pts_primal = pts_primal.reshape(-1, 3)
            # rays_o = rays_o.reshape(-1, 3)
            # rays_d = rays_d.reshape(-1, 3)

            gradients_primal = sdf_network.gradient(pts_primal).squeeze()
            # print("zs_primal", zs_primal.shape, gradients_primal.shape)
            sdf_nn_output = sdf_network(pts_primal)
            sdf_primal = sdf_nn_output[:, :1]

            weights_primal = calc_neus_weight(torch.repeat_interleave(rays_d_no_repeat, n_samples, dim=0), gradients_primal, sdf_primal, deviation_network, dists, batch_size, n_samples)
            weights_primal, zs_primal, sdf_primal, gradients_primal, _, _ = sample_top_K(weights_primal.reshape(batch_size, n_samples), 
                                                                                       zs_primal.reshape(batch_size, n_samples), 
                                                                                       gradients_primal.reshape(batch_size, n_samples, 3), 
                                                                                       sdf_primal.reshape(batch_size, n_samples), 
                                                                                       deviation_network, 
                                                                                       top_k, 
                                                                                       rays_d_no_repeat, 
                                                                                       dists.reshape(batch_size, n_samples))

            zs_primal = torch.repeat_interleave(zs_primal.reshape(-1), repeat_samples, dim=0)
            gradients_primal = torch.repeat_interleave(gradients_primal.reshape(-1, 3), repeat_samples, dim=0)
            weights_primal = torch.repeat_interleave(weights_primal, repeat_samples, dim=1)

            with torch.enable_grad():
                assert zs.shape[0] == rays_o.shape[0] == rays_d.shape[0] == normals.shape[0], (zs.shape, rays_o.shape, rays_d.shape, normals.shape)
                assert zs_primal.shape[0] == zs.shape[0] == gradients_primal.shape[0] == normals.shape[0], (zs_primal.shape, gradients_primal.shape, weights_primal.shape, zs.shape)
                orig_zs = zs
                orig_normals = normals
                orig_zs.requires_grad_()
                orig_normals.requires_grad_()
                # print("orig zs shape", zs.shape)
                zs = torch.cat([zs_primal, zs], dim=0)
                normals = torch.cat([gradients_primal, normals], dim=0)
                
                # print("zs.shape", zs.shape, normals.shape, rays_o.shape, rays_d.shape)
                # zs = zs.repeat(2, 1)
                rays_o = rays_o.repeat(2, 1)
                rays_d = rays_d.repeat(2, 1)
                # normals = normals.repeat(2, 1)
                result, LHS = renderer(zs, normals, rays_o, rays_d, light_to_world, light_illum, seed=step)
                                    
                assert weights is not None
                orig_weights = weights
                orig_weights.requires_grad_()
                weights = torch.cat([weights_primal, weights])
                
                 
                result = result.reshape(weights.shape[0], weights.shape[1], 3)
                split_point = result.shape[0] // 2
                assert split_point * 2 == result.shape[0]
                result_output = (weights[:, :, None] *  result).sum(dim=1) / weights[:, :, None].sum(dim=1)
                LHS = LHS.reshape(-1, weights.shape[1], 3)
                LHS_output = (weights[:, :, None] *  LHS).sum(dim=1) / weights[:, :, None].sum(dim=1)

                adjoint_sample = result_output[split_point:]
                result_output = result_output[:split_point]
                LHS_output = LHS_output[:split_point]
                result_output_all_pts, LHS_output_all_pts = result[:split_point], LHS[:split_point]

                ctx.save_for_backward(adjoint_sample, orig_zs, orig_normals, orig_weights)
            return result_output, LHS_output, result_output_all_pts, LHS_output_all_pts
        @staticmethod
        def backward(ctx, grad_output, grad_LHS, grad_output_all_pts, grad_LHS_all_pts):
            rerender, zs, normals, weights = ctx.saved_tensors
            # split_point = ctx.split_point
            # print(rerender, zs, normals)
            # print("grad outputs shape", grad_output.shape)
            zs, normals, weights = torch.autograd.grad(rerender, (zs, normals, weights), grad_outputs=grad_output)
            # print("zs shape normals shape", zs.shape, normals.shape)

            # zs, normals = zs[split_point:], normals[split_point:]
            # print("zs", zs)
            # print("normals", normals)
            # final_all_input_grads = [grad_output.mm(grad) for grad in all_input_grads]
            # print('zs, normals, weights', zs, normals, weights)
            return (None, None, None, None, zs, normals, None, None, None, None, None, None, None, None, weights, None, None)

    def calc_vol_orient(self, normals_all_pts, rays_d, step):
        normals_norm = normals_all_pts / (1e-10 + torch.linalg.norm(normals_all_pts, dim=-1, keepdim=True)) # consider remove this
        cos = (-normals_norm * rays_d[:, None]).sum(dim=-1) #N
        forward_mask = (cos > 0)
        angle = torch.acos(torch.clamp(cos, min=0.0, max=1.0))
        enable_mask = (torch.linalg.norm(normals_all_pts, dim=-1) < 0.3) | ~forward_mask
        orient_loss_period_first_half = self.config.train.orient_loss_period_first_half
        orient_loss_period_first_half_target = 2.0
        orient_loss_period_steps = 520000
        orient_loss_start_annealing_step = 100000
        orient_loss_period_first_half_ratio = min(1.0, max(step - orient_loss_start_annealing_step, 0) / (orient_loss_period_steps - orient_loss_start_annealing_step))
        orient_loss_period_first_half = orient_loss_period_first_half + \
                    (orient_loss_period_first_half_target - orient_loss_period_first_half) * orient_loss_period_first_half_ratio
        print("current orient loss first half period: ", orient_loss_period_first_half)

        orient_loss_period_second_half = 2.0
        orient_loss_period_second_half_target = 0.0
        orient_loss_period_second_half_ratio = min(1.0, max(step - orient_loss_start_annealing_step, 0) / (orient_loss_period_steps - orient_loss_start_annealing_step))
        orient_loss_period = orient_loss_period_second_half + \
                    (orient_loss_period_second_half_target - orient_loss_period_second_half) * orient_loss_period_second_half_ratio
        print("current orient loss second half period: ", orient_loss_period)

        # print("angle", angle.min(), angle.max())
        orient_weight = torch.where(angle < math.pi/4, 
                            torch.clamp(torch.cos(orient_loss_period_first_half*(angle-math.pi/4)), min=0.0), 
                            torch.clamp(torch.cos(orient_loss_period*(angle-math.pi/4)), min=0.0))
        orient_weight[enable_mask] = 1.0
        # orient_weight = angle / (math.pi / 2)
        orient_weight_all = orient_weight
        # orient_weight = (weights * orient_weight).sum(dim=1) / weights.sum(dim=1)
        return orient_weight_all
    def get_required_output(self, render_out, rays_o, rays_d, near, far, light_to_world, light_o, light_illum, light_idx, pixels_x, pixels_y, step, **kwargs):
        # print("get_required_output weights", render_out["weights"])
        t_init = time.time()
        rays_o_orig = rays_o
        print("rays_o here", rays_o)
        rays_d_orig = rays_d
        geometry_type_name = self.config.geometry_type.name
        repeat_samples = self.config.geometry_type.options.repeat_bsdf_sample
        if "geometry_type_name" in kwargs:
            geometry_type_name = kwargs["geometry_type_name"]
        if "override_repeat_samples" in kwargs:
            repeat_samples = kwargs["override_repeat_samples"]
        use_shadow = kwargs["use_shadow"] if "use_shadow" in kwargs else True
        use_ambient = kwargs.get("use_ambient", 0.0)

        if "shadow_mask" in render_out:
            shadow_mask = render_out["shadow_mask"]
            del render_out["shadow_mask"]
        else:
            shadow_mask = None
        # print("geometry type name", geometry_type_name)
        # print("repeat samples", repeat_samples)
        if geometry_type_name == "vol_sample_direct_color_net" or geometry_type_name == "vol_adjoint_sample_direct_color_net":
            zs, normals, weights, rays_o, rays_d, output_color_all_pts, output_color, valid_mask = self.get_vol_sample_pts_geometry(render_out, rays_o, rays_d, light_o, light_illum)
        elif geometry_type_name.startswith("vol"):
            top_k =  self.config.geometry_type.options.top_k
            if "override_top_k" in kwargs and kwargs["override_top_k"] is not None:
                top_k = kwargs["override_top_k"]
            zs, normals, weights, rays_o, rays_d, output_color_all_pts, output_color, valid_mask, xx, yy = self.get_vol_geometry(render_out, rays_o, rays_d, top_k, repeat_samples)
            # print("top k", top_k, xx.shape, yy.shape)
        elif geometry_type_name == "mesh_bsdf_adjoint":
            zs, normals, weights, rays_o, rays_d, output_color_all_pts, output_color, valid_mask = self.get_mesh_geometry(render_out, rays_o, rays_d, repeat_samples)
            xx, yy = None, None
            # print('mesh bsdf', valid_mask.shape)
        elif geometry_type_name.startswith("pts"):
            zs, normals, weights, rays_o, rays_d, output_color_all_pts, output_color, valid_mask = self.get_pts_geometry(render_out, rays_o, rays_d, light_o, light_illum)
        else:
            raise RuntimeError(f"geometry_type {self.config.geometry_type} not supported")
        pts =  rays_o + rays_d * zs[:, None]
        t_geometry = time.time()
        grad_scale = self.config.geometry_type.options.grad_scale
        reflectance_grad_scale = self.config.geometry_type.options.reflectance_grad_scale
        # and self.config.render.use_field_occlusion_hint:

        if False:
            with torch.no_grad():
                from models.renderer import locate_intersection
                sdf = render_out['sdf']
                z_vals = render_out["z_vals"]
                mid_z_vals = render_out["mid_z_vals"]
                sdf_intersect, pts_intersect = locate_intersection(sdf, z_vals, mid_z_vals, rays_o_orig, rays_d_orig)
                pts_intersect = pts_intersect.squeeze(dim=1) # B x 3
                light_dir = light_o - pts_intersect # B x 3
                light_dir_length = torch.norm(light_dir, dim=-1, keepdim=True) # B x 1
                light_dir = light_dir / light_dir_length # B x 3
                
                shadow_near, shadow_far = self.dataset.near_far_from_sphere(pts_intersect, light_dir)
                shadow_render_out = self.neus_renderer.render_alpha(pts_intersect, light_dir, light_o, light_illum, shadow_near, light_dir_length, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=1.0)
                shadow_render_alpha = shadow_render_out["weight_sum"]
                # print("shadow_render_alpha", shadow_render_alpha)
                # print("shadow_render_alpha thres", shadow_render_alpha > 0.5)
                
                em_occlusion = shadow_render_alpha > 0.5

                em_occlusion_mitsuba = em_occlusion.repeat(1, weights.shape[1])
                em_occlusion_mitsuba = em_occlusion_mitsuba.reshape(-1).long() # .repeat(3,1) # however let's not use this to prevent discontinuity
                em_occlusion = em_occlusion.squeeze(dim=-1)
            pass
            # self.physical_shader_gi.renderer.render_alpha_neus_torch()
        else:
            em_occlusion = None
            em_occlusion_mitsuba = None

        t_em_occlusion = time.time()

        if self.config.model.shadow_visibility_cache.type == "image_grid":
            # print("pixels_x, pixels_y", pixels_x.min(), pixels_x.max())
            # print(pixels_y.min(), pixels_x.min())
            occ_mask = None
            image_coords = torch.cat([pixels_x, pixels_y], dim=-1)
            cached_visibility_all_pts_logits = self.shadow_visibility_cache(image_coords, torch.tensor(light_idx, device=pts.device))
            print("cached_visibility_all_pts_logits.shape", cached_visibility_all_pts_logits.shape)
            cached_visibility_all_pts = torch.sigmoid(torch.repeat_interleave(cached_visibility_all_pts_logits, weights.shape[1], dim=0))
            cached_visibility_all_pts = ((cached_visibility_all_pts > 0.5).float().detach() + cached_visibility_all_pts) - cached_visibility_all_pts.detach()
            cached_visibility_all_pts_mask = cached_visibility_all_pts > 0.5
        elif False:
            pts_mesh, t_mesh, normals_mesh = self.physical_shader_gi.renderer.cast_ray_torch(rays_o_orig, rays_d_orig)
            occ_mask = self.physical_shader_gi.forward_emitter(t_mesh, rays_o_orig, rays_d_orig, normals_mesh, step)
        
            cached_visibility_all_pts_logits = self.shadow_visibility_cache(pts.reshape(-1, 3), torch.tensor(light_idx, device=pts.device).unsqueeze(dim=0).repeat(pts.shape[0])).squeeze(dim=1)
            cached_visibility_all_pts = torch.sigmoid(cached_visibility_all_pts_logits)
            
        else:
            occ_mask = None
            cached_visibility_all_pts_logits = None
            cached_visibility_all_pts = None
            cached_visibility_all_pts_mask = None
       
        if geometry_type_name == "vol_adjoint":
            output, LHS, output_all_pts, LHS_all_pts = self.AdjointRenderVisGeo.apply(self.neus_renderer, 
                                                                                    self.physical_shader_gi, 
                                                                                    self.sdf_network, 
                                                                                    self.deviation_network, 
                                                                                    scale_grad(zs, grad_scale), 
                                                                                    scale_grad(normals, grad_scale), 
                                                                                    rays_o, 
                                                                                    rays_d, 
                                                                                    near, 
                                                                                    far, 
                                                                                    light_to_world, 
                                                                                    light_illum, 
                                                                                    step, 
                                                                                    self.config.geometry_type.name, 
                                                                                    scale_grad(weights, grad_scale),
                                                                                    self.config.geometry_type.options.top_k,
                                                                                    self.config.geometry_type.options.repeat_bsdf_sample,
                                                                                    )
        elif geometry_type_name == "vol_bsdf_adjoint" or geometry_type_name == "vol_sample_direct_color_net" or geometry_type_name == "mesh_bsdf_adjoint":
            if self.config.render.use_neus_material:
                brdf_params = render_out["brdf_params"]
                if xx is not None and yy is not None:
                   brdf_params = brdf_params[xx, yy]
                brdf_params = torch.repeat_interleave(brdf_params, repeat_samples, dim=1)
            with torch.profiler.record_function("adjointrender"):
                output, output_all_pts, sec_ray_o, sec_ray_d, em_sample_result_no_occ, bsdf_result, occ_mask_vol, bsdf_result_ambient = self.AdjointRender.apply(self.physical_shader_gi, 
                                                                                    scale_grad(zs, grad_scale).float(), 
                                                                                    scale_grad(normals, grad_scale).float(), 
                                                                                    rays_o, 
                                                                                    rays_d, 
                                                                                    light_to_world, 
                                                                                    light_illum, 
                                                                                    step, 
                                                                                    self.config.geometry_type.name, 
                                                                                    scale_grad(weights, grad_scale).float(),
                                                                                    {
                                                                                        'neus_light_o': light_o,
                                                                                        'neus_light_lum': light_illum,
                                                                                        'neus_near': near,
                                                                                        'neus_far': far
                                                                                    },
                                                                                    # None,
                                                                                    # None,
                                                                                    # albedo.reshape(-1, 3).float() if self.config.render.use_neus_material else None,
                                                                                    # roughness.reshape(-1, 1).float() if self.config.render.use_neus_material else None,
                                                                                    scale_grad(brdf_params.reshape(-1, brdf_params.shape[-1]).float(), reflectance_grad_scale) if self.config.render.use_neus_material else None,
                                                                                    cached_visibility_all_pts,
                                                                                    use_shadow,
                                                                                    use_ambient # use_ambient
                                                                                    ) 

        elif geometry_type_name == "vol_bsdf_direct":
            if self.config.render.use_neus_material:
                brdf_params = render_out["brdf_params"]
                brdf_params = brdf_params[xx, yy]
                brdf_params = torch.repeat_interleave(brdf_params, repeat_samples, dim=1)
            output_all_pts, sec_ray_o, sec_ray_d, em_sample_result_no_occ, bsdf_result, occ_mask_vol, bsdf_result_ambient = self.physical_shader_gi(
                                                                    scale_grad(zs, grad_scale).float(), 
                                                                    scale_grad(normals, grad_scale).float(), 
                                                                    rays_o, 
                                                                    rays_d, 
                                                                    light_to_world, 
                                                                    light_illum, 
                                                                    step, 
                                                                    neus_input={
                                                                        'neus_light_o': light_o,
                                                                        'neus_light_lum': light_illum,
                                                                        'neus_near': near,
                                                                        'neus_far': far
                                                                    },
                                                                    override_material=brdf_params.reshape(-1, brdf_params.shape[-1]).float() if self.config.render.use_neus_material else None,
                                                                    use_shadow=use_shadow,
                                                                    use_ambient=use_ambient
                                                                    ) 
            output_all_pts = output_all_pts.reshape(weights.shape[0], weights.shape[1], 3)
            sec_ray_o = sec_ray_o.reshape(weights.shape[0], weights.shape[1], 3)
            sec_ray_d = sec_ray_d.reshape(weights.shape[0], weights.shape[1], 3)
            em_sample_result_no_occ = em_sample_result_no_occ.reshape(weights.shape[0], weights.shape[1], 3)
            if bsdf_result is not None:
                bsdf_result = bsdf_result.reshape(weights.shape[0], weights.shape[1], 3)
            if bsdf_result_ambient is not None:
                bsdf_result_ambient = bsdf_result_ambient.reshape(weights.shape[0], weights.shape[1], 3)
            # output = (scale_grad(weights[:, :, None], grad_scale) * output_all_pts).sum(dim=1) #/ weights[:, :, None].sum(dim=1).detach()
            output = (weights[:, :, None] * output_all_pts).sum(dim=1)
        elif geometry_type_name == "vol_adjoint_sample_direct_color_net":
            sdf = render_out['sdf']
            weights_orig = render_out['weights'] # B x N
            
            zs_orig = render_out['z'].detach()
            normals_orig = render_out['gradients']
            output, LHS, output_all_pts, LHS_all_pts = self.AdjointRenderSampleWeight.apply(self.physical_shader_gi, 
                                                                                scale_grad(zs, grad_scale), 
                                                                                scale_grad(normals, grad_scale), 
                                                                                rays_o, 
                                                                                rays_d, 
                                                                                light_to_world, 
                                                                                light_illum, 
                                                                                step, 
                                                                                self.config.geometry_type.name, 
                                                                                self.config.geometry_type.options.top_k,
                                                                                self.config.geometry_type.options.repeat_bsdf_sample,
                                                                                sdf,
                                                                                weights_orig.detach(),
                                                                                zs_orig.detach(),
                                                                                normals_orig.detach(),
                                                                                scale_grad(weights, grad_scale),
                                                                                {
                                                                                    'neus_light_o': light_o,
                                                                                    'neus_light_lum': light_illum,
                                                                                    'neus_near': near,
                                                                                    'neus_far': far
                                                                                })
        else:
            raise NotImplementedError(self.config.geometry_type.name)

        t_render = time.time()
        if self.config.train.subsample_radiosity is not None:
            assert repeat_samples==1
            subsample_lhs_rate = self.config.train.subsample_radiosity
            device = weights.device
            batch_size = weights.shape[0]
            lhs_total_n_samples = weights.shape[1]
            lhs_xx = torch.arange(weights.shape[0], device=device)[:, None].expand(weights.shape[0], lhs_total_n_samples)[:, :subsample_lhs_rate]
            lhs_weights_idx = torch.multinomial(weights, num_samples=subsample_lhs_rate, replacement=True)
            lhs_weights_idx,_ = torch.sort(lhs_weights_idx, dim=-1)

            zs_lhs = zs.reshape(weights.shape[0], weights.shape[1])[lhs_xx, lhs_weights_idx]
            rays_o_lhs = rays_o.reshape(weights.shape[0], weights.shape[1],3)[lhs_xx, lhs_weights_idx]
            rays_d_lhs = rays_d.reshape(weights.shape[0], weights.shape[1],3)[lhs_xx, lhs_weights_idx]
            normals_lhs = normals.reshape(weights.shape[0], weights.shape[1],3)[lhs_xx, lhs_weights_idx]
            weights_lhs = weights[lhs_xx, lhs_weights_idx].reshape(batch_size, subsample_lhs_rate)
            valid_mask_lhs = valid_mask.reshape(weights.shape[0], weights.shape[1])[lhs_xx, subsample_lhs_rate].reshape(-1)
        else:
            rays_o_lhs = rays_o
            rays_d_lhs = rays_d
            normals_lhs = normals
            weights_lhs = weights
            zs_lhs = zs
            valid_mask_lhs = valid_mask
        if self.config.render.use_separate_emitter_bsdf_cache:
            override_em_weight = torch.full_like(rays_o_lhs, 3, dtype=torch.int) # 3 is network 0 - > direct
            LHS_all_pts = self.physical_shader_gi.renderer.forward_lhs(zs_lhs.detach(), rays_o_lhs, rays_d_lhs, normals_lhs.detach(), step, override_em_weight, valid_mask_lhs.cpu().numpy())
            LHS_all_pts = LHS_all_pts.reshape(weights_lhs.shape[0], weights_lhs.shape[1], 3)
            override_em_weight = torch.ones_like(rays_o_lhs, dtype=torch.int)
            LHS_all_pts_all_occ = self.physical_shader_gi.renderer.forward_lhs(zs_lhs.detach(), rays_o_lhs, rays_d_lhs, normals_lhs.detach(), step, override_em_weight, valid_mask_lhs.cpu().numpy())
            LHS_all_pts_all_occ = LHS_all_pts_all_occ.reshape(weights_lhs.shape[0], weights_lhs.shape[1], 3)
            if self.config.render.ambient_light:
                override_em_weight = torch.full_like(rays_o_lhs, 4, dtype=torch.int)
                LHS_all_pts_ambient = self.physical_shader_gi.renderer.forward_lhs(zs_lhs.detach(), rays_o_lhs, rays_d_lhs, normals_lhs.detach(), step, override_em_weight, valid_mask_lhs.cpu().numpy())
                LHS_all_pts_ambient = LHS_all_pts_ambient.reshape(weights_lhs.shape[0], weights_lhs.shape[1], 3)
            else:
                LHS_all_pts_ambient = None
        else:
            LHS_all_pts = self.physical_shader_gi.renderer.forward_lhs(zs_lhs.detach(), rays_o_lhs, rays_d_lhs, normals_lhs.detach(), step, None, valid_mask_lhs.cpu().numpy())
            LHS_all_pts = LHS_all_pts.reshape(weights_lhs.shape[0], weights_lhs.shape[1], 3)
        t_lhs = time.time()
        occ_mask_vol = occ_mask_vol.reshape(weights.shape[0], weights.shape[1])
        
        LHS = (LHS_all_pts * weights_lhs.detach()[..., None]).sum(dim=1) / weights_lhs[..., None].sum(dim=1)
        print("WARNING normalizing LHS")
        if self.config.geometry_type.name.startswith("vol"):
            output_all_pts = output_all_pts.reshape(-1, weights.shape[1], 3)
        elif self.config.geometry_type.name.startswith("pts"):
            raise NotImplemented()
            pass
        elif self.config.geometry_type.name.startswith("mesh"):
            output_all_pts = output_all_pts.reshape(-1, weights.shape[1], 3)
        else:
            raise RuntimeError(f"geometry_type {self.config.geometry_type} not supported")


        normals_all_pts = normals.reshape(-1, weights.shape[1], 3)
        normals = (weights[:, :, None].detach() * normals_all_pts).sum(dim=1) / weights[:, :, None].sum(dim=1).detach()
        zs_out = (weights[:, :, None].detach() * zs.reshape(-1, weights.shape[1], 1)).sum(dim=1) / weights[:, :, None].sum(dim=1).detach()

        # if self.config.train.protective_loss_type == "orient_vol":
        #     orient_all = self.calc_vol_orient(normals_all_pts, rays_d_orig, step)
        #     output_all_pts = scale_grad_no_scalar(output_all_pts, orient_all[:,:, None])
        #     weights = scale_grad_no_scalar(weights, orient_all)

        # output = (scale_grad(weights[:, :, None], grad_scale) * output_all_pts).sum(dim=1) #/ weights[:, :, None].sum(dim=1).detach()

        # print("normals here before", normals_all_pts, normals_all_pts.min(), normals_all_pts.max())
        out = {
            "gi_rendered": output,
            "gi_rendered_all_pts": output_all_pts,
            "color_all_pts": output_color_all_pts,
            "color_rendered": output_color,
            'LHS':LHS,
            "LHS_all_pts": LHS_all_pts,
            "weights": weights,
            "valid_mask": valid_mask,
            "normals": normals,
            "normals_all_pts": normals_all_pts,
            "zs": zs_out,
            "zs_all_pts": zs.reshape(-1, weights.shape[1]),
            "sec_rays_o": sec_ray_o,
            "sec_rays_d": sec_ray_d,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "em_occlusion": em_occlusion,
            "occ_mask": occ_mask,
            "occ_mask_vol": occ_mask_vol,
            "cached_visibility_all_pts_logits": cached_visibility_all_pts_logits,
            "cached_visibility_all_pts": cached_visibility_all_pts,
            "pts": pts
        }

        if self.config.render.use_separate_emitter_bsdf_cache:
            out["LHS_all_pts_all_occ"] = LHS_all_pts_all_occ
            out["LHS_all_occ"] = (weights_lhs[:, :, None].detach() * LHS_all_pts_all_occ).sum(dim=1) / weights_lhs[:, :, None].sum(dim=1).detach()

            out["gi_rendered_all_pts_no_occ"] = em_sample_result_no_occ + bsdf_result if bsdf_result is not None else em_sample_result_no_occ
            out["gi_rendered_no_occ"] = (weights[:, :, None].detach() * out["gi_rendered_all_pts_no_occ"]).sum(dim=1) / weights[:, :, None].sum(dim=1).detach()

            out["gi_rendered_emitter_only"] = (weights[:, :, None].detach() * em_sample_result_no_occ).sum(dim=1) / weights[:, :, None].sum(dim=1).detach()
            out["gi_rendered_emitter_only_all_pts"] = em_sample_result_no_occ
            out["gi_rendered_all_pts_all_occ"] = bsdf_result if bsdf_result is not None else torch.zeros_like(em_sample_result_no_occ)
            out["gi_rendered_all_occ"] = (weights[:, :, None].detach() * out["gi_rendered_all_pts_all_occ"]).sum(dim=1) / weights[:, :, None].sum(dim=1).detach()

            if self.config.render.ambient_light:
                out["LHS_all_pts_ambient"] = LHS_all_pts_ambient
                out["gi_rendered_all_pts_ambient"] = bsdf_result_ambient
                if LHS_all_pts_ambient is not None:
                    out["LHS_ambient"] = (weights_lhs[:, :, None].detach() *LHS_all_pts_ambient).sum(dim=1) / weights[:, :, None].sum(dim=1).detach()
                if bsdf_result_ambient is not None:
                    out["gi_rendered_ambient"] = (weights[:, :, None].detach() *bsdf_result_ambient).sum(dim=1) / weights[:, :, None].sum(dim=1).detach()
                
            if self.config.train.subsample_radiosity is not None:
                subsample_lhs_rate = self.config.train.subsample_radiosity
                # print("gi_rendered_all_pts_no_occ",  out["gi_rendered_all_pts_no_occ"].shape)
                # print("gi_rendered_emitter_only", out["gi_rendered_emitter_only"].shape)
                # exit()
                # out["subsample_LHS_all_pts_all_occ"] = out["LHS_all_pts_all_occ"][lhs_xx, lhs_weights_idx]
                out["subsample_gi_rendered_all_pts"] = out["gi_rendered_all_pts"][lhs_xx, lhs_weights_idx]

                out["subsample_gi_rendered_all_pts_no_occ"] = out["gi_rendered_all_pts_no_occ"][lhs_xx, lhs_weights_idx]
                out["subsample_gi_rendered_all_pts_all_occ"]  = out["gi_rendered_all_pts_all_occ"][lhs_xx, lhs_weights_idx]
                out["subsample_gi_rendered_emitter_only"]  = out["gi_rendered_emitter_only_all_pts"][lhs_xx, lhs_weights_idx]
                
                out["subsample_occ_mask_vol"] = out["occ_mask_vol"][lhs_xx, lhs_weights_idx]
                if self.config.render.ambient_light:
                    out["subsample_gi_rendered_all_pts_ambient"] = out["gi_rendered_all_pts_ambient"][lhs_xx, lhs_weights_idx]
            else:
                # out["subsample_LHS_all_pts_all_occ"] = out["LHS_all_pts_all_occ"]
                out["subsample_gi_rendered_all_pts"] = out["gi_rendered_all_pts"]
                out["subsample_gi_rendered_all_pts_no_occ"] = out["gi_rendered_all_pts_no_occ"]
                out["subsample_gi_rendered_all_pts_all_occ"]  = out["gi_rendered_all_pts_all_occ"]
                out["subsample_gi_rendered_emitter_only"]  = out["gi_rendered_emitter_only_all_pts"]
                
                if self.config.render.ambient_light:
                    out["subsample_gi_rendered_all_pts_ambient"] = out["gi_rendered_all_pts_ambient"]
        if self.config.train.subsample_radiosity is not None:
            subsample_lhs_rate = self.config.train.subsample_radiosity
            out["subsample_gi_rendered_all_pts"] = out["gi_rendered_all_pts"][lhs_xx, lhs_weights_idx]
            out["subsample_weights"] = weights_lhs
        else:
            # out["subsample_LHS_all_pts_all_occ"] = out["LHS_all_pts_all_occ"]
            out["subsample_gi_rendered_all_pts"] = out["gi_rendered_all_pts"]
            out["subsample_weights"] = weights
        # if "req_material" in options and options["req_material"]:
        if True:
            # print(rays_o.shape, rays_d.shape, zs.shape)
            # pts = rays_o + rays_d * zs[:, None]
            
            # print("albedo & roughness just from neus")
            # print("albedo range", albedo.min(), albedo.max(), albedo.mean())
            if self.config.render.use_neus_material:
                # albedo_all_pts = F.sigmoid(albedo)
                # # albedo_all_pts = albedo
                # roughness_all_pts = F.sigmoid(roughness[:, :, None].repeat(1,1, 3))

                subsurface, metallic, specular, clearcoat, roughness, clearcoat_gloss, base_color = torch.split(F.sigmoid(brdf_params), [1,1,1,1,1,1,3], dim=-1)

                material_all_pts_dict = {
                    "eta": specular, # * 2.0 yes it should be multiplied by two. but for visualization we should keep it between 0-1
                    "clearcoat": clearcoat,
                    "roughness": roughness,
                    "clearcoat_gloss": clearcoat_gloss,
                    "albedo": base_color
                }
            else:
                # albedo_all_pts = self.albedo_renderer(pts)
                # roughness_all_pts = self.roughness_renderer(pts)
                material_all_pts_dict = {
                    k: v(pts)
                    for k,v in self.material_renderers.items()
                }
            material_dict = {
                k: (weights[:, :, None].detach() * v.reshape(weights.shape[0], weights.shape[1], -1)).sum(dim=1)  / weights[:, :, None].sum(dim=1).detach()
                for k,v in material_all_pts_dict.items()
            }
            for k,v in material_all_pts_dict.items():
                out["{}_all_pts".format(k)] = v
            for k,v in material_dict.items():
                out["{}".format(k)] = v
        
        t_misc = time.time()
        # print("geometry", t_geometry - t_init, "occlusion", t_em_occlusion - t_geometry, "render", t_render - t_em_occlusion, "lhs", t_lhs-t_render,  "misc", t_misc - t_lhs)
        return out

    def estimate_diffuse_value(self, zs, rays_o, rays_d, albedo, roughness, normals, light_o ):
        # pts = rays_o[:, None, :] + zs[:, :, None] * rays_d[:, None, :]
        # print("rays_o rays_d zs", rays_o.shape, rays_d.shape, zs.shape)
        pts = rays_o + zs[:, None] * rays_d
        light_dirs = pts - light_o
        view_dirs = rays_d
        flashlight = 1 
        print("WARNING: forcing flash to be 1 in estimate diffuse value") 
        irradiance = flashlight / (light_dirs*light_dirs).sum(-1,keepdim=True) # (..., 3) or (..., 1)
        print("irradiance here", irradiance)
        normals = F.normalize(normals, dim=-1)
        light_dirs = F.normalize(light_dirs, dim=-1)
        view_dirs = F.normalize(view_dirs, dim=-1)

        



        # falloff = F.relu(-(normals * light_dirs).sum(-1)) # (...)
        # forward_facing = (normals * view_dirs).sum(-1) < 0
        # visible_mask = ((falloff > 0) & forward_facing) # (...) boolean
        # falloff = torch.where(visible_mask, falloff, torch.zeros(1, device=falloff.device)) # (...) cosine falloff, 0 if not visible
        # irradiance = torch.unsqueeze(falloff, dim=-1) * irradiance  # (..., 3) or (..., 1)

        # diffuse_color = albedo * irradiance
        # class DummyBRDFConfig(dict):
        #     def __init__(self):
        #         se

        #     def get(self, key, default):
        #         return default

        brdf_config = {
            "no_sigmoid": True,
        }

            
        brdf_params = torch.zeros(pts.shape[0], 9, device=pts.device)
        brdf_params[:, -3:] = albedo
        brdf_params[:, 4] = roughness.mean(dim=-1)
        brdf_params[:, 2] = 1
        diffuse_color, specular_color  = models.physicalshader._apply_shading_burley(pts, normals, view_dirs, light_dirs, irradiance, brdf_params, brdf_config)
        return diffuse_color, specular_color, irradiance

    def forward(self, render_out, rays_o, rays_d, near, far, light_to_world, light_o, light_illum, step, max_steps, true_rgb, idx, pixels_x, pixels_y, combined_mask=None, shadow_mask=None, is_val=False):
        # print("forward")
        print("sample idx", idx)
        if self.config.train.adapt_step is not None and self.config.train.adapt_step > 0:
            assert self.config.train.mitsuba_start_step is not None
            mitsuba_steps = step - self.config.train.mitsuba_start_step
            assert mitsuba_steps >= 0
            if mitsuba_steps < self.config.train.adapt_step:
                print("no geo grad")
                render_out = render_utils.detach_rec(render_out)
                
                # light_o.requires_grad_()
        if combined_mask is None:
            # print(true_rgb.shape)
            combined_mask = torch.ones_like(true_rgb)
            # print(combined_mask.shape)
        render_out["shadow_mask"] = shadow_mask
        rays_o.requires_grad_(True)
        print("light illum", light_illum)
        if self.config.render.ambient_light:
            use_ambient = (light_illum)[0]
        else:
            use_ambient = 0.0
        print("use_ambient", use_ambient)
        with torch.profiler.record_function("get_required_output_primary"):
            use_shadow = self.config.model.shadow_visibility_cache.type == "image_grid" # use shadow if we are using shadow_visibility_cache of image_grid
            out = self.get_required_output(render_out, rays_o, rays_d, near, far, light_to_world, light_o, light_illum, idx, pixels_x, pixels_y, step, use_shadow=use_shadow, use_ambient=use_ambient)

        weights = out["weights"]
        sec_rays_o = out["sec_rays_o"]#.reshape(weights.shape[0], weights.shape[1], 3)
        sec_rays_d = out["sec_rays_d"]#.reshape(weights.shape[0], weights.shape[1], 3)
        xx = torch.arange(weights.shape[0], device=weights.device)[:, None].expand(weights.shape[0], 1)
        yy = torch.multinomial(weights, num_samples=1, replacement=True) #B x 1
        yy,_ = torch.sort(yy, dim=-1)
        sec_rays_o = sec_rays_o[xx, yy][:, 0]
        sec_rays_d = sec_rays_d[xx, yy][:, 0]
        sec_rays_near, sec_rays_far = self.dataset.near_far_from_sphere(sec_rays_o, sec_rays_d)
        if self.config.train.use_neus_secondary_bounce:
            render_out_sec = self.neus_renderer.render(sec_rays_o, sec_rays_d, light_o, light_illum, sec_rays_near, sec_rays_far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=1.0)
            # print("render_out_sec", render_out_sec["weights"])
        else:
            pts, zs, normals, valid_mask = self.physical_shader_gi.renderer.cast_ray_torch(sec_rays_o, sec_rays_d)
            
            render_out_sec = {
                "z": zs,
                "gradients": normals,
                "valid_mask": valid_mask
            }
            if False:
                sdf_nn_output = self.sdf_network(pts)
                sdf = sdf_nn_output[:, :1]
                n_brdf_dim = self.color_network.n_brdf_dim
                brdf_params = sdf_nn_output[:, 1:1+n_brdf_dim]
                render_out_sec["brdf_params"] = brdf_params.unsqueeze(dim=1)
        
        render_out_sec = render_utils.detach_rec(render_out_sec)
        if self.config.train.use_secondary_bounce:
            if self.config.train.secondary_random_light:
                if is_val:
                    sec_light_to_world = light_to_world
                    sec_light_o = light_o
                    sec_mitsuba_light_lumen = light_illum
                    sec_light_idx = idx
                else:
                    sec_light_idx = np.random.randint(self.dataset.n_images)
                    sec_light_to_world, sec_mitsuba_light_lumen = self.dataset.gen_light_params_pose(sec_light_idx)
                    print("sec using index", sec_light_idx, "main idx", idx)
                    sec_light_o = sec_light_to_world[:3, 3]

                
                    
            sec_out = self.get_required_output(render_out_sec, sec_rays_o, sec_rays_d, sec_rays_near, sec_rays_far, sec_light_to_world, sec_light_o, sec_mitsuba_light_lumen, sec_light_idx, pixels_x, pixels_y, step, override_top_k=self.config.train.secondary_subsample, use_ambient=use_ambient) # , override_repeat_samples=4
            sec_gi_rendered_all_pts = sec_out["gi_rendered_all_pts"]
            sec_LHS_all_pts = sec_out["LHS_all_pts"]
            sec_weights = sec_out["weights"]
            

        gi_rendered_all_pts = out["gi_rendered_all_pts"]
        gi_rendered = out["gi_rendered"]
        color_all_pts = out["color_all_pts"]
        color_rendered = out["color_rendered"]
        
        LHS_all_pts = out["LHS_all_pts"]
        # print(color_all_pts.shape)
        # print(combined_mask.shape)
        combined_mask = combined_mask.bool().all(-1)
        rendering_mask = out["valid_mask"]
        em_occlusion = out["em_occlusion"]
        normals = out["normals"]
        normals_all_pts = out["normals_all_pts"]
        occ_mask = out["occ_mask"]
        # print(combined_mask.shape, rendering_mask.shape)
        repeat_samples = self.config.geometry_type.options.repeat_bsdf_sample
        combined_mask &= rendering_mask.reshape(weights.shape[0], weights.shape[1]).all(dim=-1)
        if em_occlusion is not None:
            combined_mask &= (em_occlusion==0)
        # if shadow_mask is not None:
        #     combined_mask = (shadow_mask.mean(dim=-1) > 0.1)  * combined_mask
        if shadow_mask is not None:
            # occ_mask_combined = (weights * occ_mask).sum(dim=1) / weights.sum(dim=1)
            # print(occ_mask_combined.min(), occ_mask_combined.max())
            # occ_mask_combined =occ_mask_combined > 0.5
            shadow_mask_combined = shadow_mask.any(dim=1)
            # print(combined_mask.shape, shadow_mask.shape, occ_mask_combined.shape)
            # combined_mask &= (~((~shadow_mask_combined.bool()) & occ_mask_combined)) # exclude non overlap area between shadow_mask and occ_mask
            # combined_mask &= (~((shadow_mask_combined.bool()) & occ_mask_combined))
            combined_mask &= shadow_mask_combined.bool()
            # combined_mask &= (~occ_mask_combined)
            # print("combined_mask", combined_mask)
            # print("occ_mask", occ_mask_combined)
        # cap_pixel_val = 1.0
        # print("warning cap_pixel val hard coded to be 1")
        # valid_pixel_mask = ((color_fine < cap_pixel_val) | (true_rgb < cap_pixel_val)).float()
        

        out["valid_mask"] = combined_mask
        # print("combined_mask here", combined_mask)
        mask_sum = combined_mask.sum() + 1e-5

        normals_norm = normals / (1e-10 + torch.linalg.norm(normals, dim=-1, keepdim=True)) # consider remove this

        

        if self.config.train.protective_loss_type == "orient":
            cos = (-normals_norm * rays_d).sum(dim=1) #N
            # print("cos shape", cos.shape)
            forward_mask = (cos > 0)
            angle = torch.acos(torch.clamp(cos, min=0.0, max=1.0))
            enable_mask = (torch.linalg.norm(normals, dim=-1) < 0.2) | ~forward_mask
            orient_loss_period = 0.0 # short period
            # print("orient loss period set to 2")
            orient_loss_period_first_half = self.config.train.orient_loss_period_first_half


            # orient_loss_period_first_half_target = 2.0
            # orient_loss_period_steps = 520000
            # orient_loss_start_annealing_step = 100000
            # orient_loss_period_first_half_ratio = min(1.0, max(step - orient_loss_start_annealing_step, 0) / (orient_loss_period_steps - orient_loss_start_annealing_step))
            # orient_loss_period_first_half = orient_loss_period_first_half + \
            #             (orient_loss_period_first_half_target - orient_loss_period_first_half) * orient_loss_period_first_half_ratio
            # print("current orient loss first half period: ", orient_loss_period_first_half)


            # orient_loss_period_second_half = 2.0
            # orient_loss_period_second_half_target = 0.0
            # orient_loss_period_second_half_ratio = min(1.0, max(step - orient_loss_start_annealing_step, 0) / (orient_loss_period_steps - orient_loss_start_annealing_step))
            # orient_loss_period = orient_loss_period_second_half + \
            #             (orient_loss_period_second_half_target - orient_loss_period_second_half) * orient_loss_period_second_half_ratio
            # print("current orient loss second half period: ", orient_loss_period)

            orient_weight = torch.where(angle < math.pi/4, 
                                torch.clamp(torch.cos(orient_loss_period_first_half*(angle-math.pi/4)), min=0.0), 
                                torch.clamp(torch.cos(orient_loss_period*(angle-math.pi/4)), min=0.0))
            orient_weight[enable_mask] = 1.0
            # orient_weight = angle / (math.pi / 2)

            # orient_weight[:] = 1.0 # disable orient
            orient_weight = orient_weight.detach().unsqueeze(dim=-1).repeat(1,3)
            # print(orient_weight.shape)
        elif self.config.train.protective_loss_type == "orient_vol":

            orient_all = self.calc_vol_orient(normals_all_pts, rays_d, step)
            orient_weight = (weights * orient_all).sum(dim=1) / weights.sum(dim=1)
            # orient_weight[:] = 1.0 # disable orient
            orient_weight = orient_weight.detach().unsqueeze(dim=-1).repeat(1,3)
        elif self.config.train.protective_loss_type == "orient_min":
            orient_all = self.calc_vol_orient(normals_all_pts, rays_d, step)
            max_weights, _ = weights.max(dim=-1, keepdims=True)
            orient_ratio = orient_all / (weights/max_weights+1e-6)
            # print(orient_all, weights, orient_ratio)
            orient_weight,_ = torch.min(orient_ratio, dim=-1)
            # orient_weight = torch.clamp(orient_weight, min=0.0, max=1.0)
            # orient_weight = (weights * orient_all).sum(dim=-1) / (weights.sum(dim=-1)+1e-6)
            orient_weight = orient_weight.detach().unsqueeze(dim=-1).repeat(1,3)
        elif self.config.train.protective_loss_type == "diffuse_ratio":
            zs_all_pts = out["zs_all_pts"]
            processed_rays_o = out["rays_o"]
            processed_rays_d = out["rays_d"]
            albedo_all_pts = out["albedo_all_pts"]
            roughness_all_pts = out["roughness_all_pts"]
            normals_all_pts = out["normals_all_pts"]

            diffuse_color_all_pts, specular_color_all_pts, irradiance = self.estimate_diffuse_value(zs_all_pts.reshape(-1), processed_rays_o.reshape(-1, 3), processed_rays_d.reshape(-1, 3), albedo_all_pts.reshape(-1,3), roughness_all_pts.reshape(-1, 3), normals_all_pts.reshape(-1, 3), light_o )
            diffuse_color_all_pts = diffuse_color_all_pts.reshape(gi_rendered_all_pts.shape[0], gi_rendered_all_pts.shape[1], 3)
            specular_color_all_pts = specular_color_all_pts.reshape(gi_rendered_all_pts.shape[0], gi_rendered_all_pts.shape[1], 3) # B x N x 3
            irradiance = irradiance.reshape(gi_rendered_all_pts.shape[0], gi_rendered_all_pts.shape[1], 1)
            specular_color_all_pts_no_rad = specular_color_all_pts / irradiance
            protective_weight = 1 / (25*torch.pow(specular_color_all_pts, exponent=2.0) +1e-6)
            # protective_weight = torch.exp(-1/(1.0-torch.clamp(specular_color_all_pts, min=0.0, max=1.0)))
            protective_weight = torch.clamp(protective_weight, min=0.0, max=1.0)
            protective_weight = (diffuse_color_all_pts) / (gi_rendered_all_pts + 1e-6)
            # print(weights.shape, protective_weight.shape)
            # print("diffuse color", diffuse_color_all_pts.min(), diffuse_color_all_pts.max())
            # print("specular color", specular_color_all_pts.min(), specular_color_all_pts.max())
            # print("protective weight here", protective_weight.min(), protective_weight.max())
            orient_weight = (weights[:, :, None] * protective_weight).sum(dim=1) / weights[:, :, None].sum(dim=1)
            # orient_weight = torch.all((specular_color_all_pts == 0.0), dim=1).float()
            # orient_weight += 0.0001
            orient_weight = torch.clamp(orient_weight, min=0.0, max=1.0).detach()

            # print("orient weight", orient_weight)
            diffuse_color = (weights[:, :, None] * diffuse_color_all_pts).sum(dim=1) / weights[:, :, None].sum(dim=1)
            specular_color = (weights[:, :, None] * specular_color_all_pts).sum(dim=1) / weights[:, :, None].sum(dim=1)
            out["diffuse_color"] = diffuse_color
            out["specular_color"] = specular_color
            pass
        elif self.config.train.protective_loss_type == "specular_grad":
            zs_all_pts = out["zs_all_pts"]
            processed_rays_o = out["rays_o"]
            processed_rays_d = out["rays_d"]
            albedo_all_pts = out["albedo_all_pts"]
            roughness_all_pts = out["roughness_all_pts"]
            normals_all_pts = out["normals_all_pts"]
            with torch.enable_grad():
                normals_all_pts.requires_grad_(True)
                diffuse_color_all_pts, specular_color_all_pts, irradiance = self.estimate_diffuse_value(zs_all_pts.reshape(-1), processed_rays_o.reshape(-1, 3), processed_rays_d.reshape(-1, 3), albedo_all_pts.reshape(-1,3), roughness_all_pts.reshape(-1, 3), normals_all_pts.reshape(-1, 3), light_o )
                specular_all_pts_grad = torch.autograd.grad(specular_color_all_pts, normals_all_pts, grad_outputs=torch.ones_like(specular_color_all_pts), retain_graph=True)[0]
                diffuse_all_pts_grad = torch.autograd.grad(diffuse_color_all_pts, normals_all_pts, grad_outputs=torch.ones_like(diffuse_color_all_pts) )[0]

                diffuse_color_all_pts = diffuse_color_all_pts.reshape(weights.shape[0], weights.shape[1], 3)
                specular_color_all_pts = specular_color_all_pts.reshape(weights.shape[0], weights.shape[1], 3)
                specular_all_pts_grad = specular_all_pts_grad.reshape(weights.shape[0], weights.shape[1], 3)
                diffuse_all_pts_grad = diffuse_all_pts_grad.reshape(weights.shape[0], weights.shape[1], 3)
                
            specular_all_pts_grad_norm = (specular_all_pts_grad*specular_all_pts_grad).sum(dim=-1).sqrt()
            diffuse_all_pts_grad_norm = (diffuse_all_pts_grad*diffuse_all_pts_grad).sum(dim=-1).sqrt()
            # print("normals_all_pts_grad_norm", normals_all_pts_grad_norm.min(), normals_all_pts_grad_norm.max(), normals_all_pts_grad_norm.mean())
            # print("diffuse_all_pts_grad_norm", diffuse_all_pts_grad_norm.min(), diffuse_all_pts_grad_norm.max(), diffuse_all_pts_grad_norm.mean())
            filter_grad = 0.5



            max_allowed_grad = 0.3
            # print("specular_all_pts_grad_norm", specular_all_pts_grad_norm.shape, specular_all_pts_grad_norm.min(), specular_all_pts_grad_norm.max(), specular_all_pts_grad_norm.mean())
            
            weighted_grad_norm = (specular_all_pts_grad_norm * weights) / weights.sum(dim=1, keepdim=True)
            # print("weighted_grad_norm", weighted_grad_norm.shape, weighted_grad_norm.min(), weighted_grad_norm.max(), weighted_grad_norm.mean())
            
            # spec_ratio = max_allowed_grad / weighted_grad_norm
            # print("spec_ratio", spec_ratio.shape, spec_ratio.min(), spec_ratio.max(), spec_ratio.mean())
            # spec_ratio = torch.clamp(spec_ratio, max=1.0)
            # coeff = torch.min(spec_ratio) # min ratio for entire batch
            # print("coeff", coeff)

            weighted_grad_norm_sum = weighted_grad_norm.sum(dim=1)
            orient_weight_filtering = torch.where(weighted_grad_norm_sum > filter_grad, 0.0, 1.0)
            spec_ratio =  max_allowed_grad / weighted_grad_norm_sum
            spec_ratio = torch.where(weighted_grad_norm_sum > filter_grad, 10, spec_ratio)
            spec_ratio = torch.clamp(spec_ratio, max=1.0)
            coeff = torch.min(spec_ratio)

            # orient_weight = coeff * torch.ones((weights.shape[0],3), dtype=weights.dtype, device=weights.device)
            orient_weight = (coeff * orient_weight_filtering).unsqueeze(dim=-1).repeat(1,3)


            # coeff = torch.where(weighted_grad_norm > max_allowed_grad, 0, 1.0) 
            
            # perc = torch.quantile(weighted_grad_norm_sum, 0.9)
            # coeff = torch.where(weighted_grad_norm_sum > perc, 0, 1.0)
            # coeff,_ = torch.min(coeff, dim=1)

            # coeff = torch.where(weighted_grad_norm_sum > max_allowed_grad, 0, 1.0) 
            

            # orient_weight = coeff.detach().unsqueeze(dim=-1).repeat(1,3)

            # orient_weight = orient_weight / orient_weight.max()
            # orient_weight[:] = 1.0 # disable orient
            diffuse_color = (weights[:, :, None] * diffuse_color_all_pts).sum(dim=1) / weights[:, :, None].sum(dim=1)
            specular_color = (weights[:, :, None] * specular_color_all_pts).sum(dim=1) / weights[:, :, None].sum(dim=1)
            out["diffuse_color"] = diffuse_color
            out["specular_color"] = specular_color
            # print(weights.shape, diffuse_all_pts_grad_norm.shape)
            out["diffuse_color_grad"] = (weights * diffuse_all_pts_grad_norm).sum(dim=1) / weights.sum(dim=1)
            # out["specular_color_grad"] = weighted_grad_norm.sum(dim=1)
            # out["specular_color_grad"][out["specular_color_grad"] > perc] = 0.0
            specular_color_grad = weighted_grad_norm
            # specular_color_grad[specular_color_grad > max_allowed_grad] = 0
            specular_color_grad = specular_color_grad.sum(dim=1)
            specular_color_grad[specular_color_grad > filter_grad] = 0
            out["specular_color_grad"] = specular_color_grad
            # print("specular_color_grad", out["specular_color_grad"].shape, out["specular_color_grad"].min(), out["specular_color_grad"].max(), out["specular_color_grad"].mean())

        else:
            raise NotImplementedError(self.config.train.protective_loss_type)
        print("gi_rendered", gi_rendered, "true_rgb", true_rgb)
        color_error = (gi_rendered - true_rgb) # * weights[:, :, None].sum(dim=1)[combined_mask].detach() 
        if self.config.loss.loss_type == "relative_orient":
            if self.config.loss.loss_denominator == "both":
                denominator = (torch.clamp((true_rgb.detach() + gi_rendered.detach()), min=0.0)/2).detach()
            elif self.config.loss.loss_denominator == "prediction":
                denominator =  (torch.clamp((true_rgb.detach() + self.config.loss.loss_denominator_val), min=0.0)).detach()
            else:
                raise NotImplementedError(self.config.loss.loss_denominator)
            loss_output_all = (
                    (
                        F.smooth_l1_loss(
                        color_error/denominator, 
                        torch.zeros_like(color_error), reduction='none')
                        # F.smooth_l1_loss(
                        # color_error/((1+torch.clamp(true_rgb[combined_mask].detach(), min=0))/2).detach(), 
                        # torch.zeros_like(color_error), reduction='none')
                    )
                * orient_weight.detach() 
                )
            loss_output = loss_output_all[combined_mask].sum()/ mask_sum
            # print(gi_rendered[combined_mask])
            # if (loss_output_all > 100).any():
            #     print("========mitsuba large loss=============", loss_output_all[loss_output_all>10], color_error[loss_output_all > 10], (1+true_rgb[combined_mask][loss_output_all > 10])/2)
        elif self.config.loss.loss_type == "relative_l1_orient":
            if self.config.loss.loss_denominator == "both":
                denominator = (torch.clamp((true_rgb.detach() + gi_rendered.detach()), min=0.0)/2).detach()
            elif self.config.loss.loss_denominator == "prediction":
                denominator =  (torch.clamp((true_rgb.detach() + self.config.loss.loss_denominator_val), min=0.0)).detach()
            else:
                raise NotImplementedError(self.config.loss.loss_denominator)
            loss_output_all = (
                    (
                        F.l1_loss(
                        color_error/denominator, 
                        torch.zeros_like(color_error), reduction='none')
                    )
                * orient_weight.detach() 
                )
            loss_output = loss_output_all[combined_mask].sum()/ mask_sum
        elif self.config.loss.loss_type == "l1_orient":
            loss_output_all = (
                        F.l1_loss(
                            color_error, 
                            torch.zeros_like(color_error), reduction='none')
                * orient_weight.detach() 
                )
            loss_output = loss_output_all[combined_mask].sum() / mask_sum
        else:
            raise RuntimeError(f"loss_type {self.config.loss.loss_type} not supported")
            
        # print("weight", torch.isfinite(weights).all(), "loss_radiosity", torch.isfinite(loss_radiosity).all(), "loss_output", torch.isfinite(loss_output).all())
        # print("weight", weight.min(),  weight.max(), weight.shape, color_error.shape)
        out["orient_loss_weight"] = orient_weight
        out["loss_output_all"] = loss_output_all
        # loss_output = (
        #     (F.l1_loss(
        #         color_error, torch.zeros_like(color_error), reduction='none'
        #     ) * weight[combined_mask].unsqueeze(dim=-1).detach()).sum()/ mask_sum
        # )

        # print("loss_radiosity", loss_radiosity.item(), "loss_output", loss_output.item())
        # if step < 10000:
        #     loss = loss_output
        # else:
        #     loss = loss_radiosity + loss_output
        # loss = loss_radiosity + loss_output
        loss = loss_output
        losses = {
            "loss_output": loss_output.item(), 
            
        }
        if self.config.loss.use_radiosity:
            if self.config.geometry_type.name.startswith("vol") or self.config.geometry_type.name.startswith("mesh"):
                # loss_radiosity = F.l1_loss(color_all_pts*self.light_conv_factor, gi_rendered_all_pts.detach(), reduction='none')
                # print(color_all_pts.shape, combined_mask.shape, gi_rendered_all_pts.shape)
                # loss_radiosity = (
                #     (
                #         (color_all_pts[combined_mask]*self.light_conv_factor - gi_rendered_all_pts.detach()[combined_mask])
                #         / (torch.clamp((color_all_pts[combined_mask]*self.light_conv_factor + gi_rendered_all_pts[combined_mask] )/2, min=0.001)).detach()
                #         )**2

                # ) 
                if self.config.loss.rhs_radiosity:
                    # loss_radiosity = (
                    #         (LHS_all_pts[combined_mask]*self.light_conv_factor - gi_rendered_all_pts.detach()[combined_mask])
                    #         / (torch.clamp((LHS_all_pts[combined_mask]*self.light_conv_factor)/2, min=0.1)).detach()
                    #         )**2
                    # radiosity_error = (LHS_all_pts[combined_mask]*self.light_conv_factor - gi_rendered_all_pts.detach()[combined_mask]
                    #      / ((1+torch.clamp((LHS_all_pts[combined_mask]*self.light_conv_factor), min=0.0))/2).detach())
                    if self.config.render.use_separate_emitter_bsdf_cache:
                        radiosity_weights = out["subsample_weights"].detach()
                        LHS_all_pts_all_occ = out["LHS_all_pts_all_occ"]
                        gi_rendered_all_pts_no_occ = out["subsample_gi_rendered_emitter_only"]
                        gi_rendered_all_pts_all_occ = out["subsample_gi_rendered_all_pts_all_occ"]
                        if self.config.loss.radiosity_loss_denominator == "both":
                            denominator = torch.clamp((LHS_all_pts*self.light_conv_factor + gi_rendered_all_pts_no_occ).detach(), min=0.001)/2
                        elif self.config.loss.radiosity_loss_denominator == "prediction":
                            denominator = torch.clamp((LHS_all_pts*self.light_conv_factor + self.config.loss.loss_denominator_val).detach(), min=0.001)
                        else:
                            raise NotImplementedError(self.config.loss.radiosity_loss_denominator)
                        
                        radiosity_error = (
                            (LHS_all_pts*self.light_conv_factor - gi_rendered_all_pts_no_occ.detach())
                            / denominator)
                        # print("combined_mask sum", combined_mask.sum())
                        # print("radiosity_error non occ", radiosity_error, radiosity_error.mean(), torch.isfinite(radiosity_error).all())
                        loss_radiosity_no_occ = F.mse_loss(radiosity_error, torch.zeros_like(radiosity_error), reduction='none')
                        loss_radiosity_no_occ = ((loss_radiosity_no_occ[combined_mask] * radiosity_weights[:, :, None].detach()[combined_mask] ).sum(dim=-1) 
                        ).mean()  # / weights[:, :, None].sum(dim=1).detach()[combined_mask]
                        # loss_radiosity_no_occ = ((loss_radiosity_no_occ).sum(dim=-1) ).mean() 

                        if self.config.loss.radiosity_loss_denominator == "both":
                            denominator = torch.clamp((LHS_all_pts_all_occ*self.light_conv_factor + 0.1).detach(), min=0)
                        elif self.config.loss.radiosity_loss_denominator == "prediction":
                            denominator = torch.clamp((LHS_all_pts_all_occ*self.light_conv_factor + self.config.loss.loss_denominator_val).detach(), min=0)
                        else:
                            raise NotImplementedError(self.config.loss.radiosity_loss_denominator)

                        radiosity_error = (
                            (LHS_all_pts_all_occ*self.light_conv_factor - gi_rendered_all_pts_all_occ.detach())
                            / denominator)
                        # print("radiosity_error occ", radiosity_error, radiosity_error.mean(), torch.isfinite(radiosity_error).all())
                        loss_radiosity_all_occ_all = F.mse_loss(radiosity_error, torch.zeros_like(radiosity_error), reduction='none').sum(dim=-1) # sum over rgb
                        loss_radiosity_all_occ = ((loss_radiosity_all_occ_all[combined_mask] * radiosity_weights.detach()[combined_mask] )).mean()   # / weights[:, :, None].sum(dim=1).detach()[combined_mask]
                        # loss_radiosity_all_occ = ((loss_radiosity_all_occ ).sum(dim=-1)).mean()
                        out["loss_radiosity_all_occ_all"] = loss_radiosity_all_occ_all
                        loss_radiosity = loss_radiosity_no_occ + loss_radiosity_all_occ
                        if self.config.render.ambient_light:
                            gi_rendered_all_pts_ambient = out["subsample_gi_rendered_all_pts_ambient"]
                            LHS_all_pts_ambient = out["LHS_all_pts_ambient"]
                            # print("gi_rendered_all_pts_ambient", gi_rendered_all_pts_ambient)
                            # print("LHS_all_pts_ambient", LHS_all_pts_ambient)
                            if self.config.loss.radiosity_loss_denominator == "both":
                                denominator = torch.clamp((LHS_all_pts_ambient*self.light_conv_factor + 0.1).detach(), min=0)
                            elif self.config.loss.radiosity_loss_denominator == "prediction":
                                denominator = torch.clamp((LHS_all_pts_ambient*self.light_conv_factor + self.config.loss.loss_denominator_val).detach(), min=0)
                            else:
                                raise NotImplementedError(self.config.loss.radiosity_loss_denominator)

                            radiosity_error = (
                                (LHS_all_pts_ambient*self.light_conv_factor - gi_rendered_all_pts_ambient.detach())
                                / denominator)
                            
                            # print("radiosity_error occ", radiosity_error, radiosity_error.mean(), torch.isfinite(radiosity_error).all())
                            loss_radiosity_ambient_all = F.mse_loss(radiosity_error, torch.zeros_like(radiosity_error), reduction='none').sum(dim=-1) # sum over rgb
                            loss_radiosity_ambient = ((loss_radiosity_ambient_all[combined_mask] * radiosity_weights.detach()[combined_mask] )).mean()   # / weights[:, :, None].sum(dim=1).detach()[combined_mask]
                            # loss_radiosity_all_occ = ((loss_radiosity_all_occ ).sum(dim=-1)).mean()
                            out["loss_radiosity_ambient_all"] = loss_radiosity_ambient_all
                            # print("loss_radiosity_ambient_all", loss_radiosity_ambient_all)
                            loss_radiosity = loss_radiosity + loss_radiosity_ambient
                            pass
                    else:
                        radiosity_error = (
                            (LHS_all_pts[combined_mask]*self.light_conv_factor - out["subsample_gi_rendered_all_pts"].detach()[combined_mask])
                            / torch.clamp((LHS_all_pts[combined_mask]*self.light_conv_factor + out["subsample_gi_rendered_all_pts"][combined_mask]).detach(), min=0.001)/2)
                        # loss_radiosity = (
                        #     ()
                        #     / (((LHS_all_pts[combined_mask]*self.light_conv_factor+gi_rendered_all_pts.detach()[combined_mask])/2).detach()/2)
                        #     )**2
                        radiosity_weights = out["subsample_weights"].detach()
                        loss_radiosity = F.smooth_l1_loss(radiosity_error, torch.zeros_like(radiosity_error), reduction='none')
                        
                        loss_radiosity = ((loss_radiosity * radiosity_weights[:, :, None].detach()[combined_mask] ).sum(dim=1) 
                        / radiosity_weights[:, :, None].sum(dim=1).detach()[combined_mask]).mean()   #* orient_weight[combined_mask].unsqueeze(dim=-1).detach()
                else:
                    loss_radiosity = (
                        (
                            (color_all_pts[combined_mask]*self.light_conv_factor - gi_rendered_all_pts.detach()[combined_mask])
                            / (torch.clamp((color_all_pts[combined_mask]*self.light_conv_factor  )/2, min=0.001)).detach()
                            )**2

                    ) 
                    # print(loss_radiosity.shape, weights.shape)
                    loss_radiosity = ((loss_radiosity * weights[:, :, None].detach()[combined_mask] ).sum(dim=1) 
                                        / weights[:, :, None].sum(dim=1).detach()[combined_mask]
                                    ).mean()   #
                    # * orient_weight[combined_mask].unsqueeze(dim=-1).detach()
            elif self.config.geometry_type.name == "pts":
                # print("color_rendered", color_rendered, torch.isfinite(color_rendered).all())
                # print("gi_rendered", gi_rendered, torch.isfinite(gi_rendered).all())
                # print("combined_mask", combined_mask, torch.isfinite(combined_mask).all())
                # loss_radiosity = F.l1_loss(color_rendered[combined_mask]*self.light_conv_factor, gi_rendered.detach()[combined_mask], reduction='none')
                # print("loss_radiosity", loss_radiosity, torch.isfinite(loss_radiosity).all())

                loss_radiosity = (
                    (
                        (color_rendered[combined_mask]*self.light_conv_factor - gi_rendered.detach()[combined_mask])
                        / ((color_rendered[combined_mask]*self.light_conv_factor + gi_rendered[combined_mask])/2).detach()
                        )**2

                ) * orient_weight[combined_mask].unsqueeze(dim=-1).detach()
                
                    
                loss_radiosity = loss_radiosity.sum() / mask_sum
            else:
                raise RuntimeError(f"geometry_type {self.config.geometry_type} not supported")
            loss = loss + loss_radiosity
            losses["loss_radiosity"] = loss_radiosity.item()
        # print("pixels_x", pixels_x.shape, "pixels_y", pixels_y.shape)
        sam_mask, width, height = self.dataset.gen_sam_mask_at(idx, pixels_x.squeeze(dim=-1), pixels_y.squeeze(dim=-1))
        # smooth_keys = ["roughness", "eta", "clearcoat", "clearcoat_gloss"]
        smooth_keys = ["roughness", "eta"]
        pixels = torch.cat([pixels_x, pixels_y], dim=-1)
        # print(pixels.shape, torch.tensor([width, height], device=pixels.device).shape)
        pixels = pixels / torch.tensor([width, height], device=pixels.device)
        for k in smooth_keys:
            if k not in out:
                continue
            bilateral_loss = reg_utils.bilateral_sem(pixels, out[k][:, 0], sam_mask, out["albedo"].detach(), combined_mask, 0.3, 1.0)
            losses["smoothness_{}".format(k)] = bilateral_loss
            loss = loss + self.config.train.reg_weight * bilateral_loss
        if self.config.loss.use_radiosity and self.config.train.use_secondary_bounce:
            if self.config.loss.rhs_radiosity:
                if self.config.render.use_separate_emitter_bsdf_cache:
                    sec_LHS_all_pts_all_occ = sec_out["LHS_all_pts_all_occ"]
                    sec_gi_rendered_all_pts_no_occ = sec_out["subsample_gi_rendered_emitter_only"]
                    sec_gi_rendered_all_pts_all_occ = sec_out["subsample_gi_rendered_all_pts_all_occ"]
                    out["sec_subsample_gi_rendered_all_pts_no_occ"] = sec_gi_rendered_all_pts_no_occ
                    sec_weights = sec_out["subsample_weights"].detach()
                    out["sec_weights"] = sec_weights

                    if self.config.loss.radiosity_loss_denominator == "both":
                        denominator = torch.clamp((sec_LHS_all_pts*self.light_conv_factor + sec_gi_rendered_all_pts_no_occ).detach(), min=0.001)
                    elif self.config.loss.radiosity_loss_denominator == "prediction":
                        denominator = torch.clamp((sec_LHS_all_pts*self.light_conv_factor + self.config.loss.loss_denominator_val).detach(), min=0.001)
                    else:
                        raise NotImplementedError(self.config.loss.radiosity_loss_denominator)
                    

                    radiosity_error = (
                        (sec_LHS_all_pts*self.light_conv_factor - sec_gi_rendered_all_pts_no_occ.detach())
                        / denominator)
                    sec_loss_radiosity_no_occ = F.mse_loss(radiosity_error, torch.zeros_like(radiosity_error), reduction='none')
                    sec_loss_radiosity_no_occ = (sec_loss_radiosity_no_occ * sec_weights[:, :, None].detach() )
                    sec_loss_radiosity_no_occ_all = sec_loss_radiosity_no_occ.sum(dim=-1)
                    # print(sec_loss_radiosity_no_occ.shape)
                    # exit()
                    sec_loss_radiosity_no_occ = sec_loss_radiosity_no_occ_all[combined_mask].mean()
                    
                       # / weights[:, :, None].sum(dim=1).detach()[combined_mask]
                    if self.config.loss.radiosity_loss_denominator == "both":
                        denominator = torch.clamp((sec_LHS_all_pts_all_occ*self.light_conv_factor + 0.1).detach(), min=0.001)
                    elif self.config.loss.radiosity_loss_denominator == "prediction":
                        denominator = torch.clamp((sec_LHS_all_pts_all_occ*self.light_conv_factor + self.config.loss.loss_denominator_val).detach(), min=0.001)
                    else:
                        raise NotImplementedError(self.config.loss.radiosity_loss_denominator)
                    # print("WARNING: ========== denominator", denominator)
                    # denominator = 1
                    radiosity_error = (
                        (sec_LHS_all_pts_all_occ*self.light_conv_factor - sec_gi_rendered_all_pts_all_occ.detach())
                        / denominator)
                    sec_loss_radiosity_all_occ = F.mse_loss(radiosity_error, torch.zeros_like(radiosity_error), reduction='none')
                    sec_loss_radiosity_all_occ = (sec_loss_radiosity_all_occ * sec_weights[:, :, None].detach() )
                    sec_loss_radiosity_all_occ_all = sec_loss_radiosity_all_occ.sum(dim=-1)
                    sec_loss_radiosity_all_occ = sec_loss_radiosity_all_occ_all[combined_mask].mean()
                    sec_loss_radiosity = sec_loss_radiosity_no_occ + sec_loss_radiosity_all_occ
                    out["sec_loss_radiosity_no_occ"] = sec_loss_radiosity_no_occ_all
                    out["sec_loss_radiosity_all_occ"] = sec_loss_radiosity_all_occ_all
                    
                    # else:
                    #     raise NotImplementedError()
                    if self.config.render.ambient_light:
                        sec_gi_rendered_all_pts_ambient = sec_out["subsample_gi_rendered_all_pts_ambient"]
                        sec_LHS_all_pts_ambient = sec_out["LHS_all_pts_ambient"]
                        if self.config.loss.radiosity_loss_denominator == "both":
                            denominator = torch.clamp((sec_LHS_all_pts_ambient*self.light_conv_factor + 0.1).detach(), min=0)
                        elif self.config.loss.radiosity_loss_denominator == "prediction":
                            denominator = torch.clamp((sec_LHS_all_pts_ambient*self.light_conv_factor + self.config.loss.loss_denominator_val).detach(), min=0)
                        else:
                            raise NotImplementedError(self.config.loss.radiosity_loss_denominator)

                        radiosity_error = (
                            (sec_LHS_all_pts_ambient*self.light_conv_factor - sec_gi_rendered_all_pts_ambient.detach())
                            / denominator)
                        # radiosity_error_nan_mask = ~(torch.isfinite(radiosity_error).all(dim=-1))
                        # print("sec_LHS_all_pts_ambient", sec_LHS_all_pts_ambient[radiosity_error_nan_mask], "sec_gi_rendered_all_pts_ambient", sec_gi_rendered_all_pts_ambient[radiosity_error_nan_mask], sec_weights[radiosity_error_nan_mask], radiosity_error_nan_mask)
                        # print("radiosity_error occ", radiosity_error, radiosity_error.mean(), torch.isfinite(radiosity_error).all())
                        sec_loss_radiosity_ambient_all = F.mse_loss(radiosity_error, torch.zeros_like(radiosity_error), reduction='none').sum(dim=-1) # sum over rgb
                        sec_loss_radiosity_ambient = ((sec_loss_radiosity_ambient_all[combined_mask] * sec_weights.detach()[combined_mask] )).mean()   # / weights[:, :, None].sum(dim=1).detach()[combined_mask]
                        # loss_radiosity_all_occ = ((loss_radiosity_all_occ ).sum(dim=-1)).mean()
                        out["sec_loss_radiosity_ambient_all"] = sec_loss_radiosity_ambient_all
                        sec_loss_radiosity = sec_loss_radiosity + sec_loss_radiosity_ambient
                        pass
                    loss = loss + sec_loss_radiosity
                    pass
                else:
                    sec_LHS_all_pts = sec_out["LHS_all_pts"]
                    sec_gi_rendered_all_pts = sec_out["subsample_gi_rendered_all_pts"]
                    sec_weights = sec_out["subsample_weights"].detach()
                    sec_radiosity_error = (
                        (sec_LHS_all_pts[combined_mask]*self.light_conv_factor - sec_gi_rendered_all_pts[combined_mask].detach())
                        / torch.clamp((sec_LHS_all_pts*self.light_conv_factor + sec_gi_rendered_all_pts).detach(), min=0.0)[combined_mask]/2)
                    sec_loss_radiosity = F.smooth_l1_loss(sec_radiosity_error, torch.zeros_like(sec_radiosity_error), reduction='none')
                    
                    sec_loss_radiosity = ((sec_loss_radiosity * sec_weights[:, :, None].detach()[combined_mask] ).sum(dim=1) 
                    / sec_weights[:, :, None].sum(dim=1).detach()[combined_mask]).mean()   #* orient_weight[combined_mask].unsqueeze(dim=-1).detach()
                    loss = loss + sec_loss_radiosity
            else:
                raise NotImplementedError

            
        # loss = loss_output
        # loss = loss_output
        # print("loss_radiosity", loss_output, loss_radiosity)
        if False: # we do not use loss visiblity for now
            if "cached_visibility_all_pts" in out and out["cached_visibility_all_pts"] is not None:
                cached_visibility_all_pts = out["cached_visibility_all_pts"]
                loss_visibility_reg = torch.abs(cached_visibility_all_pts).mean()
                losses["loss_visibility_reg"] = loss_visibility_reg
                loss = loss + 0.01 * loss_visibility_reg

            cached_visibility_all_pts_logits = out["cached_visibility_all_pts_logits"]
            if self.config.loss.use_visibility_loss:
                # print(cached_visibility_all_pts_logits.shape, occ_mask.shape)
                loss_visibility = F.binary_cross_entropy_with_logits(cached_visibility_all_pts_logits, occ_mask.reshape(-1))
                losses["loss_visibility"] = loss_visibility.item()

                loss = loss + 0.1 * loss_visibility
                if self.config.train.use_secondary_bounce:
                    sec_cached_visibility_all_pts_logits = sec_out["cached_visibility_all_pts_logits"]
                    sec_occ_mask = sec_out["occ_mask"]
                    loss_sec_visibility = F.binary_cross_entropy_with_logits(sec_cached_visibility_all_pts_logits, sec_occ_mask.reshape(-1))
                    losses["loss_sec_visibility"] = loss_sec_visibility.item()
                    loss = loss + 0.1 * loss_sec_visibility

        if self.config.loss.use_radiosity:
            if self.config.train.use_secondary_bounce:
                losses["sec_loss_radiosity"] = sec_loss_radiosity.item()

            if self.config.render.use_separate_emitter_bsdf_cache:
                losses["loss_radiosity_no_occ"]= loss_radiosity_no_occ
                losses["loss_radiosity_all_occ"]= loss_radiosity_all_occ

            if self.config.train.use_secondary_bounce and self.config.render.use_separate_emitter_bsdf_cache:
                losses["sec_loss_radiosity_no_occ"] = sec_loss_radiosity_no_occ.item()
                losses["sec_loss_radiosity_all_occ"] = sec_loss_radiosity_all_occ.item()

                out["sec_gi_rendered_all_pts_no_occ"] = sec_out["gi_rendered_all_pts_no_occ"]
                out["sec_gi_rendered_all_pts_all_occ"] = sec_out["gi_rendered_all_pts_all_occ"]
                out["sec_subsample_gi_rendered_all_pts_all_occ"] = sec_out["subsample_gi_rendered_all_pts_all_occ"]
                out["sec_gi_rendered_all_occ"] = (sec_out["weights"][:, :, None].detach() * sec_out["gi_rendered_all_pts_all_occ"]).sum(dim=1) / sec_out["weights"][:, :, None].sum(dim=1).detach()
                out["sec_LHS_all_pts"] = sec_out["LHS_all_pts"]
                out["sec_LHS_all_occ"] = sec_out["LHS_all_occ"]
                out["sec_LHS_all_pts_all_occ"] = sec_out["LHS_all_pts_all_occ"]
                # out["sec_subsample_LHS_all_pts_all_occ"] = sec_out["subsample_LHS_all_pts_all_occ"]
                # out["sec_LHS_all_pts_no_occ"] = sec_out["LHS_all_pts_no_occ"]

        out["light_conv_factor"] = self.light_conv_factor.detach()

        if self.config.train.use_secondary_bounce:
            out["sec_gi_rendered"] = sec_out["gi_rendered"]
            out["sec_LHS"] = sec_out["LHS"]


            out["sec_gi_rendered_all_pts"] = sec_out["gi_rendered_all_pts"]
            # out["sec_gi_rendered_emitter_only_all_pts"] = sec_out["gi_rendered_emitter_only_all_pts"]
            out["sec_normals_all_pts"] = sec_out["normals_all_pts"]
            # out["sec_albedo_all_pts"] = sec_out["albedo_all_pts"]
            out["sec_weights"] = sec_out["weights"]
            out["sec_occ_mask_vol"] = sec_out["occ_mask_vol"]
            out["sec_subsample_occ_mask_vol"] = sec_out["occ_mask_vol"]
            if self.config.render.ambient_light:
                out["sec_LHS_ambient"] = sec_out["LHS_ambient"]
                out["sec_gi_rendered_ambient"] = sec_out["gi_rendered_ambient"]
        return loss, losses, out

    def init_per_step(self):
        self.physical_shader_gi.zero_torch_grad()
        if self.optim is not None:
            self.optim.zero_grad()

    def step(self):
        has_nan = False
        for name, param in list(self.physical_shader_gi.renderer.network.named_parameters()):
            if param.grad is not None and (not torch.isfinite(param.grad).all()):
                # print(name, param.grad)
                print("nan gradient found in integrator network {}".format(name))
                has_nan = True
        for name, module in self.physical_shader_gi.learned_info["learned_modules"].items():
            # print("scanned module", name) 
            for param_name, param in module.named_parameters():
                # print(name, param_name, param.grad)
                if param.grad is None:
                    print(param_name, "is none, is this what you want")
                else:
                    if (not torch.isfinite(param.grad).all()):
                        print("nan gradient found in module {}: {}".format(name, param_name))
                        has_nan = True
        if has_nan:
            return
        self.physical_shader_gi.update_mi_params()
        self.physical_shader_gi.update_torch_params()
        # print("grad before step", self.shadow_visibility_cache.mask.grad.max(), self.shadow_visibility_cache.mask.grad.min())
        if self.optim is not None:
            self.optim.step()

    def get_s_grad(self, render_out, rays_o, rays_d, near, far, light_to_world, light_o, light_illum, step, max_steps, true_rgb, idx, pixels_x, pixels_y, combined_mask=None, shadow_mask=None):
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        loss, losses, mitsuba_out = self(render_out, rays_o, rays_d, near, far, light_to_world, light_o, light_illum, step=step, max_steps=max_steps, true_rgb=true_rgb, idx=idx, pixels_x=pixels_x, pixels_y=pixels_y, combined_mask=combined_mask, shadow_mask=shadow_mask)                
        
        # begin 
        dl_dweights = torch.autograd.grad((loss,), (mitsuba_out["weights"]), retain_graph=True)[0] # 512x128
        dl_ds_per_loc = torch.autograd.grad((mitsuba_out["weights"]), (render_out["inv_s"],), grad_outputs=dl_dweights, retain_graph=True)[0]
        
        dl_ds_per_loc = dl_ds_per_loc.reshape(dl_dweights.shape[0], dl_dweights.shape[1], dl_dweights.shape[1])
        pts = mitsuba_out["pts"]
        dl_ds_per_loc = dl_ds_per_loc.sum(dim=-1)
        dl_dsdf = torch.autograd.grad((mitsuba_out["weights"]), (render_out["sdf_grad"],), grad_outputs=dl_dweights, retain_graph=True)[0] # 512x128


        dl_dn_per_loc = torch.autograd.grad((mitsuba_out["weights"]), (render_out["gradients_orig"],), grad_outputs=dl_dweights)[0] # 512 x 128 x 3
        #ends

        # dl_ds_per_loc = torch.autograd.grad((loss,), (render_out["inv_s"]), retain_graph=True)[0]
        # return dl_dweights, mitsuba_out["loss_output_all"], dl_ds_per_loc, mitsuba_out["weights"], pts
        # dl_dweights = torch.autograd.grad((loss,), (mitsuba_out["weights"]), retain_graph=True)[0] # 512x128
        # dl_ds_per_loc = torch.autograd.grad((mitsuba_out["weights"]), (render_out["inv_s"],), grad_outputs=dl_dweights)[0]
        return dl_dweights, mitsuba_out["loss_output_all"], mitsuba_out["valid_mask"], dl_ds_per_loc,mitsuba_out["weights"], pts, dl_dn_per_loc, dl_dsdf
    def validate_image(self, color_outs, out_dir, idx, H, W, render_outs, rays_o, rays_d, dataset, light_to_world, light_o, mitsuba_light_lumen, step, max_steps, light_idx, mask, pixels_x, pixels_y, basic_only=False, repeat=1, shadow_mask=None):
        # rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        # rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        out_mitsuba = []
        true_rgb = []
        # repeat = 1
        seed_counter = 0
        for i in range(repeat):
            for render_out, rays_o_batch, rays_d_batch, color_batch, mask_batch, shadow_mask_batch, pixels_x_batch, pixels_y_batch in zip(render_outs, rays_o, rays_d, color_outs, mask, shadow_mask, pixels_x, pixels_y):
                near, far = dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                render_out = render_utils.render_out_to_cuda(render_out)
                loss, losses, mitsuba_out = self(render_out, rays_o_batch, rays_d_batch, near, far, light_to_world, light_o, mitsuba_light_lumen, step=step + seed_counter, max_steps=max_steps, true_rgb=color_batch, idx=light_idx, pixels_x=pixels_x_batch, pixels_y=pixels_y_batch, combined_mask=mask_batch, shadow_mask=shadow_mask_batch, is_val=True)                
                render_out = render_utils.detach_rec(render_out, to_cpu=True)
                mitsuba_out = render_utils.detach_rec(mitsuba_out, to_cpu=True)
                color_batch = color_batch.detach().cpu()
                out_mitsuba.append(mitsuba_out)
                true_rgb.append(color_batch)
                seed_counter += 1
        suffix = "webp"
        
        color = np.concatenate([o.numpy() for o in true_rgb], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
        cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_gt.{}'.format(step, idx, suffix)), (image_utils.lin2srgb(color)[...,::-1]*256).clip(0, 255))


        img_mitsuba = np.concatenate([o["gi_rendered"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
        # print("at here", img_mitsuba.min(), img_mitsuba.max(), img_mitsuba.mean())
        img_mitsuba_out = ( image_utils.lin2srgb(img_mitsuba) * 256).clip(0, 255)
        # print("============== warning no srgb here")
        # img_mitsuba_out = ( img_mitsuba * 255).clip(0, 255)

        # print(img_mitsuba.min(), img_mitsuba.max(), img_mitsuba.mean())
        # print(img_mitsuba.shape)
        cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}.{}'.format(step, idx, suffix)), img_mitsuba_out[...,::-1])
        orient = np.concatenate([o["orient_loss_weight"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
        # print(os.path.join(out_dir,'{:0>8d}_{}_orient.png'.format(step, idx)))
        # print("WARNING temporarily normalizing orient")
        # orient_max = orient.max()
        # orient = orient / orient.max()
        cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_orient.{}'.format(step, idx, suffix)), (orient*256).clip(0, 255))


        normals = np.concatenate([o["normals"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
        normals_vis = (normals * 0.5 + 0.5) * 256
        
        cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_normals.{}'.format(step, idx, suffix)), normals_vis[...,::-1].clip(0, 255))
        cv2.imwrite(os.path.join(out_dir, '{:0>08d}_{}_normals.exr'.format(step, idx)), normals[...,::-1])


        if not basic_only:
            if len(out_mitsuba) > 0 and "sec_gi_rendered" in out_mitsuba[0].keys():
                img_mitsuba = np.concatenate([o["sec_gi_rendered"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                # print("at here", img_mitsuba.min(), img_mitsuba.max(), img_mitsuba.mean())
                img_mitsuba_out = ( image_utils.lin2srgb(img_mitsuba) * 256).clip(0, 255)
                # print(img_mitsuba.min(), img_mitsuba.max(), img_mitsuba.mean())
                # print(img_mitsuba.shape)
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_sec.{}'.format(step, idx, suffix)), img_mitsuba_out[...,::-1])

            # mitsuba_error = (((img_mitsuba - color)/(color+img_mitsuba)/2)**2).mean(axis=-1)
            mitsuba_error_all = np.concatenate([o["loss_output_all"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0).mean(axis=-1)
            # print("mitsuba_erro", mitsuba_error)
            mitsuba_error_out = cm.jet(mitsuba_error_all)
            mitsuba_error_out = mitsuba_error_out[:, :, :3]
            # print("error_out", mitsuba_error_out)
            cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_error.{}'.format(step, idx, suffix)), (mitsuba_error_out[...,::-1]*256).clip(0, 255))
            cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_error.exr'.format(step, idx)), mitsuba_error_all)
            
            img_hypothesis_folder = os.path.join(out_dir,'{:0>8d}_{}_hypothesis'.format(step, idx))
            os.makedirs(img_hypothesis_folder, exist_ok=True)
            img_hypothesis_all = np.concatenate([o["gi_rendered_all_pts"].numpy() for o in out_mitsuba], axis=0).reshape([H, W, -1, 3])
            for i in range(img_hypothesis_all.shape[2]):
                # hp = random.randint(0, img_hypothesis.shape[2]-1)
                img_hypothesis = ( image_utils.lin2srgb(img_hypothesis_all[:, :, i]) * 256).clip(0, 255)
                # print("============= warning no srgb here")
                # img_hypothesis = ( img_hypothesis_all[:, :, i] * 256).clip(0, 255)
                cv2.imwrite(os.path.join(img_hypothesis_folder,'{:0>8d}_{}_hypothesis_{}.{}'.format(step, idx, i, suffix)), img_hypothesis[...,::-1])
                
            img_hypothesis_folder = os.path.join(out_dir,'{:0>8d}_{}_color_hypothesis'.format(step, idx))
            os.makedirs(img_hypothesis_folder, exist_ok=True)
            if "color_all_pts" in out_mitsuba[0] and out_mitsuba[0]["color_all_pts"] is not None:
                img_hypothesis_all = np.concatenate([o["color_all_pts"].numpy() for o in out_mitsuba], axis=0).reshape([H, W, -1, 3])
                for i in range(img_hypothesis_all.shape[2]):
                    # hp = random.randint(0, img_hypothesis.shape[2]-1)
                    img_hypothesis = ( image_utils.lin2srgb(img_hypothesis_all[:, :, i]) * 256).clip(0, 255)
                    cv2.imwrite(os.path.join(img_hypothesis_folder,'{:0>8d}_{}_color_hypothesis_{}.{}'.format(step, idx, i, suffix)), img_hypothesis[...,::-1])
                

            LHS = np.concatenate([image_utils.lin2srgb(o["LHS"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
            cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_LHS.{}'.format(step, idx, suffix)), (LHS[...,::-1]*256).clip(0, 255))
            if "LHS_all_pts" in  out_mitsuba[0]:
                LHS_all_pts = np.concatenate([image_utils.lin2srgb(o["LHS_all_pts"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1, 3]).mean(axis=0)
                LHS_all_pts_folder = os.path.join(out_dir,'{:0>8d}_{}_LHS'.format(step, idx))
                os.makedirs(LHS_all_pts_folder, exist_ok=True)
                for i in range(LHS_all_pts.shape[2]):
                    a = LHS_all_pts[:, :, i]
                    cv2.imwrite(os.path.join(LHS_all_pts_folder,'{:0>8d}_{}_LHS_{}.{}'.format(step, idx, i, suffix)), (a[:, :, ::-1]*256).clip(0, 255))
         
            if len(out_mitsuba) > 0 and "sec_LHS" in out_mitsuba[0].keys():
                LHS = np.concatenate([image_utils.lin2srgb(o["sec_LHS"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_sec_LHS.{}'.format(step, idx, suffix)), (LHS[...,::-1]*256).clip(0, 255))
            

            albedo = np.concatenate([image_utils.lin2srgb(o["albedo"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
            cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_albedo.{}'.format(step, idx, suffix)), (albedo[...,::-1]*256).clip(0, 255))


            albedo_all_pts = np.concatenate([image_utils.lin2srgb(o["albedo_all_pts"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1, 3]).mean(axis=0)
            albedo_folder = os.path.join(out_dir,'{:0>8d}_{}_albedo'.format(step, idx))
            os.makedirs(albedo_folder, exist_ok=True)
            for i in range(albedo_all_pts.shape[2]):
                a = albedo_all_pts[:, :, i]
                cv2.imwrite(os.path.join(albedo_folder,'{:0>8d}_{}_albedo_{}.{}'.format(step, idx, i, suffix)), (a[:, :, ::-1]*256).clip(0, 255))
            

            # roughness = np.concatenate([image_utils.lin2srgb(o["roughness"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W]).mean(axis=0)
            # cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_roughness.{}'.format(step, idx, suffix)), (roughness*256).clip(0, 255))


            # roughness_folder = os.path.join(out_dir,'{:0>8d}_{}_roughness'.format(step, idx))
            # os.makedirs(roughness_folder, exist_ok=True)
            # roughness_all_pts = np.concatenate([image_utils.lin2srgb(o["roughness_all_pts"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1, 3]).mean(axis=0)
            # for i in range(roughness_all_pts.shape[2]):
            #     r = roughness_all_pts[:, :, i]
            #     cv2.imwrite(os.path.join(roughness_folder,'{:0>8d}_{}_roughness_{}.{}'.format(step, idx, i, suffix)), (r[:, :, 0]*256).clip(0, 255))

            vis_keys = ["roughness", "eta", "clearcoat", "clearcoat_gloss"]
            # vis_keys = ["roughness", "eta"]
            for k in vis_keys:
                if k in out_mitsuba[0]:
                    mat = np.concatenate([image_utils.lin2srgb(o[k].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1]).mean(axis=0)[:, :, 0]
                    cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_{}.{}'.format(step, idx, k, suffix)), (mat*256).clip(0, 255))
                    pass
            


            valid = np.concatenate([o["valid_mask"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W]).mean(axis=0)
            # print(os.path.join(out_dir,'{:0>8d}_{}_orient.png'.format(step, idx)))
            # print(valid.min(), valid.max(), valid.mean())
            cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_valid.{}'.format(step, idx, suffix)), (valid*256).clip(0, 255))
            # valid_clip = (valid*256).clip(0, 255)
            # print("valid_clip", valid_clip.min(), valid_clip.max(), valid_clip.mean())
            # print(os.path.join(out_dir,'{:0>8d}_{}_valid.{}'.format(step, idx, suffix)))
            # valid2=cv2.imread(os.path.join(out_dir,'{:0>8d}_{}_valid.{}'.format(step, idx, suffix)), -1)
            # print("valid2", valid2.min(), valid2.max(), valid2.mean())



            normals_folder = os.path.join(out_dir,'{:0>8d}_{}_normals'.format(step, idx))
            os.makedirs(normals_folder, exist_ok=True)
            normals_all_pts = np.concatenate([o["normals_all_pts"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1, 3]).mean(axis=0)
            # print("normals all pts", normals_all_pts, normals_all_pts.min(), normals_all_pts.max())
            for i in range(normals_all_pts.shape[2]):
                n = normals_all_pts[:, :, i]
                n_vis = (n * 0.5 + 0.5) * 256
                cv2.imwrite(os.path.join(normals_folder,'{:0>8d}_{}_{}_normals.{}'.format(step, idx, i, suffix)), n_vis[...,::-1].clip(0, 255))
                # print("n here", n)
                # cv2.imwrite(os.path.join(out_dir, '{:0>08d}_{}_{}_normals.exr'.format(step, idx, i)), n[...,::-1])

            zs = np.concatenate([o["zs"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W]).mean(axis=0)
            zs_vis = cm.jet(zs)[:, :, :3]
            cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_z.{}'.format(step, idx, suffix)), (zs_vis[...,::-1]*256).clip(0, 255))
            # cv2.imwrite(os.path.join(out_dir, '{:0>08d}_{}_z.exr'.format(step, idx)), zs)


            if "LHS_all_occ" in out_mitsuba[0]:
                LHS_all_occ = np.concatenate([image_utils.lin2srgb(o["LHS_all_occ"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_LHS_all_occ.{}'.format(step, idx, suffix)), (LHS_all_occ[...,::-1]*256).clip(0, 255))
            if "sec_LHS_all_occ" in out_mitsuba[0]:
                LHS_all_occ = np.concatenate([image_utils.lin2srgb(o["sec_LHS_all_occ"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_sec_LHS_all_occ.{}'.format(step, idx, suffix)), (LHS_all_occ[...,::-1]*256).clip(0, 255))


            if "sec_gi_rendered_all_occ" in out_mitsuba[0]:
                sec_gi_rendered_all_occ = np.concatenate([image_utils.lin2srgb(o["sec_gi_rendered_all_occ"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_sec_all_occ.{}'.format(step, idx, suffix)), (sec_gi_rendered_all_occ[...,::-1]*256).clip(0, 255))

            if "gi_rendered_no_occ" in out_mitsuba[0]:
                render_no_occ = np.concatenate([image_utils.lin2srgb(o["gi_rendered_no_occ"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_rendered_no_occ.{}'.format(step, idx, suffix)), (render_no_occ[...,::-1]*256).clip(0, 255))
            if "gi_rendered_emitter_only" in out_mitsuba[0]:
                render_emitter_only = np.concatenate([image_utils.lin2srgb(o["gi_rendered_emitter_only"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_rendered_emitter_only.{}'.format(step, idx, suffix)), (render_emitter_only[...,::-1]*256).clip(0, 255))
            if "gi_rendered_all_occ" in out_mitsuba[0]:
                render_all_occ = np.concatenate([image_utils.lin2srgb(o["gi_rendered_all_occ"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_rendered_all_occ.{}'.format(step, idx, suffix)), (render_all_occ[...,::-1]*256).clip(0, 255))
            
            if "gi_rendered_all_pts_all_occ" in out_mitsuba[0]:
                gi_rendered_all_pts_all_occ = np.concatenate([image_utils.lin2srgb(o["gi_rendered_all_pts_all_occ"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1, 3]).mean(axis=0)
                gi_rendered_all_pts_all_occ_folder = os.path.join(out_dir,'{:0>8d}_{}_all_pts_all_occ'.format(step, idx))
                os.makedirs(gi_rendered_all_pts_all_occ_folder, exist_ok=True)
                for i in range(gi_rendered_all_pts_all_occ.shape[2]):
                    a = gi_rendered_all_pts_all_occ[:, :, i]
                    cv2.imwrite(os.path.join(gi_rendered_all_pts_all_occ_folder,'{:0>8d}_{}_all_occ_{}.{}'.format(step, idx, i, suffix)), (a[:, :, ::-1]*256).clip(0, 255))
                gi_rendered_all_pts_all_occ_folder = os.path.join(out_dir,'{:0>8d}_{}_all_pts_all_occ_weighted'.format(step, idx))

                os.makedirs(gi_rendered_all_pts_all_occ_folder, exist_ok=True)
                weights = np.concatenate([o["weights"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1]).mean(axis=0)
                for i in range(gi_rendered_all_pts_all_occ.shape[2]):
                    a = (gi_rendered_all_pts_all_occ * weights[:, :, :, None] * 10)[:, :, i]
                    cv2.imwrite(os.path.join(gi_rendered_all_pts_all_occ_folder,'{:0>8d}_{}_all_occ_{}.{}'.format(step, idx, i, suffix)), (a[:, :, ::-1]*256).clip(0, 255))

            if "LHS_ambient" in out_mitsuba[0]:
                LHS_ambient = np.concatenate([image_utils.lin2srgb(o["LHS_ambient"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_lhs_ambient.{}'.format(step, idx, suffix)), (LHS_ambient[...,::-1]*256).clip(0, 255))

            if "gi_rendered_ambient" in out_mitsuba[0]:
                gi_rendered_ambient = np.concatenate([image_utils.lin2srgb(o["gi_rendered_ambient"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_gi_rendered_ambient.{}'.format(step, idx, suffix)), (gi_rendered_ambient[...,::-1]*256).clip(0, 255))

            if "sec_LHS_ambient" in out_mitsuba[0]:
                LHS_ambient = np.concatenate([image_utils.lin2srgb(o["sec_LHS_ambient"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_sec_lhs_ambient.{}'.format(step, idx, suffix)), (LHS_ambient[...,::-1]*256).clip(0, 255))

            if "sec_gi_rendered_ambient" in out_mitsuba[0]:
                gi_rendered_ambient = np.concatenate([image_utils.lin2srgb(o["sec_gi_rendered_ambient"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_sec_gi_rendered_ambient.{}'.format(step, idx, suffix)), (gi_rendered_ambient[...,::-1]*256).clip(0, 255))

            if "sec_gi_rendered_all_pts" in  out_mitsuba[0]:
                sec_img_all_pts = np.concatenate([image_utils.lin2srgb(o["sec_gi_rendered_all_pts"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1, 3]).mean(axis=0)
                sec_img_all_pts_folder = os.path.join(out_dir,'{:0>8d}_{}_sec_img'.format(step, idx))
                os.makedirs(sec_img_all_pts_folder, exist_ok=True)
                for i in range(sec_img_all_pts.shape[2]):
                    a = sec_img_all_pts[:, :, i]
                    cv2.imwrite(os.path.join(sec_img_all_pts_folder,'{:0>8d}_{}_sec_img_{}.{}'.format(step, idx, i, suffix)), (a[:, :, ::-1]*256).clip(0, 255))
            
            if "sec_gi_rendered_all_pts_no_occ" in  out_mitsuba[0]:
                sec_img_all_pts = np.concatenate([image_utils.lin2srgb(o["sec_gi_rendered_all_pts"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1, 3]).mean(axis=0)
                sec_img_all_pts_folder = os.path.join(out_dir,'{:0>8d}_{}_sec_img'.format(step, idx))
                os.makedirs(sec_img_all_pts_folder, exist_ok=True)
                for i in range(sec_img_all_pts.shape[2]):
                    a = sec_img_all_pts[:, :, i]
                    cv2.imwrite(os.path.join(sec_img_all_pts_folder,'{:0>8d}_{}_sec_img_{}.{}'.format(step, idx, i, suffix)), (a[:, :, ::-1]*256).clip(0, 255))
            
            if "sec_subsample_gi_rendered_all_pts_no_occ" in  out_mitsuba[0]:
                sec_img_all_pts = np.concatenate([image_utils.lin2srgb(o["sec_subsample_gi_rendered_all_pts_no_occ"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1, 3]).mean(axis=0)
                sec_img_all_pts_folder = os.path.join(out_dir,'{:0>8d}_{}_sec_subsample_gi_rendered_all_pts_no_occ'.format(step, idx))
                os.makedirs(sec_img_all_pts_folder, exist_ok=True)
                for i in range(sec_img_all_pts.shape[2]):
                    a = sec_img_all_pts[:, :, i]
                    cv2.imwrite(os.path.join(sec_img_all_pts_folder,'{:0>8d}_{}_sec_subsample_gi_rendered_all_pts_no_occ_{}.{}'.format(step, idx, i, suffix)), (a[:, :, ::-1]*256).clip(0, 255))
            if "sec_gi_rendered_all_pts_all_occ" in  out_mitsuba[0]:
                sec_img_all_pts = np.concatenate([image_utils.lin2srgb(o["sec_gi_rendered_all_pts_all_occ"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1, 3]).mean(axis=0)
                sec_img_all_pts_folder = os.path.join(out_dir,'{:0>8d}_{}_sec_img_all_occ'.format(step, idx))
                os.makedirs(sec_img_all_pts_folder, exist_ok=True)
                for i in range(sec_img_all_pts.shape[2]):
                    a = sec_img_all_pts[:, :, i]
                    cv2.imwrite(os.path.join(sec_img_all_pts_folder,'{:0>8d}_{}_sec_img_all_occ_{}.{}'.format(step, idx, i, suffix)), (a[:, :, ::-1]*256).clip(0, 255))
            if "sec_LHS_all_pts" in  out_mitsuba[0]:
                sec_LHS_all_pts = np.concatenate([image_utils.lin2srgb(o["sec_LHS_all_pts"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1, 3]).mean(axis=0)
                sec_LHS_all_pts_folder = os.path.join(out_dir,'{:0>8d}_{}_sec_LHS'.format(step, idx))
                os.makedirs(sec_LHS_all_pts_folder, exist_ok=True)
                for i in range(sec_LHS_all_pts.shape[2]):
                    a = sec_LHS_all_pts[:, :, i]
                    cv2.imwrite(os.path.join(sec_LHS_all_pts_folder,'{:0>8d}_{}_sec_LHS_{}.{}'.format(step, idx, i, suffix)), (a[:, :, ::-1]*256).clip(0, 255))
            if "sec_albedo_all_pts" in  out_mitsuba[0]:
                sec_albedo_all_pts = np.concatenate([image_utils.lin2srgb(o["sec_albedo_all_pts"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1, 3]).mean(axis=0)
                sec_albedo_all_pts_folder = os.path.join(out_dir,'{:0>8d}_{}_sec_albedo'.format(step, idx))
                os.makedirs(sec_albedo_all_pts_folder, exist_ok=True)
                for i in range(sec_albedo_all_pts.shape[2]):
                    a = sec_albedo_all_pts[:, :, i]
                    cv2.imwrite(os.path.join(sec_albedo_all_pts_folder,'{:0>8d}_{}_sec_albedo_{}.{}'.format(step, idx, i, suffix)), (a[:, :, ::-1]*256).clip(0, 255))
            
            if "sec_normals_all_pts" in  out_mitsuba[0]:
                sec_albedo_all_pts = np.concatenate([image_utils.lin2srgb(o["sec_normals_all_pts"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1, 3]).mean(axis=0)
                sec_albedo_all_pts_folder = os.path.join(out_dir,'{:0>8d}_{}_sec_normals'.format(step, idx))
                os.makedirs(sec_albedo_all_pts_folder, exist_ok=True)
                for i in range(sec_albedo_all_pts.shape[2]):
                    a = sec_albedo_all_pts[:, :, i]
                    cv2.imwrite(os.path.join(sec_albedo_all_pts_folder,'{:0>8d}_{}_sec_normals_{}.{}'.format(step, idx, i, suffix)), (a[:, :, ::-1]*256).clip(0, 255))

            if len(out_mitsuba) > 0 and "sec_weights" in out_mitsuba[0].keys():
                weights = np.concatenate([o["sec_weights"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1]).mean(axis=0)
                # weights_vis = cm.jet(zs)[:, :, :3]

                weights_folder = os.path.join(out_dir, '{:0>8d}_{}_sec_weights'.format(step, idx))
                os.makedirs(weights_folder, exist_ok=True)
                for i in range(weights.shape[2]):
                    a = weights[:, :, i]
                    a_vis = cm.jet(a)[:, :, :3]
                    cv2.imwrite(os.path.join(weights_folder,'{:0>8d}_{}_{}.{}'.format(step, idx, i, suffix)), (a_vis[:, :, ::-1]*256).clip(0, 255))


            weights = np.concatenate([o["weights"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1]).mean(axis=0)
            weights_folder = os.path.join(out_dir, '{:0>8d}_{}_weights'.format(step, idx))
            os.makedirs(weights_folder, exist_ok=True)
            for i in range(weights.shape[2]):
                a = weights[:, :, i]
                a_vis = cm.jet(a)[:, :, :3]
                cv2.imwrite(os.path.join(weights_folder,'{:0>8d}_{}_{}.{}'.format(step, idx, i, suffix)), (a_vis[:, :, ::-1]*256).clip(0, 255))

            if "diffuse_color" in out_mitsuba[0]:
                # print("WARNING hijacking diffuse for visualization")
                diffuse_color = np.concatenate([image_utils.lin2srgb(o["diffuse_color"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                # diffuse_color = np.concatenate([o["diffuse_color"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                # diffuse_color_max = diffuse_color.max()
                # diffuse_color = diffuse_color / diffuse_color_max
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_diffuse_color.{}'.format(step, idx, suffix)), (diffuse_color[...,::-1]*256).clip(0, 255))

            if "specular_color" in out_mitsuba[0]:
                specular_color = np.concatenate([image_utils.lin2srgb(o["specular_color"].numpy()) for o in out_mitsuba], axis=0).reshape([repeat, H, W, 3]).mean(axis=0)
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_specular_color.{}'.format(step, idx, suffix)), (specular_color[...,::-1]*256).clip(0, 255))

            if "diffuse_color_grad" in out_mitsuba[0]:
                diffuse_color = np.concatenate([o["diffuse_color_grad"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W]).mean(axis=0)
                diffuse_color_max = diffuse_color.max()
                diffuse_color = diffuse_color / diffuse_color_max
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_{}_diffuse_color_grad.{}'.format(step, idx, diffuse_color_max, suffix)), (diffuse_color*256).clip(0, 255))

            if "specular_color_grad" in out_mitsuba[0]:
                specular_color = np.concatenate([o["specular_color_grad"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W]).mean(axis=0)
                specular_color_max = specular_color.max()
                specular_color = specular_color / specular_color_max
                cv2.imwrite(os.path.join(out_dir,'{:0>8d}_{}_{}_specular_color_grad.{}'.format(step, idx, specular_color_max, suffix)), (specular_color*256).clip(0, 255))
            
            if "cached_visibility_all_pts" in out_mitsuba[0] and out_mitsuba[0]["cached_visibility_all_pts"]  is not None:
                cached_visibility_all_pts = np.concatenate([o["cached_visibility_all_pts"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1]).mean(axis=0)
                cached_visibility_all_pts_folder = os.path.join(out_dir,'{:0>8d}_{}_cached_visibility'.format(step, idx))
                os.makedirs(cached_visibility_all_pts_folder, exist_ok=True)
                for i in range(cached_visibility_all_pts.shape[2]):
                    a = cached_visibility_all_pts[:, :, i]
                    cv2.imwrite(os.path.join(cached_visibility_all_pts_folder,'{:0>8d}_{}_cached_visibility_{}.{}'.format(step, idx, i, suffix)), (a*256).clip(0, 255))
            
            if "occ_mask" in out_mitsuba[0] and out_mitsuba[0]["occ_mask"]:
                occ_mask_all_pts = np.concatenate([o["occ_mask"].numpy() for o in out_mitsuba], axis=0).reshape([repeat, H, W, -1]).mean(axis=0)
                occ_maskall_pts_folder = os.path.join(out_dir,'{:0>8d}_{}_occ_mask'.format(step, idx))
                os.makedirs(occ_maskall_pts_folder, exist_ok=True)
                for i in range(occ_mask_all_pts.shape[2]):
                    a = occ_mask_all_pts[:, :, i]
                    cv2.imwrite(os.path.join(occ_maskall_pts_folder,'{:0>8d}_{}_occ_mask_{}.{}'.format(step, idx, i, suffix)), (a*256).clip(0, 255))
    def restore_trainer_modules(self, ckpt_path):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "train_modules" not in ckpt:
                print("====warning train module missing")
            else:
                trainer_modules = ckpt["train_modules"] # unfortunate naming
                # print("WARNING SKIP LOADING temporarily")
                if self.shadow_visibility_cache is not None:
                    print("loading visibility cache", ckpt_path)
                    self.shadow_visibility_cache.load_state_dict(trainer_modules["shadow_visibility_cache"])
                if self.physical_shader_gi.renderer.env_map is not None:
                    print("loading envmap", ckpt_path)
                    if "env_map" not in trainer_modules:
                        print("warning env map not loaded")
                    else:
                        self.physical_shader_gi.renderer.env_map.load_state_dict(trainer_modules["env_map"])
    def save_checkpoint(self, step, out_root, is_latest):
        out_root = pathlib.Path(out_root)
        out_dir = out_root / "checkpoints"
        out_dir.mkdir(parents=True, exist_ok=True)
        if is_latest:
            file = f"latest.ckpt"
        else:
            file = f"{step}.ckpt"
        
        trainer_modules = {}
        if self.shadow_visibility_cache is not None:
            trainer_modules["shadow_visibility_cache"] = self.shadow_visibility_cache.state_dict()
        if self.physical_shader_gi.renderer.env_map is not None:
            print("saving envmap")
            trainer_modules["env_map"] = self.physical_shader_gi.renderer.env_map.state_dict()
        ckpt = {
            "step": step,
            "train_modules": trainer_modules
        }
        torch_optim = self.learned_info["torch_optim"]
        modules = self.learned_info["learned_modules"]
        # scheduler = self.learned_info["mi_scheduler"]
        scheduler = None
        mi_optim = self.learned_info["mi_optim"]
        mi_params = self.learned_info["mi_optimized_params"]
        if torch_optim is not None:
            ckpt.update({
                "optim": torch_optim.state_dict(),
                "modules": {
                    k: v.state_dict() for k, v in modules.items()
                }
            })

        if scheduler is not None:
            ckpt.update({
                "scheduler": scheduler.state_dict(),
            })

        if mi_optim is not None:
            ckpt.update({
                "mi_optim": {k: [v.torch() for v in s] for k, s in mi_optim.state.items()},
                "mi_params": {
                    k: v.torch() for k, v in mi_params.items()
                },
            })
        logger.info(f"Save checkpoint: {out_dir / file}")
        torch.save(ckpt, out_dir / file)
