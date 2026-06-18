from typing import Any, Callable, Optional, Tuple

import drjit as dr
import mitsuba as mi

from nerad.integrator import register_integrator


@register_integrator("regularization")
class RegularizationIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.texture = None
        pass

    def post_init(self, texture):
        self.texture = texture

    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               medium: mi.Medium,
               active: mi.Bool):
        bsdf_ctx = mi.BSDFContext()

        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)
        active = mi.Bool(active)                      # Active SIMD lanes

        si = scene.ray_intersect(ray,
                                 ray_flags=mi.RayFlags.All,
                                 coherent=dr.eq(depth, 0))

        val, grad = self.texture.eval_mean_grad_3d(si, active)
        bsdf = si.bsdf(ray)
        reflect = bsdf.eval_diffuse_reflectance(si)
        return (val, si.is_valid(), [val.x, val.y, val.z, grad.x, grad.y, grad.z])
    def aov_names(self):
        return ["roughness.x", "roughness.y", "roughness.z", "grad.x", "grad.y", "grad.z"]
    def to_string(self):
        return (
            "RegularizationIntegrator[\n"
            f"  texture={self.texture}\n"
            "]"
        )
