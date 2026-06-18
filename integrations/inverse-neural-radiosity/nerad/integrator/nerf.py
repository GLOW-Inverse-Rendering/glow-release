import mitsuba as mi
import drjit as dr
import torch.nn as nn

from nerad.integrator import register_integrator
from nerad.texture.dictionary import MiDictionary
from nerad.mitsuba_wrapper import wrapper_registry


@register_integrator("nerf")
class NerfIntegrator(mi.SamplingIntegrator, nn.Module):
    def __init__(self, props: mi.Properties):
        nn.Module.__init__(self)
        mi.SamplingIntegrator.__init__(self, props)
        props.get("config")
        self.nerf_model = None
        self.bbox = mi.ScalarBoundingBox3f([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])

    def post_init(
        self,
        function: str,
        kwargs: MiDictionary,
    ):
        self.nerf_model = wrapper_registry.build(function, kwargs)


    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               medium: mi.Medium,
               active: mi.Bool):


        hit, mint, maxt = self.bbox.ray_intersect(ray)
        rgb, rgb0 = self.nerf_model.eval(ray.o, ray.d, mi.TensorXf(mint, shape=[len(mint),1]), mi.TensorXf(maxt, shape=[len(maxt),1]))
        #rgb = dr.select(hit, rgb, mi.Vector3f(0))
        #rgb0 = dr.select(hit, rgb0, mi.Vector3f(0))

        return rgb, True, [rgb0.x, rgb0.y, rgb0.z]

    def aov_names(self):
        return ['rgb0.x', 'rgb0.y', 'rgb0.z']

    def to_string(self):
        return (
            "NerfIntegrator[\n"
            "]"
        )

    def traverse(self, callback):
        self.nerf_model.traverse(callback)
