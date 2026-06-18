from typing import Any

import mitsuba as mi
import torch.nn as nn

from nerad.mitsuba_wrapper import wrapper_registry
from nerad.texture import register_texture

import drjit as dr

@register_texture("mytexture")
class MyTexture(mi.Texture, nn.Module):
    def __init__(self, props: mi.Properties) -> None:
        mi.Texture.__init__(self, props)
        nn.Module.__init__(self)
        self.network = None
    def post_init(
        self,
        function: str,
        kwargs: dict[str, Any],
    ):
        self.network = wrapper_registry.build(function, kwargs)

    def traverse(self, callback):
        if self.network is not None:
            self.network.traverse(callback)
        callback.put_parameter("texture", self, mi.ParamFlags.NonDifferentiable)

    def eval(self, si, active=True):
        # print(dr.grad_enabled(si)) # si grad is enabled here
        result = self.network.eval(si.p, active=active)
        return result

    def eval_1(self, si, active=True):
        return mi.Float(self.eval(si, active=active)[0])

    def eval_1_grad(self, *args, **kwargs):
        raise NotImplementedError()

    def eval_mean_grad_3d(self, si, active=True):
        val, grad = self.network.eval_and_grad(si.p, active=active)
        return val, grad

    def eval_3(self, *args, **kwargs):
        raise NotImplementedError()

    def mean(self, *args, **kwargs):
        raise NotImplementedError()

    def to_string(self):
        return (
            "MyTexture[\n"
            f"  network={self.network}\n"
            "]"
        )
