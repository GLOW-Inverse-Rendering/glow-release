from typing import Any

import mitsuba as mi
import torch.nn as nn

from nerad.mitsuba_wrapper import wrapper_registry
from nerad.emitter import register_emitter


@register_emitter("offsetlight")
class OffsetLight(mi.Emitter):
    def __init__(self, props: mi.Properties) -> None:
        self.wrappped_light = props["wrapped_light"]
        mi.Emitter.__init__(self, props)

    def traverse(self, callback):
        self.wrapped_light.traverse(callback)

    def eval(self, si: mi.SurfaceInteraction3f, active: bool = True) -> mi.Color3f:
        return self.wrapped_light.eval(si, active=active)

    def sample_direction(self, it, sample: mi.Point2f, active: bool = True):
        return self.wrapped_light.sample_direction(it, sample, active)
    
    def pdf_direction(self, it: mi.Interaction3f, ds: mi.DirectionSample3f, active: bool = True) -> float:
        return self.wrapped_light.pdf_direction(it, ds, active)

    def eval_direction(self, it, ds, active: bool = True) -> mi.Color3f:
        # TODO: implement this
        raise NotImplementedError()

    def sample_ray(self, time: float, sample1: float, sample2: mi.Point2f, sample3: mi.Point2f, active: bool = True) -> tuple[mi.Ray3f, mi.Color3f]:
        # We probably don't need this
        raise NotImplementedError()

    def bbox(self) -> mi.BoundingBox3f:
        return mi.BoundingBox3f()

    def to_string(self):
        return (
            "MyEnvmap[\n"
            f"    {self.network}\n"
            "]"
        )
