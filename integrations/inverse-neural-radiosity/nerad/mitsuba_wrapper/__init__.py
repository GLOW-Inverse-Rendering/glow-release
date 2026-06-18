import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from mytorch.registry import Registry, import_children
from mytorch.utils.profiling_utils import counter_profiler, time_profiler
from nerad.utils.mitsuba_utils import vec_to_tens_safe


class MitsubaWrapper(nn.Module):
    def __init__(self, scene_min: float, scene_max: float, name: str = None):
        super().__init__()
        self.grad_activator = mi.Vector3f(0)
        self.scene_min = scene_min
        self.scene_max = scene_max
        self.name = name or type(self).__name__
        self.scale_input = None
    def eval(self, pts, dirs=None, norms=None, albedo=None, pts2=None, dirs2=None, em_cond=None, active=True):
        if counter_profiler.enabled:
            counter_profiler.record(f"{self.name}.eval.pts", dr.shape(pts)[1])
        time_profiler.start(f"{self.name}.eval")
        #Normalize locations
        pts = self.normalize(pts)
        #if position of point light is passed
        pts2 = self.normalize(pts2)

        result = self._eval(pts, dirs, norms, albedo, pts2, dirs2, em_cond, active)
        time_profiler.end(f"{self.name}.eval")
        # print("scene_min & scene_max", self.scene_min, self.scene_max)
        return result

    def eval_and_grad(self, pts, active=True):
        pts = self.normalize(pts)
        result = self._eval_and_grad(pts, active)
        return result

    def _eval_and_grad(self, pts, active=True):
        raise NotImplementedError()

    def set_input_scale(self, scale):
        self.scale_input = scale


    def normalize(self, pts):
        if pts is None:
            return None
        # print( (pts - self.scene_min), (self.scene_max - self.scene_min), self.scene_min, self.scene_max)
        # print("scene min max", pts, self.scene_min, self.scene_max)
        if self.scale_input is not None:
            pts = pts * self.scale_input
        orig_result = (pts - self.scene_min) / (self.scene_max - self.scene_min)
        return orig_result

    def traverse(self, callback):
        callback.put_parameter("grad_activator", self.grad_activator, mi.ParamFlags.Differentiable)
        self._traverse(callback)

    def _eval(self, pts, dirs, norms, albedo, pts2):
        raise NotImplementedError()

    def _traverse(self, callback):
        pass


class MitsubaTensorWrapper(MitsubaWrapper):
    def __init__(
        self,
        scene_min: float,
        scene_max: float,
        grid_size: int,
        value: float = 0.5,
    ):
        super().__init__(scene_min, scene_max)
        self.tensor = nn.parameter.Parameter(
            torch.ones(1, 3, grid_size, grid_size, grid_size) * value,
            requires_grad=True,
        )

    def _eval(self, pts, dirs, norms, albedo, pts2):
        pts = 2 * pts - 1
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        torch_out = self.eval_torch(p_tensor)
        output = dr.unravel(mi.Vector3f, torch_out.array)
        return self._post_process(output)

    def _post_process(self, output):
        raise NotImplementedError()

    @dr.wrap_ad(source="drjit", target="torch")
    def eval_torch(self, pts):
        return torch.nn.functional.grid_sample(
            self.tensor,
            pts[None, None, None],
            align_corners=False,
            padding_mode="border",
        ).view(3, -1).transpose(0, 1)

    def _traverse(self, callback):
        callback.put_parameter("tensor", self.tensor, mi.ParamFlags.Differentiable)


class MitsubaTensorWrapper2D(MitsubaWrapper):
    def __init__(
        self,
        scene_min: float,
        scene_max: float,
        width: int,
        height: int,
        value: float = 0.5,
    ):
        super().__init__(scene_min, scene_max)
        self.tensor = nn.parameter.Parameter(
            torch.ones(1, 3, height, width) * value,
            requires_grad=True,
        )

    def _eval(self, pts, dirs, norms, albedo, pts2):
        pts = 2 * pts - 1
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        torch_out = self.eval_torch(p_tensor)
        output = dr.unravel(mi.Vector3f, torch_out.array)
        return self._post_process(output)

    def _post_process(self, output):
        raise NotImplementedError()

    @dr.wrap_ad(source="drjit", target="torch")
    def eval_torch(self, pts):
        return torch.nn.functional.grid_sample(
            self.tensor,
            pts[None, None],
            align_corners=False,
            padding_mode="border",
        ).view(3, -1).transpose(0, 1)

    def _traverse(self, callback):
        callback.put_parameter("tensor", self.tensor, mi.ParamFlags.Differentiable)


class MitsubaTextureWrapper(MitsubaWrapper):
    def __init__(
        self,
        scene_min: float,
        scene_max: float,
        grid_size: int,
        device: str,
        value: float = 0.5,
    ):
        super().__init__(scene_min, scene_max)
        value = torch.ones(grid_size, grid_size, grid_size, 3, device=device) * value
        self.texture = mi.Texture3f(mi.TensorXf(value), use_accel=False)

    def _eval(self, pts, dirs, norms, albedo, pts2):
        output = mi.Vector3f(self.texture.eval(pts))
        return self._post_process(output)

    def _post_process(self, output):
        raise NotImplementedError()

    def _traverse(self, callback):
        callback.put_parameter("mi_texture", self.texture.tensor(), mi.ParamFlags.Differentiable)


wrapper_registry = Registry("wrapper", MitsubaWrapper)
import_children(__file__, __name__)
