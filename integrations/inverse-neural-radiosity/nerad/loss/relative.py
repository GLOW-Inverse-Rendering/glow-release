import drjit as dr

from nerad.loss import LossFucntion, loss_registry
import numpy as np

class RelativeLoss(LossFucntion):
    def __init__(self, n_steps: int) -> None:
        super().__init__()
        self.step = -1
        self.n_steps = n_steps
        self.exponent = 0

    def normalize_loss(self, denominator, numerator, annealing):
        normalizer = (dr.detach(denominator)+0.01)
        if (annealing):
            normalizer = normalizer**(self.exponent)
        # print("numerator", numerator, dr.max(dr.max(numerator)), dr.min(dr.min(numerator)))
        # print("normalizer", normalizer, dr.max(dr.max(normalizer)), dr.min(dr.min(normalizer)))
        normalized = (numerator/normalizer)
        # print("normalized", normalized, dr.max(dr.max(normalized)), dr.min(dr.min(normalized)))
        return normalized

    def update_state(self, step: int):
        self.step = step
        self.exponent = max(0, 1 - self.step/(self.n_steps*0.8))

    def denominator(self, img, gt):
        raise NotImplementedError()

class L2RelativeLoss(RelativeLoss):
    def __init__(self, n_steps: int, annealing: bool) -> None:
        super().__init__(n_steps)
        self.annealing = annealing

    def compute_loss(self, img, gt, mask=None):
        if self.annealing:
            assert self.step >= 0
        non_relative_loss = dr.sqr(img - gt)
        denom = dr.detach(self.denominator(img, gt)) # technically rel loss requires detach denominator
        rel_loss = self.normalize_loss(denom, non_relative_loss, self.annealing)
        # print("rel_loss before", rel_loss.numpy())
        if mask is not None:
            # print("==mask used!")
            # print(mask)
            # print("rel_loss before", np.array(rel_loss))
            # print(len(mask.shape))
            if len(mask.shape) == 2:
                rel_loss[:, :, 0] = dr.select(mask, rel_loss[:, :, 0], 0.0).array
                rel_loss[:, :, 1] = dr.select(mask, rel_loss[:, :, 1], 0.0).array
                rel_loss[:, :, 2] = dr.select(mask, rel_loss[:, :, 2], 0.0).array
                rel_loss[:, :, 3] = dr.select(mask, rel_loss[:, :, 3], 0.0).array
                pass
            else:
                raise NotImplementedError(mask.shape)
            # print("rel_loss after", np.array(rel_loss))
            # rel_loss = dr.select(mask, rel_loss, 0.0)
        # print("rel_loss", rel_loss.numpy())
        return rel_loss


@loss_registry.register("l2_relative_gt")
class L2RelativeGt(L2RelativeLoss):
    def denominator(self, img, gt):
        return dr.sqr(gt)


@loss_registry.register("l2_relative_prediction")
class L2RelativePrediction(L2RelativeLoss):
    def denominator(self, img, gt):
        return dr.sqr(img)

@loss_registry.register("l2_relative_both")
class L2RelativeBoth(L2RelativeLoss):
    def denominator(self, img, gt):
        return dr.sqr(0.5*(gt+img))
