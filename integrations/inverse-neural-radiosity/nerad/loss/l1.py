import drjit as dr

from nerad.loss import LossFucntion, loss_registry


@loss_registry.register("l1")
class L1Loss(LossFucntion):
    def compute_loss(self, img, gt, mask=None):
        loss =  dr.abs(img - gt)
        if mask is not None:
            if len(mask.shape) == 2:
                loss[:, :, 0] = dr.select(mask, loss[:, :, 0], 0.0).array
                loss[:, :, 1] = dr.select(mask, loss[:, :, 1], 0.0).array
                loss[:, :, 2] = dr.select(mask, loss[:, :, 2], 0.0).array
                loss[:, :, 3] = dr.select(mask, loss[:, :, 3], 0.0).array
                pass
            else:
                raise NotImplementedError(mask.shape)
        return loss
