import torch
import torch.nn
import torch.nn.functional as F
class AdaptiveL2LogLinLoss(torch.nn.Module):
    def __init__(self, beta=0.99):
        super().__init__()
        # self.mean_loss = torch.nn.Parameter()
        self.register_buffer("mean_loss", torch.zeros(1))
        # self.mean_loss.requires_grad_(False)
        self.beta = beta
        pass
    def update_mean(self, color_rel):
        with torch.no_grad():
            color_rel = torch.abs(color_rel)
            self.mean_loss.copy_(self.beta*self.mean_loss.detach() + (1-self.beta)*torch.median(color_rel))
            # self.mean_loss.copy_(self.beta*self.mean_loss.detach() + (1-self.beta)*torch.quantile(color_rel, 0.9))
            # print("90 quantile")
            pass
        
    def forward(self, color_fine, color_error, update_mean=True):
        # self.mean_loss.requires_grad_(False)
        color_rel = color_error/(torch.clamp(color_fine.detach(), min=0.01))
        # print('rgrad', self.mean_loss.requires_grad)
        
        
        mse_loss = F.mse_loss(color_rel, torch.zeros_like(color_error), reduction='none')
        l1_loss = F.l1_loss(color_rel, torch.zeros_like(color_error), reduction='none')

        loss = torch.where(torch.abs(color_rel) < self.mean_loss[0].detach(), mse_loss, (2*self.mean_loss[0].detach()*l1_loss)-(self.mean_loss[0].detach()*self.mean_loss[0].detach()))
        if update_mean:
            self.update_mean(color_rel)

        return loss
