import os
import torch
import torch.nn as nn
import torch.nn.functional as f

from train.utils import check_loss, get_cls_from_pkg, dummy_context
from monai.networks.nets.dynunet import DynUNet


class CrossEntropyLoss(nn.Module):
    def forward(self, x0_pred, x0, **kwargs):
        loss = f.cross_entropy(x0_pred, x0.argmax(1), reduction="none")
        check_loss(loss)
        loss = loss.sum()
        return loss


class DiffusionKLLoss(nn.Module):
    def __init__(self, attn_weight, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.attn_weight = attn_weight
    
    def forward(self, xt, x0, x0_pred, t, **kwargs):
        prob_xtm1_given_xt_x0pred = self.diffusion_model.theta_post(xt, x0, t)
        prob_xtm1_given_xt_x0 = self.diffusion_model.theta_post(xt, x0_pred, t)
        
        loss = nn.functional.kl_div(
            torch.log(torch.clamp(prob_xtm1_given_xt_x0pred, min=1e-12)),
            prob_xtm1_given_xt_x0,
            reduction='none'
        )
        loss = loss.sum(dim=1) * self.attn_weight[x0_pred.argmax(1)]
        check_loss(loss)
        loss = loss.sum()
        return loss
    
    
class LPIPS(nn.Module):
    backbones = {"lpips": {"target": "monai.networks.nets.dynunet.DynUNet",
                           "params": {"filters": (16, 32, 64, 128, 256),
                                      "strides": (2, 2, 2, 2, 2),
                                      "kernel_size": (3, 3, 3, 3, 3),
                                      "upsample_kernel_size": (2, 2, 2, 2)},
                           "lpips": {"n_layers": (16, 32, 64, 128, 256)}}
                }
    def __init__(self, 
                 in_channels,
                 out_channels,
                 ndim=3,
                 model_path="", 
                 net_backbone=None,
                 is_training=False,
                 use_linear=True,
                 spatial_average=False):
        super().__init__()
        self.ndim = ndim
        self.model_path = model_path
        self.use_linear = use_linear
        self.is_training = is_training
        self.spatial_average = spatial_average
        
        net_backbone = self.backbones.get(net_backbone, self.backbones["lpips"])
        self.perceptual_net = get_cls_from_pkg(net_backbone,
                                               in_channels=in_channels, out_channels=out_channels, spatial_dims=ndim,
                                               deep_supervision=True, deep_supr_num=len(net_backbone["lpips"]["n_layers"]) - 2)
        if os.path.exists(model_path) and os.path.isfile(model_path):
            self.perceptual_net.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        
        if not self.is_training: self.eval()
        if self.use_linear: self.linear_layers = self.get_lin_layer(net_backbone["lpips"]["n_layers"])
    
    def get_lin_layer(self, in_channel, out_channel=1, dropout=0.):
        n = len(in_channel) - 1
        linear_layers = nn.ModuleList()
        in_channel = (1,) + in_channel[1:-1]
        conv_nd = nn.Conv3d if self.ndim == 3 else nn.Conv2d if self.ndim == 2 else nn.Conv1d
        dropout_nd = nn.Dropout3d if self.ndim == 3 else nn.Dropout2d if self.ndim == 2 else nn.Dropout1d
        for i in range(n):
            layer = nn.Module()
            layer.dropout = dropout_nd(dropout) if dropout > 0 else nn.Identity()
            layer.conv = conv_nd(in_channel[i], out_channel, 1)
            linear_layers.append(layer)
        return linear_layers
    
    @staticmethod
    def tensor_normalize(tensor):
        return tensor / ((tensor ** 2).sum(dim=1, keepdim=True) + 1e-8)

    def tensor_average(self, tensor, size=None):
        b, c, *shp = tensor.shape
        if not self.spatial_average: return tensor.mean(dim=[i for i in range(2, 2 + len(shp))], keepdim=True)
        else: return nn.Upsample(size, mode="bilinear", align_corners=False)(tensor)
        
    def forward(self, x0, x0_pred, is_training=False, **kwargs):
        lpips = []
        b, c, *shp = x0.shape
        is_training = is_training & self.is_training
        with torch.no_grad():
            i_embed, t_embed = self.perceptual_net(x0.argmax(1, keepdim=True).float()), self.perceptual_net(x0_pred.argmax(1, keepdim=True).float())

        diffs = [(self.tensor_normalize(i_embed[k]) - self.tensor_normalize(t_embed[k])) ** 2 for k in range(len(i_embed))]
        
        with dummy_context() if is_training else torch.no_grad():
            for k in range(len(diffs)):
                if self.use_linear:
                    diff = self.linear_layers[k].conv(self.linear_layers[k].dropout(diffs[k]))
                else: diff = diffs[k].sum(dim=1, keepdim=True)
                diff = self.tensor_average(diff, size=shp)
                lpips.append(diff)
                
        lpips_metric = sum(lpips).squeeze()
        return lpips_metric
    