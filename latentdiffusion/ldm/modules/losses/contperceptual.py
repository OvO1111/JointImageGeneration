import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from taming.modules.discriminator.model import weights_init
from taming.modules.util import ActNorm

# from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no? -> LESS
from ldm.modules.losses.lpips import LPIPS
from einops import rearrange


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1. - logits_real), dim=[1,2,3])
    loss_fake = torch.mean(F.relu(1. + logits_fake), dim=[1,2,3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def l1(x, y):
    return torch.abs(x-y)

def l2(x, y):
    return torch.pow((x-y), 2)


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=1.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", pixel_loss="l1", image_gan_weight=0.5, ct_gan_weight=0.5, gan_feat_weight=1.0, dims=2):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_loss = l1 if pixel_loss == "l1" else l2
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.frame_discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 getIntermFeat=dims == 3
                                                 ).apply(weights_init)
        self.ct_discriminator = NLayerDiscriminator3D(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.image_gan_weight = image_gan_weight
        self.ct_gan_weight = ct_gan_weight
        self.gan_feat_weight = gan_feat_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        b, c, *shp = inputs.shape
        assert inputs.shape == reconstructions.shape, f"{inputs.shape} and {reconstructions.shape} not match"
        if c > 1:
            inputs = rearrange(inputs, "b c ... -> (b c) 1 ...")
            reconstructions = rearrange(reconstructions, "b c ... -> (b c) 1 ...")
        if len(shp) == 3:
            out = self.forward_3d(inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer, cond, split, weights)
        elif len(shp) == 2:
            out = self.forward_2d(inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer, cond, split, weights)
        # if c > 1:
        #     out = rearrange(out, "(b c) 1 ... -> b c ...", c=c)
        return out

    def forward_2d(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.frame_discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.frame_discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.frame_discriminator(inputs.contiguous().detach())
                logits_fake = self.frame_discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.frame_discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.frame_discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

    def forward_3d(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        # rearrange inputs so that z axis in 3d is grouped with batch dim
        frames = rearrange(inputs, "b c h w d -> (b h) c w d")
        frames_rec = rearrange(reconstructions, "b c h w d -> (b h) c w d")
        if cond is not None:
            frame_cond = rearrange(cond, "b c h w d -> (b h) c w d")
        
        rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(frames.contiguous(), frames_rec.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.])

        nll_loss = rec_loss #/ torch.exp(self.logvar) + self.logvar
        """weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]"""
        nll_loss = torch.mean(nll_loss)
        weighted_nll_loss = nll_loss
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                if self.image_gan_weight > 0:
                    logits_fake_image, pred_fake_image = self.frame_discriminator(frames_rec.contiguous())
                if self.ct_gan_weight > 0:
                    logits_fake_ct, pred_fake_ct = self.ct_discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                if self.image_gan_weight > 0:
                    logits_fake_image, pred_fake_image = self.frame_discriminator(torch.cat((frames_rec.contiguous(), frame_cond), dim=1))
                if self.ct_gan_weight > 0:
                    logits_fake_ct, pred_fake_ct = self.ct_discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -(torch.mean(logits_fake_image) + torch.mean(logits_fake_ct)) / 2

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)
                
            # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
            frame_gan_feat_loss = 0
            ct_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.gan_feat_weight > 0:
                if self.image_gan_weight > 0:
                    logits_frame_real, pred_image_real = self.frame_discriminator(frames)
                    for i in range(len(pred_fake_image)-1):
                        frame_gan_feat_loss += feat_weights * F.l1_loss(pred_fake_image[i], pred_image_real[i].detach())
                if self.ct_gan_weight > 0:
                    logits_ct_real, pred_ct_real = self.ct_discriminator(inputs)
                    for i in range(len(pred_fake_ct)-1):
                        ct_gan_feat_loss += feat_weights * F.l1_loss(pred_fake_ct[i], pred_ct_real[i].detach())
                gan_feat_loss = disc_factor * self.gan_feat_weight * \
                    (frame_gan_feat_loss + ct_gan_feat_loss)
            else:
                gan_feat_loss = torch.tensor(0.).to(kl_loss.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss + self.gan_feat_weight * gan_feat_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), 
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), 
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/gan_feat_loss".format(split): gan_feat_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                if self.image_gan_weight > 0:
                    logits_frame_real, _ = self.frame_discriminator(frames.contiguous().detach())
                    logits_frame_fake, _ = self.frame_discriminator(frames_rec.contiguous().detach())
                if self.ct_gan_weight > 0:
                    logits_ct_real, _ = self.ct_discriminator(inputs.contiguous().detach())
                    logits_ct_fake, _ = self.ct_discriminator(reconstructions.contiguous().detach())
            else:
                if self.image_gan_weight > 0:
                    logits_frame_real, _ = self.frame_discriminator(torch.cat((frames.contiguous().detach(), frame_cond), dim=1))
                    logits_frame_fake, _ = self.frame_discriminator(torch.cat((frames_rec.contiguous().detach(), frame_cond), dim=1))
                if self.ct_gan_weight > 0:
                    logits_ct_real, _ = self.ct_discriminator(torch.cat((inputs.contiguous().detach(), frame_cond), dim=1))
                    logits_ct_fake, _ = self.ct_discriminator(torch.cat((reconstructions.contiguous().detach(), frame_cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * (self.disc_loss(logits_frame_real, logits_frame_fake) + self.disc_loss(logits_ct_real, logits_ct_fake)) / 2

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_frames_real".format(split): logits_frame_real.detach().mean(),
                   "{}/logits_frames_fake".format(split): logits_frame_fake.detach().mean(),
                   "{}/logits_ct_real".format(split): logits_ct_real.detach().mean(),
                   "{}/logits_ct_fake".format(split): logits_ct_fake.detach().mean(),
                   }
            return d_loss, log


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True, use_actnorm=False):
        # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        if use_actnorm:
            norm_layer = ActNorm
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input)


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True, use_actnorm=False):
        super(NLayerDiscriminator3D, self).__init__()
        if use_actnorm:
            norm_layer = ActNorm
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input)