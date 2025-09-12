"""
MLP models for diffusion policies.

"""

import torch
import torch.nn as nn
import logging
import einops
from copy import deepcopy
from typing import Union

from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from torchvision import transforms

log = logging.getLogger(__name__)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out



class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8,
        input_len=1
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM 
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level. 
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        len_tensor = input_len
        len_tensors = []

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                len_tensors.append(len_tensor)
                len_tensor = (len_tensor + 1) // 2

        print(len_tensors)

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)

            out_odd = len_tensors.pop() % 2

            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in, out_odd=out_odd) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # print(timesteps.shape, sample.shape)
        # print(sample.dtype, global_cond.dtype)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            # print(idx, x.shape)
            x = downsample(x)

        for idx, mid_module in enumerate(self.mid_modules):
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            # print(idx, x.shape, h[-1].shape)
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # print(x.shape)
            x = resnet2(x, global_feature)
            # print(x.shape)
            x = upsample(x)
            # print(x.shape)

        # print(x.shape)
        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim, out_odd=False):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 3 if out_odd else 4, 2, 1)
        # self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class VisionDiffusionMLP(nn.Module):
    """With ViT backbone"""

    def __init__(
        self,
        backbone,
        action_dim,
        horizon_steps,
        cond_dim,
        img_cond_steps=1,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        spatial_emb=0,
        visual_feature_dim=128,
        dropout=0,
        num_img=1,
        augment=False,
    ):
        super().__init__()

        # vision
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps



        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=cond_dim,
            input_len=horizon_steps,
            diffusion_step_embed_dim=256
        )

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )
        self.num_inference_timesteps = 10



    def forward(
        self,
        x,
        time,
        cond: dict,
        **kwargs,
    ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)

        TODO long term: more flexible handling of cond
        """
        B, Ta, Da = x.shape
        _, T_rgb, C, H, W = cond["rgb"].shape

        # flatten chunk
        x = x.view(B, -1)

        # flatten history
        state = torch.zeros(B, 10).to(x.device)

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"][:, -self.img_cond_steps :]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        # convert rgb to float32 for augmentation
        rgb = rgb.float()

        # get vit output - pass in two images separately
        if self.num_img > 1:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            feat = self.backbone(rgb1, rgb2)
        else:
            raise NotImplementedError

        cond_encoded = torch.cat([feat, state], dim=-1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Ta, Da), device=cond_encoded.device)
        naction = noisy_action





        # init scheduler
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # print(naction.shape, obs_cond.shape)
            # predict noise
            noise_pred = self.noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=cond_encoded
            )

            # print(noise_pred.shape)

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample


        assert naction.shape == (B, Ta, Da)

        return naction


# class DiffusionMLP(nn.Module):

#     def __init__(
#         self,
#         action_dim,
#         horizon_steps,
#         cond_dim,
#         time_dim=16,
#         mlp_dims=[256, 256],
#         cond_mlp_dims=None,
#         activation_type="Mish",
#         out_activation_type="Identity",
#         use_layernorm=False,
#         residual_style=False,
#     ):
#         super().__init__()
#         output_dim = action_dim * horizon_steps
#         self.time_embedding = nn.Sequential(
#             SinusoidalPosEmb(time_dim),
#             nn.Linear(time_dim, time_dim * 2),
#             nn.Mish(),
#             nn.Linear(time_dim * 2, time_dim),
#         )
#         if residual_style:
#             model = ResidualMLP
#         else:
#             model = MLP
#         if cond_mlp_dims is not None:
#             self.cond_mlp = MLP(
#                 [cond_dim] + cond_mlp_dims,
#                 activation_type=activation_type,
#                 out_activation_type="Identity",
#             )
#             input_dim = time_dim + action_dim * horizon_steps + cond_mlp_dims[-1]
#         else:
#             input_dim = time_dim + action_dim * horizon_steps + cond_dim
#         self.mlp_mean = model(
#             [input_dim] + mlp_dims + [output_dim],
#             activation_type=activation_type,
#             out_activation_type=out_activation_type,
#             use_layernorm=use_layernorm,
#         )
#         self.time_dim = time_dim

#     def forward(
#         self,
#         x,
#         time,
#         cond,
#         **kwargs,
#     ):
#         """
#         x: (B, Ta, Da)
#         time: (B,) or int, diffusion step
#         cond: dict with key state/rgb; more recent obs at the end
#             state: (B, To, Do)
#         """
#         B, Ta, Da = x.shape

#         # flatten chunk
#         x = x.view(B, -1)

#         # flatten history
#         state = cond["state"].view(B, -1)

#         # obs encoder
#         if hasattr(self, "cond_mlp"):
#             state = self.cond_mlp(state)

#         # append time and cond
#         time = time.view(B, 1)
#         time_emb = self.time_embedding(time).view(B, self.time_dim)
#         x = torch.cat([x, time_emb, state], dim=-1)

#         # mlp head
#         out = self.mlp_mean(x)
#         return out.view(B, Ta, Da)
