from torch import nn, Tensor
from einops import rearrange
import torch
from roboscape.genie.genie.attention import SelfAttention


class Mlp(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        mlp_bias: bool = True,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=mlp_bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=mlp_bias)
        self.drop = nn.Dropout(mlp_drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)


class SingleActionEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入是单个动作维度（1维）
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),  # 输出512维嵌入
            nn.LayerNorm(output_dim),  # 加归一化，稳定训练
        )

    def forward(self, x):
        # x: (batch, 1) → 单个动作维度的批量输入
        return self.mlp(x)


class STBlock(nn.Module):
    # See Figure 4 of https://arxiv.org/pdf/2402.15391.pdf
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_bias: bool = True,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        # action encoding
        self.action_dim = 7
        # self.action_proj = nn.Linear(self.action_dim, d_model)
        self.action_proj = nn.ModuleList(
            [SingleActionEncoder() for _ in range(self.action_dim)]
        )
        # action cross attention
        self.action_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,  # 输入格式 (B, L, C)
            kdim=d_model,
            vdim=d_model,
        )
        # sequence dim is over each frame's 16x16 patch tokens
        self.spatial_attn = SelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
        )

        # sequence dim is over time sequence (16)
        self.temporal_attn = SelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
        )

        self.norm2 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        self.mlp = Mlp(
            d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop
        )

        # self.action_embed = nn.Linear(28, d_model)
        # self.norm2_action = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(d_model, 3 * d_model, bias=True)
        # )
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp_action = Mlp(d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        # with torch.no_grad():
        #     self.action_embed.weight.fill_(0)

    def forward(self, x_TSC: Tensor, action_TD) -> Tensor:
        # Process attention spatially
        B, T, S, C = x_TSC.shape

        # action_emb = self.action_embed(action_TD.float()) #4 16 256 256
        # shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(action_emb).chunk(3, dim=2)
        # x_TSC = x_TSC + gate_mlp.unsqueeze(2) * self.mlp_action(modulate(self.norm2_action(x_TSC), shift_mlp, scale_mlp))

        # 1. Spatial attention
        x_SC = rearrange(x_TSC, "B T S C -> (B T) S C")
        x_SC = x_SC + self.spatial_attn(self.norm1(x_SC))
        # 2. Action Cross Attention
        # 2.1 action dim mapping
        # action_emb = self.action_proj(action_TD.float())  # (B, T, C)
        action_emb = []
        for i in range(7):
            single_action = action_TD[:, :, i].unsqueeze(
                -1
            )  # 取第i个动作维度：(batch,1)
            emb = self.action_proj[i](single_action)  # (batch,512)
            action_emb.append(emb)
        action_emb = torch.stack(action_emb, dim=2)  # B T S_action C
        # 2.2  reshape to (B*T, 1, C) as K/V；Q is visual token
        action_kv = rearrange(action_emb, "B T S C -> (B T) S C")
        q = x_SC  # (B*T, S, C)
        # 2.3 cross-attn with residual connection
        attn_out, _ = self.action_cross_attn(
            q, action_kv, action_kv
        )  # (B*T, S_action, C)
        x_SC = x_SC + attn_out

        # 3. Temporal Casual attention
        x_TC = rearrange(x_SC, "(B T) S C -> (B S) T C", T=T)
        x_TC = x_TC + self.temporal_attn(x_TC, causal=True)

        # 4. Apply the MLP
        x_TC = x_TC + self.mlp(self.norm2(x_TC))
        x_TSC = rearrange(x_TC, "(B S) T C -> B T S C", S=S)
        return x_TSC


class STTransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_bias: bool = True,
        mlp_drop: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                STBlock(
                    num_heads=num_heads,
                    d_model=d_model,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    qk_norm=qk_norm,
                    use_mup=use_mup,
                    attn_drop=attn_drop,
                    mlp_ratio=mlp_ratio,
                    mlp_bias=mlp_bias,
                    mlp_drop=mlp_drop,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, tgt, action):
        x = tgt
        for layer in self.layers:
            x = layer(x, action)

        return x
