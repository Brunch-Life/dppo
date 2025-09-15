import math

import mup
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from tqdm import tqdm
from transformers.utils import ModelOutput

from env.roboscape.genie.genie.factorization_utils import FactorizedEmbedding, factorize_labels
from env.roboscape.genie.genie.config import GenieConfig
from env.roboscape.genie.genie.st_transformer import STTransformerDecoder
from timm.models.vision_transformer import Mlp


def cosine_schedule(u):
    """u in [0, 1]"""
    if isinstance(u, torch.Tensor):
        cls = torch
    elif isinstance(u, float):
        cls = math
    else:
        raise NotImplementedError(f"Unexpected {type(u)=} {u=}")

    return cls.cos(u * cls.pi / 2)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)


class STMaskGIT(nn.Module, PyTorchModelHubMixin):
    # Next-Token prediction as done in https://arxiv.org/pdf/2402.15391.pdf
    def __init__(self, config: GenieConfig):
        super().__init__()
        # self.h = self.w = math.isqrt(config.S)
        self.h = 2 * math.isqrt(config.S)
        self.w = math.isqrt(config.S)
        # assert self.h**2 == config.S, "Expected S to be square"

        self.decoder = STTransformerDecoder(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            qkv_bias=config.qkv_bias,
            proj_bias=config.proj_bias,
            qk_norm=config.qk_norm,
            use_mup=config.use_mup,
            attn_drop=config.attn_drop,
            mlp_ratio=config.mlp_ratio,
            mlp_bias=config.mlp_bias,
            mlp_drop=config.mlp_drop,
        )

        self.pos_embed_TSC = torch.nn.Parameter(
            torch.zeros(1, config.T, config.S * 2, config.d_model)
        )
        # self.pos_embed_TSC = torch.nn.Parameter(
        #     torch.zeros(1, config.T, config.S, config.d_model)
        # )
        self.mask_token_id = config.image_vocab_size

        self.token_embed = (
            FactorizedEmbedding(  # also works for num_factored_vocabs = 1
                factored_vocab_size=config.factored_vocab_size,
                num_factored_vocabs=config.num_factored_vocabs,
                d_model=config.d_model,
                mask_token_id=self.mask_token_id,
            )
        )

        self.use_action = config.use_action
        self.use_text = config.use_text
        self.use_done = config.use_done

        if config.use_action:
            self.action_embed = nn.Linear(config.action_dim, config.d_model)
        if config.use_text:
            self.text_embed = nn.Linear(config.text_dim, config.d_model)
        if config.use_done:
            self.done_pred = nn.Sequential(
                nn.Linear(config.d_model, 64), nn.ReLU(), nn.Linear(64, 1)
            )
        # self.norm2 = nn.LayerNorm(config.d_model, elementwise_affine=False, eps=1e-6)
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        # )
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=config.d_model, hidden_features=int(config.d_model * config.mlp_ratio), act_layer=approx_gelu, drop=0)
        # with torch.no_grad():
        #     self.action_embed.weight.fill_(0)
        cls = (
            FixedMuReadout if config.use_mup else nn.Linear
        )  # (Fixed)MuReadout might slow dow down compiled training?
        self.out_x_proj = cls(
            config.d_model, config.factored_vocab_size * config.num_factored_vocabs
        )

        self.config = config

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int,
        min_new_tokens: int = None,
        return_logits: int = False,
        maskgit_steps: int = 1,
        temperature: float = 0.0,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Args designed to match the format of Llama.
        We ignore `attention_mask`, and use `max_new_tokens` to determine the number of frames to generate.

        Returns: `(sample_THW, factored_logits)` if `return_logits` else `sample_THW`
            sample_THW: size (B, num_new_frames * H * W) corresponding to autoregressively generated
                unfactorized token ids for future frames.
            Optionally, factored_logits: size (B, factored_vocab_size, num_factored_vocabs, num_new_frames, H, W).
        """
        assert min_new_tokens in (
            None,
            max_new_tokens,
        ), "Expecting `min_new_tokens`, if specified, to match `max_new_tokens`."

        assert (
            max_new_tokens % self.config.S == 0
        ), "Expecting `max_new_tokens` to be a multiple of `self.config.S`."
        num_new_frames = max_new_tokens // self.config.S

        inputs_THW = rearrange(
            input_ids.clone(), "b (t h w) -> b t h w", h=self.h, w=self.w
        )
        inputs_masked_THW = torch.cat(
            [
                inputs_THW,
                torch.full(
                    (input_ids.size(0), num_new_frames, self.h, self.w),
                    self.mask_token_id,
                    dtype=torch.long,
                    device=input_ids.device,
                ),
            ],
            dim=1,
        )

        all_factored_logits = []
        for timestep in range(inputs_THW.size(1), inputs_THW.size(1) + num_new_frames):
            # could change sampling hparams
            sample_HW, factored_logits = self.maskgit_generate(
                inputs_masked_THW,
                timestep,
                maskgit_steps=maskgit_steps,
                temperature=temperature,
            )
            inputs_masked_THW[:, timestep] = sample_HW
            all_factored_logits.append(factored_logits)

        predicted_tokens = rearrange(inputs_masked_THW, "B T H W -> B (T H W)")
        if return_logits:
            return predicted_tokens, torch.stack(
                all_factored_logits, dim=3
            )  # (b, factored_vocab_size, num_factored_vocabs, num_new_frames, h, w)
        else:
            return predicted_tokens

    @staticmethod
    def init_mask(prompt_THW):
        # since we generate 1 image at a time, the mask should be for a single frame, not across all frames.
        T, H, W = prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)
        unmasked = torch.zeros(
            prompt_THW.size(0), H * W, dtype=torch.bool, device=prompt_THW.device
        )
        return unmasked

    @torch.no_grad()
    def maskgit_generate(
        self,
        prompt_THW: torch.LongTensor,
        prompt_action,
        prompt_text,
        out_t: int,
        maskgit_steps: int = 1,
        temperature: float = 0.0,
        unmask_mode: str = "random",
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Performs MaskGIT-style inference to predict frame `out_t`.

        Args:
            prompt_THW: Unfactorized token ids, size (B, T, H, W)
            out_t: Will return predicted unfactorized token ids for this frame.
                Should be >= 1 as the 0th frame is assumed to be given.
                Expects all future frames to be fully masked.
            maskgit_steps: The number of MaskGIT-style inference steps to take.
            temperature: Sampling temperature.
                In the factorized case, sampling is performed for each factorized vocabulary independently.
                If temperature is <= 1e-8, will be greedy (i.e. argmax) instead of actual sampling.
            unmask_mode: The method to determine tokens to unmask during each step of MaskGIT inference.
                Options:
                    - "greedy" for unmasking the most confident tokens, which is matches the original MaskGIT
                    - "random" for randomly choosing tokens to unmask
                "greedy" tends to copy the previous frame, so we default to "random" instead.

        Returns: (sample_HW, factored_logits)
            sample_HW: size (B, H, W) corresponding to predicted unfactorized token ids for frame `out_t`.
            factored_logits: size (B, factored_vocab_size, num_factored_vocabs, H, W).
        """
        # assume we have pre-masked z{out_t}...zT with all masks
        assert out_t, "maskgit_generate requires out_t > 0"
        assert torch.all(
            prompt_THW[:, out_t:] == self.mask_token_id
        ), f"when generating z{out_t}, frames {out_t} and later must be masked"

        bs, t, h, w = (
            prompt_THW.size(0),
            prompt_THW.size(1),
            prompt_THW.size(2),
            prompt_THW.size(3),
        )

        # this will be modified in place on each iteration of this loop
        unmasked = self.init_mask(prompt_THW)

        logits_CTHW, done_pred = self.compute_logits(
            prompt_THW, prompt_action, prompt_text
        )
        logits_CHW = logits_CTHW[:, :, out_t]
        orig_logits_CHW = (
            logits_CHW.clone()
        )  # Return these original logits, not logits after partially sampling.
        for step in range(maskgit_steps):
            # Perform a single maskgit step (cosine schedule), updating unmasked in-place
            if step > 0:  # recompute logits with updated prompt
                logits_CHW, done_pred = self.compute_logits(
                    prompt_THW, prompt_action, prompt_text
                )
                logits_CHW = logits_CHW[:, :, out_t]

            factored_logits = rearrange(
                logits_CHW,
                "b (num_vocabs vocab_size) h w -> b vocab_size num_vocabs h w",
                vocab_size=self.config.factored_vocab_size,
                num_vocabs=self.config.num_factored_vocabs,
            )

            factored_probs = torch.nn.functional.softmax(factored_logits, dim=1)

            samples_HW = torch.zeros(
                (bs, h, w), dtype=torch.long, device=prompt_THW.device
            )
            confidences_HW = torch.ones(
                (bs, h, w), dtype=torch.float, device=prompt_THW.device
            )
            for probs in factored_probs.flip(2).unbind(2):
                if temperature <= 1e-8:  # greedy sampling
                    sample = probs.argmax(dim=1)
                else:
                    # Categorical expects last dim to be channel dim
                    dist = torch.distributions.categorical.Categorical(
                        probs=rearrange(probs, "b vocab_size ... -> b ... vocab_size")
                        / temperature
                    )
                    sample = dist.sample()
                samples_HW *= self.config.factored_vocab_size
                samples_HW += sample
                confidences_HW *= torch.gather(probs, 1, sample.unsqueeze(1)).squeeze(1)

            prev_unmasked = unmasked.clone()
            prev_img_flat = rearrange(prompt_THW[:, out_t], "B H W -> B (H W)")

            # samples_flat = samples_HW.reshape(bs, self.config.S)
            samples_flat = samples_HW.reshape(bs, -1)
            self.config.S = samples_flat.shape[-1]
            if step != maskgit_steps - 1:  # skip masking for last maskgit step
                # use cosine mask scheduling function, n is how many of frame out_t to mask
                n = math.ceil(
                    cosine_schedule((step + 1) / maskgit_steps) * self.config.S
                )

                if unmask_mode == "greedy":
                    # set the n patches with the least confidence to mask_token
                    confidences_flat = confidences_HW.reshape(bs, self.config.S)
                elif unmask_mode == "random":
                    # randomize confidences, so that patches are randomly masked
                    confidences_flat = torch.rand_like(confidences_HW).reshape(
                        bs, self.config.S
                    )
                    # not probability distribution anymore, but only relative order matters
                else:
                    raise NotImplementedError(
                        f"Expected `unmask_mode` to be one of ['greedy', 'random'], "
                        f"got {unmask_mode}"
                    )

                confidences_flat[unmasked] = torch.inf
                least_confident_tokens = torch.argsort(confidences_flat, dim=1)
                # unmask the (self.config.S - n) most confident tokens
                unmasked.scatter_(1, least_confident_tokens[:, n:], True)
                samples_flat.scatter_(
                    1, least_confident_tokens[:, :n], self.mask_token_id
                )

            # copy previously unmasked values from prompt input into sample
            samples_flat[prev_unmasked] = prev_img_flat[prev_unmasked]
            samples_HW = samples_flat.reshape(-1, h, w)

            # feed back to iteratively decode
            prompt_THW[:, out_t] = samples_HW

        # Return the final sample and logits
        return (
            samples_HW,
            rearrange(
                orig_logits_CHW,
                "B (num_vocabs vocab_size) H W -> B vocab_size num_vocabs H W",
                vocab_size=self.config.factored_vocab_size,
                num_vocabs=self.config.num_factored_vocabs,
                H=h,
                W=w,
            ),
            done_pred,
        )

    def compute_loss_and_acc(
        self,
        logits_CTHW,
        targets_THW,
        relevant_mask_THW,
        done_pred=None,
        done_label=None,
    ):
        # Video token prediction
        targets_THW = targets_THW.clone()
        logits_CTHW, logits_CTHW_2, targets_THW = (
            logits_CTHW[:, :, 1:],
            logits_CTHW[:, :, :-1],
            targets_THW[:, 1:],
        )  # first frame always unmasked

        # 计算指示函数的值
        indicator = (
            torch.ne(logits_CTHW_2, logits_CTHW).any(dim=1, keepdim=True).float()
        )
        match = torch.eq(logits_CTHW_2, logits_CTHW).any(dim=1)
        # 计算相等的元素总数
        equal_count = torch.sum(match).item()
        # print(equal_count/match.numel())
        # 底数 c = e
        c = torch.exp(torch.tensor(1.0))
        # 计算 ω_{i + 1}
        weight = c**indicator
        # print(weight.size())
        factored_logits = rearrange(
            logits_CTHW,
            "b (num_vocabs vocab_size) t h w -> b vocab_size num_vocabs t h w",
            vocab_size=self.config.factored_vocab_size,
            num_vocabs=self.config.num_factored_vocabs,
        )

        factored_targets = factorize_labels(targets_THW)
        # print(factored_weight.size())
        # print(factored_logits.size())
        # print(F.cross_entropy(factored_logits, factored_targets, reduction="none").size())
        loss_THW = (
            F.cross_entropy(factored_logits, factored_targets, reduction="none")
        ).sum(dim=1)
        acc_THW = (factored_logits.argmax(dim=1) == factored_targets).all(dim=1)
        loss_done = F.mse_loss(done_pred, done_label) if self.use_done else 0
        # Compute the mean masked error.
        # Multiply loss values by mask instead of indexing them, more computationally efficient.
        num_masked_tokens = torch.sum(relevant_mask_THW)
        relevant_loss = (
            torch.sum(loss_THW * relevant_mask_THW) / num_masked_tokens + 5 * loss_done
        )
        relevant_acc = (
            torch.sum(acc_THW * relevant_mask_THW).float() / num_masked_tokens
        )
        # only optimize on the masked/noised logits?
        return relevant_loss, relevant_acc, loss_done

    def compute_logits(self, x_THW, action_TD, text_TD):  # action 4 16 28
        # x_THW is for z0,...,zT while x_targets is z1,...,zT
        x_TS = rearrange(x_THW, "B T H W -> B T (H W)")  # 4 16 256
        # print(x_TS.size())
        # print(action_TD.size())
        # x_TS_merge = torch.cat([x_TS, action_TD], dim=2)
        x_TSC = self.token_embed(x_TS)  # 4 16 256 256

        # original modulate
        # x_TSC = self.decoder(x_TSC + self.pos_embed_TSC, action_TD) #B T S D

        # additive
        if self.use_action:
            action_emb = self.action_embed(action_TD.float())  # 4 16 256 256
            action_emb = rearrange(action_emb, "B T D -> B T 1 D")

        if self.use_text:
            text_emb = self.text_embed(text_TD.float())
            text_emb = rearrange(text_emb, "B T D -> B T 1 D")

        if not self.use_action:
            action_emb = torch.zeros(text_emb.size()).to(text_emb.device)
        import os

        # save_path = "/iag_ad_01/ad/tangyinzhou/tyz/observation-genie/all_zeros"
        # os.makedirs(save_path, exist_ok=True)
        # torch.save(action_emb, f"{save_path}/{len(os.listdir(save_path))}.pt")
        if not self.use_text:
            text_emb = torch.zeros(action_emb.size()).to(action_emb.device)
        # x_TSC = self.decoder(
        #     x_TSC + self.pos_embed_TSC + action_emb, action_TD
        # )  # B T S D
        if self.use_action and self.use_text:
            decode_feature = torch.concat([text_TD, action_TD], dim=-1)
        elif self.use_text:
            decode_feature = text_TD
        else:
            decode_feature = action_TD

        x_TSC = self.decoder(
            x_TSC + self.pos_embed_TSC + text_emb + action_emb,
            decode_feature,
        )  # B T S D

        # print(x_TSC.size())
        x_next_TSC = self.out_x_proj(x_TSC)
        if self.use_done:
            done_preds = [
                self.done_pred(torch.mean(x_TSC[i], dim=1))
                for i in range(x_TSC.shape[0])
            ]
            done_pred = torch.stack(done_preds)
        logits_CTHW = rearrange(
            x_next_TSC, "B T (H W) C -> B C T H W", H=self.h, W=self.w
        )
        # image_mean, image_max, image_min = (
        #     torch.mean(x_TSC),
        #     torch.max(x_TSC),
        #     torch.min(x_TSC),
        # )
        # action_mean, action_max, action_min = (
        #     torch.mean(action_emb),
        #     torch.max(action_emb),
        #     torch.min(action_emb),
        # )
        # L2Norm = (x_TSC - action_emb).norm(p=2)
        # print(
        #     f"image mean: {image_mean}, image max: {image_max}, image min: {image_min}"
        # )
        # print(
        #     f"action mean: {action_mean}, action max: {action_max}, action min: {action_min}"
        # )
        # print(f"L2 Norm: {L2Norm}")
        return logits_CTHW, done_pred

    def forward(self, input_ids, labels, actions, texts, done=None):
        T, H, W, D = self.config.T, self.h, self.w, self.config.action_dim
        x_THW = rearrange(input_ids, "B (T H W) -> B T H W", T=T, H=H, W=W)
        actions = rearrange(actions, "B (T D) -> B T D", T=T, D=D)
        texts = rearrange(texts, "B (T D) -> B T D", T=T, D=self.config.text_dim)
        done_label = (
            rearrange(done, "B (T D) -> B T D", T=T, D=1) if self.use_done else None
        )
        logits_CTHW, done_pred = self.compute_logits(x_THW, actions, texts)
        labels = rearrange(labels, "B (T H W) -> B T H W", T=T, H=H, W=W)

        # Record the loss over masked tokens only to make it more comparable to LLM baselines
        relevant_mask = (
            x_THW[:, 1:] == self.mask_token_id
        )  # could also get mask of corrupted tokens by uncommenting line in `get_maskgit_collator`
        relevant_loss, relevant_acc, loss_done = self.compute_loss_and_acc(
            logits_CTHW,
            labels,
            relevant_mask,
            done_pred,
            done_label,
        )
        return ModelOutput(
            loss=relevant_loss,
            acc=relevant_acc,
            logits=logits_CTHW,
            loss_done=loss_done,
        )

    def init_weights(self):
        """Works with and without muP."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module.weight, "infshape"):  # muP
                    mup.normal_(module.weight, mean=0.0, std=std)
                else:
                    module.weight.data.normal_(mean=0.0, std=std)

                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def set_mup_shapes(self, rescale_params=False):
        base_config = self.config.shallow_copy()
        base_config.num_heads = 8
        base_config.d_model = 256  # currently hardcoding to this shape
        base_model = STMaskGIT(base_config)

        mup.set_base_shapes(self, base_model, rescale_params=rescale_params)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Extra logic for muP."""
        model = super().from_pretrained(*args, **kwargs)
        if model.config.use_mup:
            model.set_mup_shapes(rescale_params=False)

        return model


class FixedMuReadout(mup.MuReadout):
    def forward(self, x):
        """
        Using `return super(mup.MuReadout, self).forward(self.output_mult * x / self.width_mult())` with `torch.compile`
        results in two divisions by `self.width_mult()` for some reason
        """
        # return F.linear(self.output_mult * x / self.width_mult(), self.weight, self.bias)  # equivalent
        return nn.Linear.forward(self, self.output_mult * x / self.width_mult())
