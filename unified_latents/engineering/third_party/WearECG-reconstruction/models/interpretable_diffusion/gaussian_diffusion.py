from ctypes import alignment
import math
from sympy import use
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial


from models.interpretable_diffusion.transformer import Transformer
from models.interpretable_diffusion.model_utils import (
    GELU2,
    FixedRandomProjection,
    default,
    identity,
    extract,
)
from transformers import AutoTokenizer, AutoModel

from models.interpretable_diffusion.transformer_patch import PatchTransformer
from models.losses.MyLoss import DynamicSparsityLoss, mal_mse_loss

# 生成一个线性变化的 beta 序列，长度为 timesteps，用于扩散模型每一步的噪声强度
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

# 生成一个余弦变化的 beta 序列
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class MLA_Diffusion(nn.Module):
    def __init__(
        self,
        channels,
        feature_size,
        n_layer_enc=3,
        d_model=None,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type="l2",
        beta_schedule="cosine",
        n_heads=4,
        mlp_hidden_times=4,
        eta=0.0,
        attn_pd=0.0,
        resid_pd=0.0,
        kernel_size=None,
        padding_size=None,
        use_ff=True,
        use_text=False,
        text_encoder_url=None,
        reg_weight=None,
        **kwargs,
    ):
        super(MLA_Diffusion, self).__init__()

        self.eta, self.use_ff = eta, use_ff
        self.channels = channels
        self.feature_size = feature_size
        self.ff_weight = default(reg_weight, math.sqrt(self.feature_size) / 5)  # ?
        self.use_text = use_text
        
        self.model = Transformer(
            n_channel=channels,
            n_feat=feature_size,
            n_layer_enc=n_layer_enc,
            n_embd=d_model,
            n_heads=n_heads,
            attn_pdrop=attn_pd,
            resid_pdrop=resid_pd,
            mlp_hidden_times=mlp_hidden_times,
            max_len=feature_size,
            use_text=use_text,
            text_encoder_url=text_encoder_url,
            **kwargs,
        )

        # self.binary_encoder = BinaryEncoder(input_shape=(12, 1000), latent_shape=(4, 125))
        self.matrix_projector = FixedRandomProjection(
            in_shape=(12, 1000), out_shape=(4, 125)
        )
        for param in self.matrix_projector.parameters():
            param.requires_grad = False

        if self.use_text:
            self.text_model = AutoModel.from_pretrained(
                text_encoder_url,
                cache_dir="models/text_encoder",
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                text_encoder_url,
                cache_dir="models/text_encoder",
            )
            self.text_proj = nn.Sequential(
                nn.Linear(768, 512),
                GELU2(),
                nn.Linear(512, d_model),
            )
            # Freeze the text encoder
            for param in self.text_model.parameters():
                param.requires_grad = False

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate reweighting

        register_buffer(
            "loss_weight",
            torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas / 100,
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def get_text_embed(self, x, report):
        device = x.device
        with torch.no_grad():
            encoded = self.tokenizer(
                report,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            encoded = {key: val.to(device) for key, val in encoded.items()}
            # encode the queries (use the [CLS] last hidden states as the representations)
            text_emb = self.text_model(**encoded).last_hidden_state[:, 0, :]
            text_emb = self.text_proj(text_emb)

        return text_emb

    def output(self, x, t, padding_masks=None, report=None):
        text_embed = self.get_text_embed(x, report) if self.use_text else None
        model_output = self.model(
            x, t, padding_masks=padding_masks, text_embed=text_embed
        )
        return model_output

    def model_predictions(
        self, x, t, clip_x_start=False, padding_masks=None, report=None
    ):
        padding_masks = torch.ones(
            x.shape[0], self.channels, dtype=bool, device=x.device
        )

        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )
        x_start = self.output(x, t, padding_masks, report=report)
        x_start = maybe_clip(x_start)

        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=False, report=None):
        _, x_start = self.model_predictions(x, t, report=report)
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, clip_denoised=False):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def sample(self, shape):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=False):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    def generate_mts(self, batch_size=16):
        feature_size, channels = self.feature_size, self.channels
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn((batch_size, channels, feature_size))

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        elif self.loss_type == "mal_mse_loss":
            return mal_mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    @property
    def fre_loss_fn(self):
        if self.loss_type == "mal_mse_loss":
            return mal_mse_loss
        elif self.loss_type == "l2":
            return F.mse_loss

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(
        self, x, t, target=None, noise=None, padding_masks=None, mask=None, report=None
    ):

        if target is None:
            target = x

        # latent_mask = self.matrix_projector(mask)  # mask to latent mask
        # x = x * latent_mask

        noise = default(noise, lambda: torch.randn_like(x))
        x_noise = self.q_sample(x_start=x, t=t, noise=noise)  # noise sample
        model_out = self.output(x_noise, t, padding_masks, report=report)

        # target_noise = self.q_sample(x_start=target, t=t, noise=noise)
        # denoise_loss = self.loss_fn(model_out[:, 0, :], x[:, 0, :], reduction="mean")
        # alignment_loss = self.loss_fn(model_out, target_noise, reduction="mean")
        # train_loss = denoise_loss + alignment_loss
        # print(f"denoise_loss: {denoise_loss} alignment_loss: {alignment_loss}")

        train_loss = self.loss_fn(model_out, target, reduction="mean")

        fourier_loss = torch.tensor([0.0])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm="forward")
            fft2 = torch.fft.fft(target.transpose(1, 2), norm="forward")
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.fre_loss_fn(
                torch.real(fft1), torch.real(fft2), reduction="mean"
            ) + self.fre_loss_fn(torch.imag(fft1), torch.imag(fft2), reduction="mean")
            train_loss += self.ff_weight * fourier_loss

        # train_loss = reduce(train_loss, "b ... -> b (...)", "mean")
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        train_loss = train_loss.mean()
        return train_loss

    def forward(self, x, **kwargs):
        feature_size = self.feature_size
        b, n, c, device = *x.shape, x.device

        assert n == feature_size, f"number of variable must be {feature_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self._train_loss(x=x, t=t, **kwargs)

    def fast_sample_infill(
        self,
        shape,
        target,
        sampling_timesteps,
        partial_mask=None,
        clip_denoised=False,
        model_kwargs=None,
        report=None,
    ):
        batch, device, total_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.eta,
        )
        # partial_mask = self.matrix_projector(partial_mask)  # mask to latent mask
        # target = target * partial_mask

        # print(partial_mask)
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(
            time_pairs, desc="conditional sampling loop time step"
        ):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, clip_x_start=clip_denoised, report=report
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise

            img = self.langevin_fn(
                sample=img,
                mean=pred_mean,
                sigma=sigma,
                t=time_cond,
                tgt_embs=target,
                partial_mask=partial_mask,
                report=report,
                **model_kwargs,
            )
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    def sample_infill(
        self,
        shape,
        target,
        partial_mask=None,
        clip_denoised=False,
        model_kwargs=None,
        report=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        # partial_mask = self.matrix_projector(partial_mask)  # mask to latent mask
        # target = target * partial_mask

        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="conditional sampling loop time step",
            total=self.num_timesteps,
        ):

            img = self.p_sample_infill(
                x=img,
                t=t,
                clip_denoised=clip_denoised,
                target=target,
                partial_mask=partial_mask,
                model_kwargs=model_kwargs,
                report=report,
            )

        img[partial_mask] = target[partial_mask]
        return img

    def p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        clip_denoised=False,
        model_kwargs=None,
        report=None,
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = self.p_mean_variance(
            x=x, t=batched_times, clip_denoised=clip_denoised, report=report
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(
            sample=pred_img,
            mean=model_mean,
            sigma=sigma,
            t=batched_times,
            tgt_embs=target,
            partial_mask=partial_mask,
            report=report,
            **model_kwargs,
        )

        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
        self,
        sample,
        mean,
        sigma,
        t,
        tgt_embs,
        partial_mask,
        coef,
        learning_rate,
        coef_=0.0,
        report=None,
    ):

        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        ecg_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([ecg_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=sample, t=t, report=report)

                if sigma.mean() == 0:
                    logp_term = (
                        coef * ((mean - ecg_embs_param) ** 2 / 1.0).mean(dim=0).sum()
                    )
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = (
                        coef * ((mean - ecg_embs_param) ** 2 / sigma).mean(dim=0).sum()
                    )
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss / sigma.mean()).mean(dim=0).sum()

                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(ecg_embs_param.data)
                ecg_embs_param = torch.nn.Parameter(
                    (
                        ecg_embs_param.data + coef_ * sigma.mean().item() * epsilon
                    ).detach()
                )

        sample[~partial_mask] = ecg_embs_param.data[~partial_mask]
        return sample


if __name__ == "__main__":
    pass
