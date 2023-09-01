- [How does Stable Diffusion work?](https://stable-diffusion-art.com/how-stable-diffusion-work/)
- [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
- [Github hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)
- [Github CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [Github Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion)
- [Github kakaobrain/karlo](https://github.com/kakaobrain/karlo)
- [Github lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- [Github lucidrains/DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch)

[Github kakaobrain/karlo/t2i.py](https://github.com/kakaobrain/karlo/karlo/sampler/t2i.py)
```py
import torch
from karlo.sampler.t2i import T2ISampler

model = T2ISampler.from_pretrained(
    root_dir='../karlo_ckpt/', clip_model_path="ViT-L-14.pt", clip_stat_path="ViT-L-14_stats.th", sampling_type='fast'
)

prompt = "a portrait of an old monk, highly detailed."
with torch.no_grad():
    prompts_batch, prior_cf_scales_batch, decoder_cf_scales_batch, txt_feat, txt_feat_seq, tok, mask = model.preprocess(prompt, 1)
    img_feat = model._prior(txt_feat, txt_feat_seq, mask, prior_cf_scales_batch, timestep_respacing=model._prior_sm)
img_feat

```
```py
import os
import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from karlo.models.clip import CustomizedCLIP, CustomizedTokenizer
from karlo.models.prior_model import PriorDiffusionModel
from karlo.models.decoder_model import Text2ImProgressiveModel
from karlo.models.sr_64_256 import ImprovedSupRes64to256ProgressiveModel


device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) > 0 else torch.device("cpu")


def preprocess(clip_model, tokenizer, max_txt_length, prior_cf_scale, decoder_cf_scale, prompt: str, bsz: int):
    """Setup prompts & cfg scales"""
    prompts_batch = [prompt for _ in range(bsz)]

    prior_cf_scales_batch = [prior_cf_scale] * len(prompts_batch)
    prior_cf_scales_batch = torch.tensor(prior_cf_scales_batch, device=device)

    decoder_cf_scales_batch = [decoder_cf_scale] * len(prompts_batch)
    decoder_cf_scales_batch = torch.tensor(decoder_cf_scales_batch, device=device)

    """ Get CLIP text feature """
    tok, mask = tokenizer.padded_tokens_and_mask(prompts_batch, max_txt_length)
    cf_token, cf_mask = tokenizer.padded_tokens_and_mask([""], max_txt_length)
    if not (cf_token.shape == tok.shape):
        cf_token = cf_token.expand(tok.shape[0], -1)
        cf_mask = cf_mask.expand(tok.shape[0], -1)

    tok = torch.cat([tok, cf_token], dim=0)
    mask = torch.cat([mask, cf_mask], dim=0)

    tok, mask = tok.to(device=device), mask.to(device=device)
    txt_feat, txt_feat_seq = clip_model.encode_text(tok)

    return prompts_batch, prior_cf_scales_batch, decoder_cf_scales_batch, txt_feat, txt_feat_seq, tok, mask


root_dir = "../karlo_ckpt"
clip_model_path = "ViT-L-14.pt"
clip_stat_path = "ViT-L-14_stats.th"
prior_ckpt_path = "prior-ckpt-step=01000000-of-01000000.ckpt"
decoder_ckpt_path = "decoder-ckpt-step=01000000-of-01000000.ckpt"
sr_256_ckpt_path = "improved-sr-ckpt-step=1.2M.ckpt"

prior_cf_scale = 4.0
decoder_cf_scale = 8.0
prior_sm = "25"
decoder_sm = "25"
sr_sm = "7"

prompt = "a portrait of an old monk, highly detailed."
bsz = 1
save_dir = "."

tokenizer = CustomizedTokenizer()
clip = CustomizedCLIP.load_from_checkpoint(os.path.join(root_dir, clip_model_path))
clip = torch.jit.script(clip)
clip.eval().to(device=device)

config = OmegaConf.load("configs/prior_1B_vit_l.yaml")
clip_mean, clip_std = torch.load(os.path.join(root_dir, clip_stat_path), map_location="cpu")
prior = PriorDiffusionModel.load_from_checkpoint(config, tokenizer, clip_mean, clip_std, os.path.join(root_dir, prior_ckpt_path), strict=True)
prior.eval().to(device=device)

config = OmegaConf.load("configs/decoder_900M_vit_l.yaml")
decoder = Text2ImProgressiveModel.load_from_checkpoint(config, tokenizer, os.path.join(root_dir, decoder_ckpt_path), strict=True)
decoder.eval().to(device=device)

config = OmegaConf.load("configs/improved_sr_64_256_1.4B.yaml")
sr_64_256 = ImprovedSupRes64to256ProgressiveModel.load_from_checkpoint(config, os.path.join(root_dir, sr_256_ckpt_path), strict=True)
sr_64_256.eval().to(device=device)


prompts_batch, prior_cf_scales_batch, decoder_cf_scales_batch, txt_feat, txt_feat_seq, tok, mask = preprocess(
    clip, tokenizer, prior.model.text_ctx, prior_cf_scale, decoder_cf_scale, prompt, bsz
)

""" Transform CLIP text feature into image feature """
img_feat = prior(txt_feat, txt_feat_seq, mask, prior_cf_scales_batch, timestep_respacing=prior_sm)

""" Generate 64x64px images """
images_64_outputs = decoder(
    txt_feat, txt_feat_seq, tok, mask, img_feat, cf_guidance_scales=decoder_cf_scales_batch, timestep_respacing=decoder_sm
)
images_64 = None
for k, out in enumerate(images_64_outputs):
    images_64 = out
images_64 = torch.clamp(images_64, -1, 1)
images_256 = resize(images_64, [256, 256], interpolation=InterpolationMode.BICUBIC, antialias=True)

""" Upsample 64x64 to 256x256 """
images_256_outputs = sr_64_256(images_256, timestep_respacing=sr_sm)
for k, out in enumerate(images_256_outputs):
    images_256 = out
images = torch.clamp(images_256 * 0.5 + 0.5, 0.0, 1.0)

images = torch.permute(images * 255.0, [0, 2, 3, 1]).type(torch.uint8).cpu().numpy()
image = Image.fromarray(images[0])
image_name = "_".join(prompt.split(" "))
image.save(f"{save_dir}/{image_name}.jpg")
```
