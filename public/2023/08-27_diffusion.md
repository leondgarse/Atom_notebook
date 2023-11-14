- [How does Stable Diffusion work?](https://stable-diffusion-art.com/how-stable-diffusion-work/)
- [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
- [Github hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)
- [Github CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [Github Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion)
- [Github kakaobrain/karlo](https://github.com/kakaobrain/karlo)
- [Github lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- [Github lucidrains/DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch)

# Karlo
  - [Github kakaobrain/karlo/t2i.py](https://github.com/kakaobrain/karlo/karlo/sampler/t2i.py)
  ```sh
  pip install omegaconf
  pip install git+https://github.com/openai/CLIP.git

  !wget 'https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/0b62380a75e56f073e2844ab5199153d/ViT-L-14_stats.th' && \
  wget 'https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/096db1af569b284eb76b3881534822d9/ViT-L-14.pt' && \
  wget 'https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/85626483eaca9f581e2a78d31ff905ca/prior-ckpt-step%3D01000000-of-01000000.ckpt' && \
  wget 'https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/efdf6206d8ed593961593dc029a8affa/decoder-ckpt-step%3D01000000-of-01000000.ckpt' && \
  wget 'https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/4226b831ae0279020d134281f3c31590/improved-sr-ckpt-step%3D1.2M.ckpt'
  ```
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

      """ Generate 64x64px images """
      images_64_outputs = self._decoder(
          txt_feat, txt_feat_seq, tok, mask, img_feat, cf_guidance_scales=decoder_cf_scales_batch, timestep_respacing=self._decoder_sm,
      )

      images_64 = None
      for k, out in enumerate(images_64_outputs):
          images_64 = out
      images_64 = torch.clamp(images_64, -1, 1)

      """ Upsample 64x64 to 256x256 """
      images_256 = TVF.resize(images_64, [256, 256], interpolation=InterpolationMode.BICUBIC, antialias=True)
      images_256_outputs = self._sr_64_256(images_256, timestep_respacing=self._sr_sm)

      for k, out in enumerate(images_256_outputs):
          images_256 = out
  yield torch.clamp(images_256 * 0.5 + 0.5, 0.0, 1.0)
  ```
  ```py
  import os
  import torch
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

  np.save('aa.npy', images_256.detach().cpu().numpy())
  ```
  ```py
  """ Upsample 64x64 to 256x256 """
  import os
  import torch
  from PIL import Image
  from omegaconf import OmegaConf
  from karlo.models.sr_64_256 import ImprovedSupRes64to256ProgressiveModel

  device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) > 0 else torch.device("cpu")
  root_dir = "../karlo_ckpt"
  sr_256_ckpt_path = "improved-sr-ckpt-step=1.2M.ckpt"
  sr_sm = "7"

  images_256 = torch.from_numpy(np.load('aa.npy')).to(device)
  config = OmegaConf.load("configs/improved_sr_64_256_1.4B.yaml")
  sr_64_256 = ImprovedSupRes64to256ProgressiveModel.load_from_checkpoint(config, os.path.join(root_dir, sr_256_ckpt_path), strict=True)
  sr_64_256.eval().to(device=device)

  images_256_outputs = sr_64_256(images_256, timestep_respacing=sr_sm)
  for k, out in enumerate(images_256_outputs):
      images_256 = out
  images = torch.clamp(images_256 * 0.5 + 0.5, 0.0, 1.0)

  images = torch.permute(images * 255.0, [0, 2, 3, 1]).type(torch.uint8).cpu().numpy()
  image = Image.fromarray(images[0])
  image_name = "_".join(prompt.split(" "))
  image.save(f"{save_dir}/{image_name}.jpg")
  ```
# Difusers
## Basic usage
  - [huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline)
  - A pipeline is a quick and easy way to run a model for inference, requiring no more than four lines of code to generate an image.
  ```py
  import torch
  device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) > 0 else torch.device("cpu")

  from diffusers import DDPMPipeline

  ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to(device)
  image = ddpm(num_inference_steps=25).images[0]
  image.save("aa.jpg")
  ```
## Deconstruct a basic pipeline
  - The pipeline denoises an image by taking random noise the size of the desired output and passing it through the model several times.
  - At each timestep, the model predicts the noise residual and the scheduler uses it to predict a less noisy image.
  - The pipeline repeats this process until it reaches the end of the specified number of inference steps.
  ```py
  import torch
  device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) > 0 else torch.device("cpu")

  # Load the model and scheduler
  from diffusers import DDPMScheduler, UNet2DModel

  scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
  model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to(device)

  # Set the number of timesteps to run the denoising process for
  # Each element corresponds to a timestep at which the model denoises an image.
  # When you create the denoising loop later, you’ll iterate over this tensor to denoise an image:
  scheduler.set_timesteps(50)
  print(scheduler.timesteps)

  # Create some random noise with the same shape as the desired output:
  sample_size = model.config.sample_size
  noise = torch.randn((1, 3, sample_size, sample_size)).to(device)

  # Now write a loop to iterate over the timesteps.
  # At each timestep, the model does a UNet2DModel.forward() pass and returns the noisy residual.
  # The scheduler’s step() method takes the noisy residual, timestep, and input and it predicts the image at the previous timestep.
  # This output becomes the next input to the model in the denoising loop,
  # and it’ll repeat until it reaches the end of the timesteps array.
  input = noise
  for t in scheduler.timesteps:
      with torch.no_grad():
          noisy_residual = model(input, t).sample
      previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
      input = previous_noisy_sample

  # The last step is to convert the denoised output into an image:
  from PIL import Image
  import numpy as np

  image = (input / 2 + 0.5).clamp(0, 1).squeeze()
  image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
  image = Image.fromarray(image)
  image.save("aa.jpg")
  ```
## Deconstruct the Stable Diffusion pipeline
  - Stable Diffusion is a text-to-image latent diffusion model. It is called a latent diffusion model because it works with a lower-dimensional representation of the image instead of the actual pixel space, which makes it more memory efficient.
  - The encoder compresses the image into a smaller representation, and a decoder to convert the compressed representation back into an image.
  - For text-to-image models, you’ll need a tokenizer and an encoder to generate text embeddings.
  ```py
  import torch
  device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) > 0 else torch.device("cpu")

  from PIL import Image
  from tqdm.auto import tqdm
  from transformers import CLIPTextModel, CLIPTokenizer
  from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

  vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True).to(device)
  tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
  text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True).to(device)
  unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True).to(device)

  from diffusers import UniPCMultistepScheduler

  scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

  """ Create text embeddings """
  # The next step is to tokenize the text to generate embeddings.
  # The text is used to condition the UNet model and steer the diffusion process towards something that resembles the input prompt.
  # The guidance_scale parameter determines how much weight should be given to the prompt when generating an image.
  prompt = ["a photograph of an astronaut riding a horse"]
  height = 512  # default height of Stable Diffusion
  width = 512  # default width of Stable Diffusion
  num_inference_steps = 25  # Number of denoising steps
  guidance_scale = 7.5  # Scale for classifier-free guidance
  generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise
  batch_size = len(prompt)

  # Tokenize the text and generate the embeddings from the prompt:
  text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
  with torch.no_grad():
      text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

  # Generate the unconditional text embeddings which are the embeddings for the padding token.
  # These need to have the same shape (batch_size and seq_length) as the conditional text_embeddings:
  max_length = text_input.input_ids.shape[-1]
  uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
  uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

  # concatenate the conditional and unconditional embeddings into a batch to avoid doing two forward passes:
  text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

  """ Create random noise """
  # Next, generate some initial random noise as a starting point for the diffusion process.
  # This is the latent representation of the image, and it’ll be gradually denoised.
  # At this point, the latent image is smaller than the final image size but that’s okay though
  # because the model will transform it into the final 512x512 image dimensions later.
  # The height and width are divided by 8 because the vae model has 3 down-sampling layers. You can check by running the following:
  # 2 ** (len(vae.config.block_out_channels) - 1) == 8
  latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8), generator=generator)
  latents = latents.to(device)

  """ Denoise the image """
  # Start by scaling the input with the initial noise distribution, sigma, the noise scale value,
  # which is required for improved schedulers like UniPCMultistepScheduler.
  # The last step is to create the denoising loop that’ll progressively transform the pure noise in latents
  # to an image described by your prompt. Remember, the denoising loop needs to do three things:
  # 1. Set the scheduler’s timesteps to use during denoising.
  # 2. Iterate over the timesteps.
  # 3. At each timestep, call the UNet model to predict the noise residual and pass it to the scheduler to compute the previous noisy sample.
  latents = latents * scheduler.init_noise_sigma
  scheduler.set_timesteps(num_inference_steps)
  for t in tqdm(scheduler.timesteps):
      # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
      latent_model_input = torch.cat([latents] * 2)
      latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

      # predict the noise residual
      with torch.no_grad():
          noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

      # perform guidance
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

      # compute the previous noisy sample x_t -> x_t-1
      latents = scheduler.step(noise_pred, t, latents).prev_sample

  """ Decode the image """
  # The final step is to use the vae to decode the latent representation into an image and get the decoded output with sample.
  # scale and decode the image latents with vae
  latents = 1 / 0.18215 * latents
  with torch.no_grad():
      image = vae.decode(latents).sample

  # Lastly, convert the image to a PIL.Image to see your generated image!
  image = (image / 2 + 0.5).clamp(0, 1).squeeze()
  image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
  images = (image * 255).round().astype("uint8")
  image = Image.fromarray(image)
  image
  ```
***

# DDPM
  ```py
  class DenoiseDiffusion:
      def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
          super().__init__()
          self.eps_model = eps_model

          # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
          self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

          # $\alpha_t = 1 - \beta_t$
          self.alpha = 1. - self.beta
          # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
          self.alpha_bar = torch.cumprod(self.alpha, dim=0)
          # $T$
          self.n_steps = n_steps
          # $\sigma^2 = \beta$
          self.sigma2 = self.beta

      def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
          # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
          mean = gather(self.alpha_bar, t) ** 0.5 * x0
          # $(1-\bar\alpha_t) \mathbf{I}$
          var = 1 - gather(self.alpha_bar, t)
          #
          return mean, var

      def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
          # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
          if eps is None:
              eps = torch.randn_like(x0)

          # get $q(x_t|x_0)$
          mean, var = self.q_xt_x0(x0, t)
          # Sample from $q(x_t|x_0)$
          return mean + (var ** 0.5) * eps

      def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
          # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
          eps_theta = self.eps_model(xt, t)
          # [gather](utils.html) $\bar\alpha_t$
          alpha_bar = gather(self.alpha_bar, t)
          # $\alpha_t$
          alpha = gather(self.alpha, t)
          # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
          eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
          # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
          #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
          mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
          # $\sigma^2$
          var = gather(self.sigma2, t)

          # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
          eps = torch.randn(xt.shape, device=xt.device)
          # Sample
          return mean + (var ** .5) * eps

      def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
          # Get batch size
          batch_size = x0.shape[0]
          # Get random $t$ for each sample in the batch
          t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

          # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
          if noise is None:
              noise = torch.randn_like(x0)

          # Sample $x_t$ for $q(x_t|x_0)$
          xt = self.q_sample(x0, t, eps=noise)
          # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
          eps_theta = self.eps_model(xt, t)

          # MSE loss
          return F.mse_loss(noise, eps_theta)
  ```
  ```py
  def mnist(image_size=32):
      transform = torchvision.transforms.Compose([torchvision.transforms.Resize(image_size), torchvision.transforms.ToTensor()])
      return torchvision.datasets.MNIST("datasets", train=True, download=True, transform=transform)

  class Sampler:
      def __init__(self, n_steps=50):
          self.beta = torch.linspace(0.0001, 0.02, n_steps)
          self.beta = self.beta[:, None, None, None]  # expand to calculation on batch dimension

          self.alpha = 1. - self.beta
          self.alpha_bar = torch.cumprod(self.alpha, dim=0)
          self.n_steps = n_steps
          self.sigma2 = self.beta

      def q_sample(self, x0, timestep, noise=None):
          noise = torch.randn_like(x0) if noise is None else noise
          cur_alpha = self.alpha_bar[timestep]
          # Sample from $q(x_t|x_0)$
          return cur_alpha ** 0.5 * x0 + (1 - cur_alpha) ** 0.5 * noise

      def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
          # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
          eps_theta = self.eps_model(xt, t)
          # [gather](utils.html) $\bar\alpha_t$
          alpha_bar = gather(self.alpha_bar, t)
          # $\alpha_t$
          alpha = gather(self.alpha, t)
          # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
          eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
          # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
          #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
          mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
          # $\sigma^2$
          var = gather(self.sigma2, t)

          # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
          eps = torch.randn(xt.shape, device=xt.device)
          # Sample
          return mean + (var ** .5) * eps

  self.eps_model = UNet(
      image_channels=self.image_channels, n_channels=self.n_channels, ch_mults=self.channel_multipliers, is_attn=self.is_attention,
  ).to(self.device)

  # Create [DDPM class](index.html)
  self.diffusion = DenoiseDiffusion(eps_model=self.eps_model, n_steps=self.n_steps, device=self.device)

  # Create dataloader
  self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
  # Create optimizer
  self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)

  class DenoiseDiffusionLoss:
      def __init__(self, model, n_steps=1000):
          self.sampler = Sampler(n_steps=n_steps)
          self.model = model

      def __call__(self, x0):
          timestep = torch.randint(0, self.sampler.n_steps, (x0.shape[0]))
          noise = torch.randn_like(x0)
          xt = sampler.q_sample(x0, timestep, noise)
          xt_noise = self.model(xt, timestep)
          return torch.functional.F.mse_loss(noise, xt_noise)


  for _ in monit.loop(self.epochs):
      # Train the model. Iterate through the dataset
      for data in monit.iterate('Train', self.data_loader):
          # Move data to device
          data = data.to(self.device)

          # Make the gradients zero
          self.optimizer.zero_grad()
          # Calculate loss
          loss = self.diffusion.loss(data)
          # Compute gradients
          loss.backward()
          # Take an optimization step
          self.optimizer.step()

      # Sample some images
      with torch.no_grad():
          # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
          x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size], device=self.device)

          # Remove noise for $T$ steps
          for t_ in monit.iterate('Sample', self.n_steps):
              # $t$
              t = self.n_steps - t_ - 1
              # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
              x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))
  ```
***

# StableDiffusion weights
## CLIP model
  ```py
  !GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/openai/clip-vit-large-patch14

  import torch
  from transformers import CLIPTokenizer, CLIPTextModel
  clip_model = "../learning/stable_diffusion/clip-vit-large-patch14"
  ss = torch.load('../learning/stable_diffusion/clip_model.pt')
  clip_text_embedder = CLIPTextModel.from_pretrained(clip_model, state_dict=ss)
  _ = clip_text_embedder.eval()


  import torch

  ss = torch.load('../learning/stable_diffusion/clip_model.pt')
  dd = {}
  for name in ss:
      if 'k_proj' in name:
          qq, kk, vv = ss[name.replace('.k_proj', ".q_proj")], ss[name], ss[name.replace('.k_proj',".v_proj")]
          dd[name] = torch.concat([qq, kk, vv], axis=0)
          print(f"{name = }, {qq.shape = }, {kk.shape = }, {vv.shape = }, {dd[name].shape = }")
      elif 'v_proj' in name or 'q_proj' in name or name in ['text_model.embeddings.position_ids']:
          continue
      else:
          dd[name] = ss[name]

  from keras_cv_attention_models import vit, download_and_load

  mm = vit.ViTTextLargePatch14(include_top=False, pretrained=None)
  tail_align_dict = {"attn_qkv": -1, "attn_output": -1, "mlp_dense_1": -1, "mlp_dense_2": -1}
  download_and_load.keras_reload_from_torch_model(dd, mm, tail_align_dict=tail_align_dict, tail_split_position=1, do_convert=True, do_predict=False)

  torch_out = clip_text_embedder(torch.ones([1, 77], dtype=torch.int64)).last_hidden_state
  tf_out = mm(tf.ones([1, 77]))
  np.allclose(torch_out.detach(), tf_out, atol=1e-5)
  ```
## Unet
  ```py
  sys.path.append('../learning/stable_diffusion/')
  import torch
  import unet as torch_unet
  unet_model = torch_unet.UNetModel()
  unet_model.load_state_dict(torch.load("../learning/stable_diffusion/diffusion_model.pt"))
  _ = unet_model.eval()

  from keras_cv_attention_models.stable_diffusion import stable_diffusion, unet
  from keras_cv_attention_models import download_and_load

  mm = unet.UNet(pretrained=None)

  full_name_align_dict = {"time_embed_2_dense": -2, "middle_block_attn_in_layers_conv": "middle_block_attn_in_layers_conv"}
  tail_align_dict = {"in_layers_conv": -1}
  download_and_load.keras_reload_from_torch_model(unet_model, mm, full_name_align_dict=full_name_align_dict, tail_align_dict=tail_align_dict, tail_split_position=3, do_convert=True, do_predict=False)

  tf_out = mm([tf.ones([1, 64, 64, 4]), tf.ones([1]), tf.ones([1, 77, 768])])
  torch_out = unet_model(torch.ones([1, 4, 64, 64]), torch.ones([1]), torch.ones([1, 77, 768]))
  np.allclose(torch_out.permute([0, 2, 3, 1]).detach(), tf_out, atol=1e-4)
  ```
## Decoder
  ```py
  sys.path.append('../learning/stable_diffusion/')
  import unet, autoencoder, ddim_sampler, unet_attention, torch

  decoder = autoencoder.Decoder()
  ss = torch.load("../learning/stable_diffusion/decoder_model.pt")
  decoder.load_state_dict(ss)
  _ = decoder.eval()

  pre_decoder = autoencoder.PreDecoder()
  pre_decoder.load_state_dict(torch.load("../learning/stable_diffusion/pre_decoder_model.pt"))
  _ = pre_decoder.eval()

  import kecam
  # kecam.download_and_load.keras_reload_from_torch_model(decoder)

  dd = pre_decoder.state_dict()
  keys = ['conv_in.weight', 'conv_in.bias', *[kk for kk in ss if kk.startswith("mid")],
  *[kk for kk in ss if kk.startswith("up.3")], *[kk for kk in ss if kk.startswith("up.2")],
  *[kk for kk in ss if kk.startswith("up.1")], *[kk for kk in ss if kk.startswith("up.0")],
  'norm_out.weight', 'norm_out.bias', 'conv_out.weight', 'conv_out.bias']
  dd.update({kk: ss[kk] for kk in keys})

  from keras_cv_attention_models.stable_diffusion import stable_diffusion, unet, encoder_decoder
  from keras_cv_attention_models import download_and_load

  mm = encoder_decoder.Decoder(pretrained=None)
  full_name_align_dict = {"middle_block_attn_query_conv": -1}
  download_and_load.keras_reload_from_torch_model(dd, mm, full_name_align_dict=full_name_align_dict, do_convert=True, do_predict=False)

  tf_out = mm(tf.ones([1, 64, 64, 4]))
  torch_out = decoder(pre_decoder(torch.ones([1, 4, 64, 64])))
  np.allclose(torch_out.permute([0, 2, 3, 1]).detach(), tf_out, atol=5e-2)
  ```
## Encoder
  ```py
  sys.path.append('../learning/stable_diffusion/')
  import unet, autoencoder, ddim_sampler, unet_attention, torch

  encoder = autoencoder.Encoder()
  ss = torch.load("../learning/stable_diffusion/encoder_model.pt")
  encoder.load_state_dict(ss)
  _ = encoder.eval()

  post_encoder = autoencoder.PostEncoder()
  post_encoder.load_state_dict(torch.load("../learning/stable_diffusion/post_encoder_model.pt"))
  _ = post_encoder.eval()

  ss.update(post_encoder.state_dict())

  from keras_cv_attention_models.stable_diffusion import stable_diffusion, unet, encoder_decoder
  from keras_cv_attention_models import download_and_load

  mm = encoder_decoder.Encoder(pretrained=None)
  full_name_align_dict = {"middle_block_attn_query_conv": -1}
  download_and_load.keras_reload_from_torch_model(ss, mm, full_name_align_dict=full_name_align_dict, do_convert=True, do_predict=False)

  tf_out = mm(tf.ones([1, 512, 512, 3]))
  torch_out = post_encoder.quant_conv((encoder(torch.ones([1, 3, 512, 512]))))
  np.allclose(torch_out.permute([0, 2, 3, 1]).detach(), tf_out, atol=5e-1)
  ```
## StableDiffusion
  ```py
  from keras_cv_attention_models.stable_diffusion import stable_diffusion

  mm = stable_diffusion.StableDiffusion()
  mm.clip_model.load_weights('vit_text_large_patch14_clip.h5')
  mm.unet_model.load_weights('unet_imagenet.h5')
  mm.decoder_model.load_weights('decoder_imagenet.h5')

  imm = mm.text_to_image('a photo of an astonaut riding a horse on mars', batch_size=1).numpy()
  print(f"{imm.shape = }, {imm.min() = }, {imm.max() = }")
  plt.imsave('aa.jpg', np.clip(imm / 2 + 0.5, 0, 1)[0])
  ```
***

# Train
## Cifar10
- [The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
```py
import pickle
from tqdm import tqdm
for id in range(1, 6):
    with open('cifar-10-batches-py/data_batch_{}'.format(id), 'rb') as ff:
        aa = pickle.load(ff, encoding='bytes')

    for label, data, filename in tqdm(zip(aa[b'labels'], aa[b'data'], aa[b'filenames'])):
        filename = os.path.join('cifar10', 'train', str(label), filename.decode())
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        plt.imsave(filename, data.reshape(3, 32, 32).transpose([1, 2, 0]))

with open('cifar-10-batches-py/test_batch', 'rb') as ff:
    aa = pickle.load(ff, encoding='bytes')

for label, data, filename in tqdm(zip(aa[b'labels'], aa[b'data'], aa[b'filenames'])):
    filename = os.path.join('cifar10', 'test', str(label), filename.decode())
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plt.imsave(filename, data.reshape(3, 32, 32).transpose([1, 2, 0]))
```
