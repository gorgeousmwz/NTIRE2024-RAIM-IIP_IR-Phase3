"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from basicsr.metrics import calculate_niqe
import math
import copy
import torch.nn.functional as F
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
from util_image import ImageSpliterTh
from pathlib import Path

def calc_mean_std(feat, eps=1e-5):
	"""Calculate mean and std for adaptive_instance_normalization.
	Args:
		feat (Tensor): 4D tensor.
		eps (float): A small value added to the variance to avoid
			divide-by-zero. Default: 1e-5.
	"""
	size = feat.size()
	assert len(size) == 4, 'The input feature should be 4D tensor.'
	b, c = size[:2]
	feat_var = feat.view(b, c, -1).var(dim=2) + eps
	feat_std = feat_var.sqrt().view(b, c, 1, 1)
	feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
	return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
	"""Adaptive instance normalization.
	Adjust the reference features to have the similar color and illuminations
	as those in the degradate features.
	Args:
		content_feat (Tensor): The reference feature.
		style_feat (Tensor): The degradate features.
	"""
	size = content_feat.size()
	style_mean, style_std = calc_mean_std(style_feat)
	content_mean, content_std = calc_mean_std(content_feat)
	normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
	return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def space_timesteps(num_timesteps, section_counts):
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.

	For example, if there's 300 timesteps and the section counts are [10,15,20]
	then the first 100 timesteps are strided to be 10 timesteps, the second 100
	are strided to be 15 timesteps, and the final 100 are strided to be 20.

	If the stride is a string starting with "ddim", then the fixed striding
	from the DDIM paper is used, and only one section is allowed.

	:param num_timesteps: the number of diffusion steps in the original
						  process to divide up.
	:param section_counts: either a list of numbers, or a string containing
						   comma-separated numbers, indicating the step count
						   per section. As a special case, use "ddimN" where N
						   is a number of steps to use the striding from the
						   DDIM paper.
	:return: a set of diffusion steps from the original process to use.
	"""
	if isinstance(section_counts, str):
		if section_counts.startswith("ddim"):
			desired_count = int(section_counts[len("ddim"):])
			for i in range(1, num_timesteps):
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(
				f"cannot create exactly {num_timesteps} steps with an integer stride"
			)
		section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
	size_per = num_timesteps // len(section_counts)
	extra = num_timesteps % len(section_counts)
	start_idx = 0
	all_steps = []
	for i, section_count in enumerate(section_counts):
		size = size_per + (1 if i < extra else 0)
		if size < section_count:
			raise ValueError(
				f"cannot divide section of {size} steps into {section_count}"
			)
		if section_count <= 1:
			frac_stride = 1
		else:
			frac_stride = (size - 1) / (section_count - 1)
		cur_idx = 0.0
		taken_steps = []
		for _ in range(section_count):
			taken_steps.append(start_idx + round(cur_idx))
			cur_idx += frac_stride
		all_steps += taken_steps
		start_idx += size
	return set(all_steps)

def chunk(it, size):
	it = iter(it)
	return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	print('>>>>>>>>>>>>>>>>>>>load results>>>>>>>>>>>>>>>>>>>>>>>')
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model.cuda()
	model.eval()
	return model

def load_img(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size
	print(f"loaded input image of size ({w}, {h}) from {path}")
	w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.

def read_image(im_path):
	im = np.array(Image.open(im_path).convert("RGB"))
	im = im.astype(np.float32)/255.0
	im = im[None].transpose(0,3,1,2)
	im = (torch.from_numpy(im) - 0.5) / 0.5

	return im.cuda()

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--init-img",
		type=str,
		nargs="?",
		help="path to the input image",
		default="inputs/user_upload",
	)
	parser.add_argument(
		"--outdir",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="outputs/user_upload",
	)
	parser.add_argument(
		"--C",
		type=int,
		default=4,
		help="latent channels",
	)
	parser.add_argument(
		"--f",
		type=int,
		default=8,
		help="downsampling factor, most often 8 or 16",
	)
	parser.add_argument(
		"--n_samples",
		type=int,
		default=2,
		help="how many samples to produce for each given prompt. A.k.a batch size",
	)
	parser.add_argument(
		"--config",
		type=str,
		default="configs/stableSRNew/v2-finetune_text_T_512.yaml",
		help="path to config which constructs model",
	)
	parser.add_argument(
		"--ckpt",
		type=str,
		default="models/ldm/stable-diffusion-v1/model.ckpt",
		help="path to checkpoint of model",
	)
	parser.add_argument(
		"--vqgan_ckpt",
		type=str,
		default="models/ldm/stable-diffusion-v1/epoch=000011.ckpt",
		help="path to checkpoint of VQGAN model",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="the seed (for reproducible sampling)",
	)
	parser.add_argument(
		"--precision",
		type=str,
		help="evaluate at this precision",
		choices=["full", "autocast"],
		default="autocast"
	)
	parser.add_argument(
		"--input_size",
		type=int,
		default=512,
		help="input size",
	)
	parser.add_argument(
		"--upscale",
		type=float,
		default=4.0,
		help="upsample scale",
	)
	parser.add_argument(
		"--dec_w",
		type=float,
		default=0.5,
		help="weight for combining VQGAN and Diffusion",
	)
	parser.add_argument(
		"--colorfix_type",
		type=str,
		default="nofix",
		help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
	)
	parser.add_argument(
		"--ddim_steps",
		type=int,
		default=50,
		help="number of ddim sampling steps",
	)
	parser.add_argument(
		"--ddim_eta",
		type=float,
		default=0.0,
		help="ddim eta (eta=0.0 corresponds to deterministic sampling",
	)
	parser.add_argument(
		"--scale",
		type=float,
		default=7.0,
		help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
	)
	parser.add_argument(
		"--strength",
		type=float,
		default=0.75,
		help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
	)
	parser.add_argument(
		"--use_negative_prompt",
		action='store_true',
		help="if enabled, save inputs",
	)
	parser.add_argument(
		"--use_posi_prompt",
		action='store_true',
		help="if enabled, save inputs",
	)
	parser.add_argument(
		"--vqgantile_stride",
		type=int,
		default=1000,
		help="the stride for tile operation before VQGAN decoder (in pixel)",
	)
	parser.add_argument(
		"--vqgantile_size",
		type=int,
		default=1280,
		help="the size for tile operation before VQGAN decoder (in pixel)",
	)
	parser.add_argument(
		"--tile_overlap",
		type=int,
		default=32,
		help="tile overlap size (in latent)",
	)

	opt = parser.parse_args()
	seed_everything(opt.seed)

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(opt.input_size),
		torchvision.transforms.CenterCrop(opt.input_size),
	])

	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.ckpt}")
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model = model.to(device)

	vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
	vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
	vq_model = vq_model.to(device)
	vq_model.decoder.fusion_w = opt.dec_w

	sampler = DDIMSampler(model)

	sampler.configs = config

	os.makedirs(opt.outdir, exist_ok=True)
	outpath = opt.outdir

	batch_size = opt.n_samples

	sample_path = os.path.join(outpath, "samples")
	os.makedirs(sample_path, exist_ok=True)
	input_path = os.path.join(outpath, "inputs")
	os.makedirs(input_path, exist_ok=True)

	images_path_ori = sorted(glob.glob(os.path.join(opt.init_img, "*")))
	images_path = copy.deepcopy(images_path_ori)
	for item in images_path_ori:
		img_name = item.split('/')[-1]
		if os.path.exists(os.path.join(outpath, img_name)):
			images_path.remove(item)
	print(f"Found {len(images_path)} inputs.")

	model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
						  linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
	model.num_timesteps = 1000
	model = model.to(device)

	ddim_timesteps = set(space_timesteps(1000, [opt.ddim_steps]))
	ddim_timesteps = list(ddim_timesteps)
	ddim_timesteps.sort()

	sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

	precision_scope = autocast if opt.precision == "autocast" else nullcontext
	niqe_list = []
	with torch.no_grad():
		with model.ema_scope():
			tic = time.time()
			for n in trange(len(images_path), desc="Sampling"):
				if (n + 1) % opt.n_samples == 1 or opt.n_samples == 1:
					cur_image = read_image(images_path[n])
					size_min = min(cur_image.size(-1), cur_image.size(-2))
					upsample_scale = max(opt.input_size/size_min, opt.upscale)
					cur_image = F.interpolate(
								cur_image,
								size=(int(cur_image.size(-2)*upsample_scale),
									  int(cur_image.size(-1)*upsample_scale)),
								mode='bicubic',
								)
					cur_image = cur_image.clamp(-1, 1)
					im_lq_bs = [cur_image, ]  # 1 x c x h x w, [-1, 1]
					im_path_bs = [images_path[n], ]
				else:
					cur_image = read_image(images_path[n])
					size_min = min(cur_image.size(-1), cur_image.size(-2))
					upsample_scale = max(opt.input_size/size_min, opt.upscale)
					cur_image = F.interpolate(
								cur_image,
								size=(int(cur_image.size(-2)*upsample_scale),
									  int(cur_image.size(-1)*upsample_scale)),
								mode='bicubic',
								)
					cur_image = cur_image.clamp(-1, 1)
					im_lq_bs.append(cur_image) # 1 x c x h x w, [-1, 1]
					im_path_bs.append(images_path[n]) # 1 x c x h x w, [-1, 1]

				if (n + 1) % opt.n_samples == 0 or (n+1) == len(images_path):
					im_lq_bs = torch.cat(im_lq_bs, dim=0)
					ori_h, ori_w = im_lq_bs.shape[2:]
					ref_patch=None
					if not (ori_h % 32 == 0 and ori_w % 32 == 0):
						flag_pad = True
						pad_h = ((ori_h // 32) + 1) * 32 - ori_h
						pad_w = ((ori_w // 32) + 1) * 32 - ori_w
						im_lq_bs = F.pad(im_lq_bs, pad=(0, pad_w, 0, pad_h), mode='reflect')
					else:
						flag_pad = False

					if im_lq_bs.shape[2] > opt.vqgantile_size or im_lq_bs.shape[3] > opt.vqgantile_size:
						im_spliter = ImageSpliterTh(im_lq_bs, opt.vqgantile_size, opt.vqgantile_stride, sf=1)
						for im_lq_pch, index_infos in im_spliter:
							seed_everything(opt.seed)
							init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_pch))  # move to latent space
							if opt.use_posi_prompt:
								text_init = ['(masterpiece:2), (best quality:2), (realistic:2), (very clear:2)']*im_lq_pch.size(0)
								# text_init = ['Good photo.']*im_lq_pch.size(0)
							else:
								text_init = ['']*im_lq_pch.size(0)

							semantic_c = model.cond_stage_model(text_init)
							if opt.use_negative_prompt:
								negative_text_init = ['3d, cartoon, anime, sketches, (worst quality:2), (low quality:2)']*im_lq_pch.size(0)
								# negative_text_init = ['Bad photo.']*im_lq_pch.size(0)
								nega_semantic_c = model.cond_stage_model(negative_text_init)

							noise = torch.randn_like(init_latent)
							t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_pch.size(0))
							t = t.to(device).long()
							x_T = model.q_sample(x_start=init_latent, t=t, noise=noise)
							x_T = None

							samples, _ = sampler.ddim_sampling_sr_t_canvas(cond=semantic_c,
															 struct_cond=init_latent,
															 shape=init_latent.shape,
															 unconditional_conditioning=nega_semantic_c if opt.use_negative_prompt else None,
															 unconditional_guidance_scale=opt.scale if opt.use_negative_prompt else None,
															 timesteps=np.array(ddim_timesteps),
															 x_T=x_T,
															 tile_size=opt.input_size//8,
															 tile_overlap=opt.tile_overlap,
															 batch_size=opt.n_samples)
							_, enc_fea_lq = vq_model.encode(im_lq_pch)
							x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
							if opt.colorfix_type == 'adain':
								x_samples = adaptive_instance_normalization(x_samples, im_lq_pch)
							elif opt.colorfix_type == 'wavelet':
								x_samples = wavelet_reconstruction(x_samples, im_lq_pch)
							im_spliter.update_gaussian(x_samples, index_infos)
						im_sr = im_spliter.gather()
						im_sr = torch.clamp((im_sr+1.0)/2.0, min=0.0, max=1.0)
					else:
						init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_bs))  # move to latent space
						if opt.use_posi_prompt:
							text_init = ['(masterpiece:2), (best quality:2), (realistic:2), (very clear:2)']*im_lq_bs.size(0)
							# text_init = ['Good photo.']*im_lq_bs.size(0)
						else:
							text_init = ['']*im_lq_bs.size(0)

						semantic_c = model.cond_stage_model(text_init)
						if opt.use_negative_prompt:
							negative_text_init = ['3d, cartoon, anime, sketches, (worst quality:2), (low quality:2)']*im_lq_bs.size(0)
							# negative_text_init = ['Bad photo.']*im_lq_bs.size(0)
							nega_semantic_c = model.cond_stage_model(negative_text_init)
						noise = torch.randn_like(init_latent)
						# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
						t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_bs.size(0))
						t = t.to(device).long()
						x_T = model.q_sample(x_start=init_latent, t=t, noise=noise)
						x_T = None
						samples, _ = sampler.ddim_sampling_sr_t_canvas(cond=semantic_c,
														 struct_cond=init_latent,
														 shape=init_latent.shape,
														 unconditional_conditioning=nega_semantic_c if opt.use_negative_prompt else None,
														 unconditional_guidance_scale=opt.scale if opt.use_negative_prompt else None,
														 timesteps=np.array(ddim_timesteps),
														 x_T=x_T,
														 tile_size=opt.input_size//8,
														 tile_overlap=opt.tile_overlap,
														 batch_size=opt.n_samples)
						_, enc_fea_lq = vq_model.encode(im_lq_bs)
						x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
						if opt.colorfix_type == 'adain':
							x_samples = adaptive_instance_normalization(x_samples, im_lq_bs)
						elif opt.colorfix_type == 'wavelet':
							x_samples = wavelet_reconstruction(x_samples, im_lq_bs)
						im_sr = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0)

					if upsample_scale > opt.upscale:
						im_sr = F.interpolate(
									im_sr,
									size=(int(im_lq_bs.size(-2)*opt.upscale/upsample_scale),
										  int(im_lq_bs.size(-1)*opt.upscale/upsample_scale)),
									mode='bicubic',
									)
						im_sr = torch.clamp(im_sr, min=0.0, max=1.0)

					im_sr = im_sr.cpu().numpy().transpose(0,2,3,1)*255   # b x h x w x c

					if flag_pad:
						im_sr = im_sr[:, :ori_h, :ori_w, ]

					for jj in range(im_lq_bs.shape[0]):
						img_name = str(Path(im_path_bs[jj]).name)
						basename = os.path.splitext(os.path.basename(img_name))[0]
						outpath = str(Path(opt.outdir)) + '/' + basename + '.png'
						Image.fromarray(im_sr[jj, ].astype(np.uint8)).save(outpath)

			toc = time.time()

	print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
		  f" \nEnjoy.")


if __name__ == "__main__":
	main()
