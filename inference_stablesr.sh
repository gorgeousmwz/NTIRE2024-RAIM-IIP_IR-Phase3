
cd StableSR

python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
--config configs/stableSRNew/v2-finetune_text_T_512.yaml \
--ckpt ../pretrained/stablesr_turbo.ckpt \
--vqgan_ckpt ../pretrained/vqgan_cfw_00011.ckpt \
--init-img ../HAT/results/HAT_SRx2/visualization \
--outdir ../outputs/NTIRE2024-RAIM-IIP_IR  \
--ddpm_steps 4 \
--dec_w 1 \
--upscale 1 \
--colorfix_type adain \
--vqgantile_size 1024 \
--vqgantile_stride 512

cd ..