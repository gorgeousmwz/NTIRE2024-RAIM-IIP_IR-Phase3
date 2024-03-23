import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import logging
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from arch.bpnkernel_arch import BPNKernel
from arch.fftformer_arch import fftformer
import shutil


def check_path(args):
    assert os.path.exists(args.input_folder), f'Error: input_folder ({args.input_folder}) is not found.'
    assert os.path.exists(args.kernel_model_path), f'Error: kernel_model_path ({args.kernel_model_path}) is not found.'
    assert os.path.exists(args.restore_model_path), f'Error: restore_model_path ({args.restore_model_path}) is not found.'

class GetLog:
    def __init__(self, log_abs_path):
        self.log_abs_path = log_abs_path
        self.logger = None

    def log(self, name):
        self.logger = logging.getLogger(name=name)
        fh = logging.FileHandler(self.log_abs_path)
        ch = logging.StreamHandler()
        fm = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(message)s")
        fh.setFormatter(fm)
        ch.setFormatter(fm)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.DEBUG)
        return self.logger

def init(information,args):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    args.result_path=os.path.join(args.output_folder,'deblur')
    os.makedirs(args.result_path,exist_ok=True)

    log_path=os.path.join(args.output_folder,'log.log')
    get_log=GetLog(log_path)
    logger=get_log.log(name=f"{information['competition_name']}-{information['team_name']}")
    meta=json.dumps(information, indent=4)
    config=json.dumps(vars(args), indent=4)
    logger.info("Information: %s", meta)
    logger.info("Configs: %s", config)
    return information,args,logger

def deblur(information,args,logger):
    logger.info(f"[Deblur] Build model.")
    kernel_model = BPNKernel()
    restore_model = fftformer(dim=32)

    logger.info(f"[Deblur] Loading params.")
    kernel_model.load_state_dict(torch.load(args.kernel_model_path)["params"], strict=True)
    restore_model.load_state_dict(torch.load(args.restore_model_path)["params"], strict=True)

    kernel_model = kernel_model.cuda()
    restore_model = restore_model.cuda()

    logger.info(f"[Deblur] Testing.")
    deblur_imgs={}
    with torch.no_grad():
        for img_name in os.listdir(args.input_folder):
            img = cv2.imread(os.path.join(args.input_folder, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img = img.cuda()
            kernel = kernel_model(img)
            deblur_img = restore_model(img, kernel)
            deblur_img = deblur_img.clamp(0, 1).squeeze().permute(1, 2, 0).cpu().numpy()
            deblur_img = cv2.cvtColor(deblur_img, cv2.COLOR_RGB2BGR)
            deblur_img = (deblur_img * 255).astype("uint8")
            deblur_imgs[img_name] = deblur_img
            cv2.imwrite(os.path.join(args.result_path, img_name), deblur_img)
            logger.info(f"[Deblur] {img_name} done.")
    return deblur_imgs

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="inputs/PhaseThreeData")
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument("--kernel_model_path", type=str, default="pretrained/kernel_model.pth")
    parser.add_argument("--restore_model_path", type=str, default="pretrained/restore_model.pth")
    args = parser.parse_args()

    check_path(args=args) # check path validity

    information={
        'competition_name':'NTIRE2024-RAIM',
        'competition_full_name':'Restore Any Image Model (RAIM) in the Wild: An NTIRE Challenge in Conjunction with CVPR 2024',
        'team_name':'IIP_IR',
        'team_member':['Wanjie Sun','Zhenyu Hu','Jingyun Liu','Wenzhuo Ma','Ce Wang','Hanyou Zheng','Zhenzhong Chen'],
        'test_time':datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    }

    information,args,logger=init(information,args)
    logger.info(f"{information['competition_name']}-{information['team_name']} test starts!")
    # Step 1: Deblur
    logger.info(f"[Deblur] Start! ")
    deblur_imgs=deblur(information,args,logger)
    logger.info(f"[Deblur] Done! ")

 