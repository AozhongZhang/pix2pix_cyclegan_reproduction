import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
import sys
import torchvision
from torchvision.datasets import *
import torchvision.transforms as transforms
from discriminator_model import Pix2pix_Discriminator, Cycle_Discriminator
from generator_model_myself import Pix2pix_Generator, Cycle_Generator
from config_myself import configurations
from dataloader_myself import HorseZebraDataset, Cycle_transforms
from tqdm import tqdm
from utils_myself import save_some_examples, save_checkpoint, load_checkpoint
from train_function import Train_Function_CycleGAN

if __name__ == '__main__':

    # ======= hyperparameters & data loaders =======#
    cfg = configurations[2]
    DEVICE = cfg['DEVICE']
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = cfg['BATCH_SIZE']
    NUM_WORKERS = cfg['NUM_WORKERS']
    TRAIN_DIR = cfg['TRAIN_DIR']
    VAL_DIR = cfg['VAL_DIR']
    G_H_NAME = cfg['G_H_NAME']
    G_Z_NAME = cfg['G_Z_NAME']
    D_H_NAME = cfg['D_H_NAME']
    D_Z_NAME = cfg['D_Z_NAME']
    LOSS_NAME = cfg['LOSS_NAME']
    OPTIMIZER_D = cfg['OPTIMIZER_D']
    OPTIMIZER_G = cfg['OPTIMIZER_G']
    LOAD_MODEL = cfg['LOAD_MODEL']
    SAVE_MODEL = cfg['SAVE_MODEL']
    CHECKPOINT_GEN_H = cfg['CHECKPOINT_GEN_H']
    CHECKPOINT_GEN_Z = cfg['CHECKPOINT_GEN_Z']
    CHECKPOINT_CRITIC_H = cfg['CHECKPOINT_CRITIC_H']
    CHECKPOINT_CRITIC_Z = cfg['CHECKPOINT_CRITIC_Z']
    LR = cfg['LR']
    NUM_EPOCHS = cfg['NUM_EPOCHS']
    LAMBDA_CYCLE = cfg['LAMBDA_CYCLE']
    LAMBDA_IDENTITY = cfg['LAMBDA_IDENTITY']

    # Data loader
    train_dataset = HorseZebraDataset(
        root_horse=TRAIN_DIR+"/trainA", 
        root_zebra=TRAIN_DIR+"/trainB", 
        transform=Cycle_transforms,
        )

    val_dataset = HorseZebraDataset(
        root_horse=VAL_DIR+"/testA", 
        root_zebra=VAL_DIR+"/testB", 
        transform=Cycle_transforms,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )


    # Generator
    G_DICT = {
              'Pix2pix_Generator':Pix2pix_Generator(in_channels=3, feature=64),
              'Cycle_Generator':Cycle_Generator(img_channels=3, num_residuals=9),
            }

    G_H = G_DICT[G_H_NAME]
    G_Z = G_DICT[G_Z_NAME]
    G_H = G_H.to(DEVICE)
    G_Z = G_Z.to(DEVICE)

    # Discrimator
    D_DICT = {
              'Pix2pix_Discriminator':Pix2pix_Discriminator(in_channels=3),
              'Cycle_Discriminator':Cycle_Discriminator(in_channels=3),
            }
    
    D_H = D_DICT[D_H_NAME]
    D_Z = D_DICT[D_Z_NAME]
    D_H = D_H.to(DEVICE)
    D_Z = D_Z.to(DEVICE)
    
    # LOSS
    LOSS_DICT = {
              'BCEWithLogitsLoss':nn.BCEWithLogitsLoss(),
              'BCE':nn.MSELoss(),
            }
    
    LOSS = LOSS_DICT[LOSS_NAME]
    L1_LOSS = nn.L1Loss()

    # OPTIMIZER
    OPTIMIZER_G_DICT = {
        # 'Adam_pix2pix':torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999)),
        'Adam_cyclegan':torch.optim.Adam(
            list(G_Z.parameters()) + list(G_H.parameters()),
            lr=LR,
            betas=(0.5, 0.999),
        )
    }
    OPTIMIZER_D_DICT = {
        # 'Adam_pix2pix':torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999)),
        'Adam_cyclegan':torch.optim.Adam(
            list(D_H.parameters()) + list(D_Z.parameters()),
            lr=LR,
            betas=(0.5, 0.999),
        )
    }

    D_optimizer = OPTIMIZER_D_DICT[OPTIMIZER_D]
    G_optimizer = OPTIMIZER_G_DICT[OPTIMIZER_G]

    G_scaler = torch.cuda.amp.GradScaler()
    D_scaler = torch.cuda.amp.GradScaler()

    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT_GEN_H, G_H, G_optimizer, LR)
        load_checkpoint(CHECKPOINT_GEN_Z, G_Z, G_optimizer, LR)
        load_checkpoint(CHECKPOINT_CRITIC_H, D_H, D_optimizer, LR)
        load_checkpoint(CHECKPOINT_CRITIC_Z, D_Z, D_optimizer, LR)

    for epoch in range(NUM_EPOCHS):
        Train_Function_CycleGAN(
            D_H, D_Z, G_H, G_Z, 
            train_loader, D_optimizer, G_optimizer, L1_LOSS, LOSS, 
            D_scaler, G_scaler, LAMBDA_CYCLE, LAMBDA_IDENTITY, epoch
            )

        if SAVE_MODEL:
            save_checkpoint(G_H, G_optimizer, filename=CHECKPOINT_GEN_H)
            save_checkpoint(G_Z, G_optimizer, filename=CHECKPOINT_GEN_Z)
            save_checkpoint(D_H, D_optimizer, filename=CHECKPOINT_CRITIC_H)
            save_checkpoint(D_Z, D_optimizer, filename=CHECKPOINT_CRITIC_Z)
