import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
import torchvision
# import torchvision.datasets as datasets
from torchvision.datasets import *
import torchvision.transforms as transforms
# from torch.utils.data import *
# from DCGAN.train import Loss_G
from discriminator_model import Pix2pix_Discriminator
from generator_model_myself import Pix2pix_Generator
from config_myself import configurations
from dataloader_myself import MapDataset
from tqdm import tqdm
# from module import *
from utils_myself import save_some_examples, save_checkpoint, load_checkpoint
from train_function import Train_Function_Pix2pix

        

if __name__ == '__main__':

    # ======= hyperparameters & data loaders =======#
    cfg = configurations[1]
    TRAIN_DIR = cfg['TRAIN_DIR']
    VAL_DIR = cfg['VAL_DIR']
    DEVICE = cfg['DEVICE']
    NUM_EPOCH = cfg['NUM_EPOCHS']
    BATCH_SIZE = cfg['BATCH_SIZE']
    LR = cfg['LR']
    IMAGE_SIZE = cfg['IMAGE_SIZE']
    IMAGE_CHANNELS = cfg['CHANNELS_IMG']
    L1_LAMBDA = cfg['L1_LAMBDA']
    NUM_WORKERS = cfg['NUM_WORKERS']
    LOAD_CHECKPOINT = cfg['LOAD_CHECKPOINT']
    SAVE_MODEL = cfg['SAVE_MODEL']
    CHECKPOINT_DISC = cfg['CHECKPOINT_DISC']
    CHECKPOINT_GEN = cfg['CHECKPOINT_GEN']
    G_NAME = cfg['G_NAME']
    D_NAME = cfg['D_NAME']
    LOSS_NAME = cfg['LOSS_NAME']
    OPTIMIZER_G = cfg['OPTIMIZER_G']
    OPTIMIZER_D = cfg['OPTIMIZER_D']
    SAVE_IMAGE = cfg['SAVE_IMAGE']
    

    train_set = MapDataset(TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )

    G_DICT = {
              'Pix2pix_Generator':Pix2pix_Generator(in_channels=3, feature=64),
            }

    D_DICT = {'Pix2pix_Discriminator':Pix2pix_Discriminator(in_channels=3),}
    
    G = G_DICT[G_NAME]
    G = G.to(DEVICE)
    D = D_DICT[D_NAME]
    D = D.to(DEVICE)
    
    LOSS_DICT = {
        'BCEWithLogitsLoss':nn.BCEWithLogitsLoss(),
    }
    
    LOSS = LOSS_DICT[LOSS_NAME]
    L1_LOSS = nn.L1Loss()


    OPTIMIZER_G_DICT = {
        'Adam':torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    }
    OPTIMIZER_D_DICT = {
        'Adam':torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    }

    D_optimizer = OPTIMIZER_D_DICT[OPTIMIZER_D]
    G_optimizer = OPTIMIZER_G_DICT[OPTIMIZER_G]


    G_scaler = torch.cuda.amp.GradScaler()
    D_scaler = torch.cuda.amp.GradScaler()

    val_dataset = MapDataset(root_dir=VAL_DIR)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # if LOAD_CHECKPOINT:
    #     print("=> Loading checkpoint")
    #     checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    #     model.load_state_dict(checkpoint["state_dict"])
    #     optimizer.loader_state_dict(checkpoint["optimizer"])

    if LOAD_CHECKPOINT:
        load_checkpoint(
            CHECKPOINT_GEN, G, G_optimizer, LR,
        )
        load_checkpoint(
            CHECKPOINT_DISC, D, D_optimizer, LR,
        )
        print("Loading checkpoint...")
        print("="*60)

    for epoch in range(NUM_EPOCH):
        
        D.train()
        G.train()

        Train_Function_Pix2pix(D, G, train_loader, D_optimizer, G_optimizer, L1_LOSS, LOSS, G_scaler, D_scaler, L1_LAMBDA)

        if SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, CHECKPOINT_GEN, epoch)
            save_checkpoint(disc, opt_disc, CHECKPOINT_DISC, epoch)

        save_some_examples(G, val_loader, epoch, SAVE_IMAGE)
