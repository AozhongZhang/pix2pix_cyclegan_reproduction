import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
import torchvision
from tqdm import tqdm
from config_myself import configurations

cfg = configurations[1]
DEVICE = cfg['DEVICE']

def Train_Function_Pix2pix(D, G, Data_loader, Optimizer_D, Optimizer_G, L1_loss, LOSS, G_scaler, D_scaler, L1_LAMBDA):
    
    loop = tqdm(Data_loader, leave=True)
    
    for idx, (x, y) in enumerate(loop):
        
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Train D
        with torch.cuda.amp.autocast():
            y_fake = G(x) #此处直接让x作为噪声输入

            D_real = D(x, y)

            Loss_D_real = LOSS(D_real, torch.ones_like(D_real))

            D_fake = D(x, y_fake.detach())
                
            Loss_D_fake = LOSS(D_fake, torch.zeros_like(D_fake))
            
            Loss_D = (Loss_D_real + Loss_D_fake) / 2
        
        D.zero_grad()  #zero or 累加进下面的D梯度
        D_scaler.scale(Loss_D).backward()
        D_scaler.step(Optimizer_D)
        D_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            D_fake = D(x, y_fake)
                
            G_fake_loss = LOSS(D_fake, torch.ones_like(D_fake))
                
            L1 = L1_loss(y_fake, y) * L1_LAMBDA

            Loss_G = G_fake_loss + L1

        Optimizer_G.zero_grad()
        G_scaler.scale(Loss_G).backward()
        G_scaler.step(Optimizer_G)
        G_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

def Train_Function_CycleGAN(D_H, D_Z, G_H, G_Z, Data_loader, Optimizer_D, Optimizer_G, L1_Loss, LOSS, D_scaler, G_scaler, LAMBDA_CYCLE, LAMBDA_IDENTITY, epoch):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(Data_loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(DEVICE)
        horse = horse.to(DEVICE)

        #Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = G_H(zebra)
            D_H_real = D_H(horse)
            D_H_fake = D_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = LOSS(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = LOSS(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = G_Z(horse)
            D_Z_real = D_Z(zebra)
            D_Z_fake = D_Z(fake_zebra.detach())
            D_Z_real_loss = LOSS(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = LOSS(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            D_loss = (D_H_loss + D_Z_loss) / 2

        Optimizer_D.zero_grad()
        D_scaler.scale(D_loss).backward()
        D_scaler.step(Optimizer_D)
        D_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = D_H(fake_horse)
            D_Z_fake = D_Z(fake_zebra)
            loss_G_H = LOSS(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = LOSS(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = G_Z(fake_horse)
            cycle_horse = G_H(fake_zebra)
            cycle_zebra_loss = L1_Loss(zebra, cycle_zebra)
            cycle_horse_loss = L1_Loss(horse, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = G_Z(zebra)
            identity_horse = G_H(horse)
            identity_zebra_loss = L1_Loss(zebra, identity_zebra)
            identity_horse_loss = L1_Loss(horse, identity_horse)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * LAMBDA_CYCLE
                + cycle_horse_loss * LAMBDA_CYCLE
                + identity_horse_loss * LAMBDA_IDENTITY
                + identity_zebra_loss * LAMBDA_IDENTITY
            )

        Optimizer_G.zero_grad()
        G_scaler.scale(G_loss).backward()
        G_scaler.step(Optimizer_G)
        G_scaler.update()

        if idx % 1300 == 0:
            save_image(fake_horse*0.5+0.5, f"cycle_horse_zebra/Epoch_{epoch}_fake_horse_{idx}.png")
            save_image(fake_zebra*0.5+0.5, f"cycle_horse_zebra/Epoch_{epoch}_fake_zebra_{idx}.png")
            save_image(horse*0.5+0.5, f"cycle_horse_zebra/Epoch_{epoch}_input_horse_{idx}.png")
            save_image(zebra*0.5+0.5, f"cycle_horse_zebra/Epoch_{epoch}_input_zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))