import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

configurations = {
    1: dict(
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu",
        # TRAIN_DIR = "data/maps/train",
        # VAL_DIR = "data/maps/val",
        TRAIN_DIR = "data/edges2shoes/train",
        VAL_DIR = "data/edges2shoes/val",
        LR = 2e-4,
        BATCH_SIZE = 16,
        NUM_WORKERS = 2,
        IMAGE_SIZE = 256,
        CHANNELS_IMG = 3,
        L1_LAMBDA = 100,
        LAMBDA_GP = 10,
        NUM_EPOCHS = 500,
        # LOAD_MODEL = False,
        SAVE_MODEL = False,
        LOAD_CHECKPOINT = False,
        CHECKPOINT_DISC = "disc.pth.tar",
        CHECKPOINT_GEN = "gen.pth.tar",
        # SAVE_IMAGE = "evaluation1",
        SAVE_IMAGE = "evaluation_shoes",
        # CHECKPOINT_DISC = "Discriminator_Checkpoint/",
        # CHECKPOINT_GEN = "Generator_Checkpoint/",
        G_NAME = 'Pix2pix_Generator',
        D_NAME = 'Pix2pix_Discriminator',
        LOSS_NAME = 'BCEWithLogitsLoss',
        OPTIMIZER_G = 'Adam',
        OPTIMIZER_D = 'Adam',
    ),

    2: dict(
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu",
        TRAIN_DIR = "cycle_data/horse2zebra",
        VAL_DIR = "cycle_data/horse2zebra",
        BATCH_SIZE = 1,
        LR = 1e-5,
        LAMBDA_IDENTITY = 0.0,
        LAMBDA_CYCLE = 10,
        NUM_WORKERS = 4,
        NUM_EPOCHS = 100,
        LOAD_MODEL = False,
        SAVE_MODEL = True,
        G_H_NAME = 'Cycle_Generator',
        G_Z_NAME = 'Cycle_Generator',
        D_H_NAME = 'Cycle_Discriminator',
        D_Z_NAME = 'Cycle_Discriminator',
        LOSS_NAME = 'BCE',
        OPTIMIZER_D = 'Adam_cyclegan',
        OPTIMIZER_G = 'Adam_cyclegan',
        CHECKPOINT_GEN_H = "genh.pth.tar",
        CHECKPOINT_GEN_Z = "genz.pth.tar",
        CHECKPOINT_CRITIC_H = "critich.pth.tar",
        CHECKPOINT_CRITIC_Z = "criticz.pth.tar",
    )
}

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
# both_transform = A.Compose(
#     [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
# )

# transform_only_input = A.Compose(
#     [
#         A.HorizontalFlip(p=0.5),
#         A.ColorJitter(p=0.2),
#         A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#         ToTensorV2(),
#     ]
# )

# transform_only_mask = A.Compose(
#     [
#         A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#         ToTensorV2(),
#     ]
# )