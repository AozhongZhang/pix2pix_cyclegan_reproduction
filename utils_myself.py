import torch
import config
from torchvision.utils import save_image
from config_myself import configurations

cfg = configurations[2]
DEVICE = cfg['DEVICE']

def switch_img(x):
    out = torchvision.utils.make_grid(
                    x[:32], normalize=True
                )
    return out    

def save_some_examples(gen, val_loader, epoch, DIR):
    x, y = next(iter(val_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, DIR + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, DIR + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, DIR + f"/label_{epoch}.png")
    gen.train()

def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    # torch.save(checkpoint, os.path.join(Root_checkpoint,
    #                                     "Epoch_{}_checkpoint.pth".format(epoch + 1)))


# def load_checkpoint(checkpoint_file, model, optimizer, lr):
#     print("=> Loading checkpoint")
#     checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
#     model.load_state_dict(checkpoint["state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer"])

#     # If we don't do this then it will just have learning rate of old checkpoint
#     # and it will lead to many hours of debugging \:
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False