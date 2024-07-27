import sys
sys.path.insert(0, "./models/")
from glow import Glow

from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn import preprocessing
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=32, type=int, help="batch size")
parser.add_argument("--iter", default=3000, type=int, help="maximum iterations")

parser.add_argument("--ckpt", default="", type=str, help="checkpoint path")
parser.add_argument("--ckpt_iter", default=0, type=int, help="start iteration number")

parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_false", help="use affine coupling instead of additive"
)
parser.add_argument('--n_group', default=4, type=int, help="number of groups")

parser.add_argument("--train_split", default=0.7, type=float, help="ratio of train dataset")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--schedule", action="store_false", help="use lr scheduling")
parser.add_argument("--img_size", default=4096, type=int, help="image(input) size")
parser.add_argument("--npz_path", default="", type=str, help="Path to feature npz file")
parser.add_argument("--img_path", default="", type=str, help="Path to image directory")
parser.add_argument("--save_name", default="ffhq", type=str, help="save prefix")

def sample_data(data, batch_size):
    dataset = TensorDataset(torch.Tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0
            )
            loader = iter(loader)
            yield next(loader)

    return z_shapes

def calc_loss(log_p, logdet, image_size):
    n_pixel = image_size

    loss = logdet + log_p

    return (
        (-loss / (torch.log(torch.tensor(2.0)) * n_pixel)).mean(),
        (log_p / (torch.log(torch.tensor(2.0)) * n_pixel)).mean(),
        (logdet / (torch.log(torch.tensor(2.0)) * n_pixel)).mean(),
    )


def plot_imgs(log_probs, idx_list, img_path, save_path, save_name, iteration):
    argsort_prob = np.argsort(log_probs)
    
    gs = gridspec.GridSpec(10,10, wspace=0.0)
    plt.figure(figsize=[10,10])
    for i, arg in enumerate(argsort_prob[:100]):
        plt.subplot(gs[i])
        img = Image.open(img_path + f'{:05d}.png'.format(idx_list[arg]))
        plt.imshow(np.asarray(img))
        plt.axis('off')
    plt.savefig(save_path + save_name+ f'_low_prob_images_{iteration:06d}.png', bbox_inches='tight')
    
    gs = gridspec.GridSpec(10,10, wspace=0.0)
    plt.figure(figsize=[10,10])
    for i, arg in enumerate(argsort_prob[::-1][:100]):
        plt.subplot(gs[i])
        img = Image.open(img_path + f'{:05d}.png'.format(idx_list[arg]))
        plt.imshow(np.asarray(img))
        plt.axis('off')
    plt.savefig(save_path + save_name + f'_high_prob_images_{iteration:06d}.png', bbox_inches='tight')
    
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure

def train(args, model, optimizer, scheduler):
    features = np.load(args.npz_path)['feature']
    train_features = features[-int(len(features)* args.train_split):]
    val_features = features[:-int(len(features)* args.train_split)]
    
    scaler = preprocessing.MinMaxScaler(clip=True).fit(train_features)
    joblib.dump(scaler, './scaler/' + args.save_name + '_minmax_scaler.pkl')
    
    train_features = scaler.transform(train_features).reshape((-1, args.img_size, 1, 1))
    val_features = scaler.transform(val_features).reshape((-1, args.img_size, 1, 1))
    
    dataset = iter(sample_data(train_features, args.batch))
    val_dataset = DataLoader(TensorDataset(torch.Tensor(val_features)), batch_size=args.batch, shuffle=False, drop_last=True)

    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image = next(dataset)[0]
            image = image.to(device)
            
            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(
                        image
                    )

                    continue

            else:
                log_p, logdet, _ = model(image)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            if args.schedule: 
                scheduler.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {optimizer.param_groups[0]['lr']:.7f}"
            )

            if (i+1) % 500 == 0:
                torch.save(
                    model.state_dict(), f"./checkpoint/{args.save_name}_model_{str(args.ckpt_iter + i).zfill(6)}.pt"
                )
                torch.save(
                    optimizer.state_dict(), f"./checkpoint/{args.save_name}_optim_{str(args.ckpt_iter + i).zfill(6)}.pt"
                )
                with torch.no_grad():
                    model.eval()
                    losses = []
                    log_ps = []
                    log_dets = []
                    
                    log_probs = []
                    for data in tqdm(iter(val_dataset)):
                        image = data[0]
                        image = image.to(device)
                        
                        log_p, logdet, _ = model(image)
                        
                        log_probs.append(log_p+logdet)

                        logdet = logdet.mean()

                        loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size)
                        losses.append(loss.item())
                        log_ps.append(log_p.item())
                        log_dets.append(log_det.item())

                    print(f"Validation loss: {np.mean(losses):.5f}; Validation logP: {np.mean(log_ps):.5f}; Validation logdet: {np.mean(log_dets):.5f}")
                    
                    log_probs = torch.concat(log_probs).detach().cpu().numpy()
                    plot_imgs(log_probs, np.arange(len(val_dataset.dataset)), args.img_path, 'results/', args.save_name, args.ckpt_iter + i)
                    
                    model.train()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model_single = Glow(
        1, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu, group=args.n_group, n_F=args.img_size
    )
    model = nn.DataParallel(model_single)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.ckpt != "":
        checkpoint = torch.load(f"{args.ckpt}{args.save_name}_model_{str(args.ckpt_iter).zfill(6)}.pt")
        model_single.load_state_dict(checkpoint, strict=False)
        
        optim_checkpoint = torch.load(f"{args.ckpt}{args.save_name}_optim_{str(args.ckpt_iter).zfill(6)}.pt")
        optimizer.load_state_dict(optim_checkpoint)
        
        optimizer.param_groups[0]["lr"] = args.lr
    
    model = model.to(device)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    train(args, model, optimizer, scheduler)