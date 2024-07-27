import sys
sys.path.insert(0, './')
sys.path.insert(0, './stylegan2/')

from model import Generator
from models.glow import Glow
from utils.rarity_score import MANIFOLD

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import torch
from torchvision import utils
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import argparse
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=0):
	deterministic = True
	import random
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	if deterministic:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
set_seed() # for reproducibility

parser = argparse.ArgumentParser(description="Optimization")
parser.add_argument("--zG_path", default="", type=str, help="initial zG path")

parser.add_argument("--real_feature_path", default="", type=str, help="real feature path")

parser.add_argument("--n_sample", default=10, type=int, help="number of samples to generate per initial point")
parser.add_argument("--rand_scale", default=0.1, type=float, help="scaling coefficient of random noise")

parser.add_argument("--img_size", default=1024, type=int, help="image size")
parser.add_argument("--channel_multiplier", default=2, type=int, help="channel multiplier of the generator. config-f = 2, else = 1")
parser.add_argument("--gen_ckpt", default="", type=str, help="stylegan2 ckpt path")

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
parser.add_argument("--n_group", default=4, type=int, help="number of groups")
parser.add_argument("--nf_ckpt", default="", type=str, help="glow ckpt path")

parser.add_argument("--scaler_path", default="./scaler/ffhq_minmax_scaler.pkl", type=str, help="scaler path")

parser.add_argument("--epoch", default=200, type=int, help="number of total epoch")
parser.add_argument("--lr", default=1e-2, type=float, help="lr")
parser.add_argument("--schedule", action="store_true", help="use lr scheduler")

parser.add_argument("--save_dir", default="./results/", type=str, help="result save directory")
parser.add_argument("--save_interval", default=1, type=int, help="number of interval to save the results")

parser.add_argument("--alpha", default=0.002, type=float)
parser.add_argument("--eta", default=30.0, type=float)

parser.add_argument("--dists_path", default="", type=str)
parser.add_argument("--pair_k", default=10, type=int)
parser.add_argument("--div_true", action="store_false")


def loss_fn(x_initial, penalize_distance, pair_k = 10, alpha=0.002, eta=30.0, increase=False, div_true=False):
    penalize_distance = torch.Tensor(penalize_distance).to(device)
    def loss_fn_helper(prob, xs, t):
        # rare objective
        if increase:
            loss = - prob
        else:
            loss = prob
        
        # diversity objective
        if not div_true: # random {pair_k} pairs used
            n_sample = xs.shape[0]
            xis = xs[torch.randint(0,n_sample,(pair_k,))]
            xjs = xs[torch.randint(0,n_sample,(pair_k,))]
            diff = xis - xjs
            xij_dists = torch.mean(diff*diff, dim=-1)
        else: # all pairs used
            xij_dists = torch.triu(torch.cdist(xs.unsqueeze(0), xs.unsqueeze(0)).squeeze())
            xij_dists = xij_dists*xij_dists
            
        loss = loss -alpha * torch.sum(xij_dists)
        
        # regularization objective
        distance = F.pairwise_distance(x_initial, xs)
        distance = torch.max(distance, penalize_distance)
        distance_diff = distance - penalize_distance
        loss = loss + eta * torch.sum(distance_diff*distance_diff, dim=-1)

        return loss, distance_diff.detach().cpu().numpy()
    return loss_fn_helper

def optimization(n_sample, zG_0, manifold, k=3, epoch=200, save_dir="./results/", save_name="", save_interval=10, lr=1e-2, increase=False, schedule=True, rand_scale=0.1, pair_k = 10, alpha=0.002, penalize_distance=100, eta=30.0, div_true=True):
    zG_0 = zG_0.repeat(n_sample, 1)
    zG_0 = zG_0 + rand_scale * torch.randn(zG_0.shape).to(device)
    initial_latents = zG_0.detach().clone()
    
    initial_x, _ = generator([initial_latents], return_latents=True, truncation=1, truncation_latent=None, input_is_latent=False, randomize_noise=False)
    initial_x = resize_224(initial_x)
    initial_x = vgg16.features(initial_x)
    initial_x = initial_x.view(-1, 7 * 7 * 512)
    initial_x = vgg16.classifier[:5](initial_x)
    # initial_x = ((initial_x - scaler_min)/data_range).reshape((-1, 4096, 1, 1))
    initial_x = initial_x.detach().clone()
    
    initial_guess = torch.zeros(zG_0.shape, device=device).requires_grad_(True)
    optimizer = optim.Adam([initial_guess], lr=lr)
    if schedule: scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    scaler = GradScaler()
    
    best_loss = [float('inf') for _ in range(n_sample)]
    best_rarity = [-float('inf') for _ in range(n_sample)]
    
    log_pxs = []
    rarities = []
    
    L = loss_fn(initial_x, penalize_distance, alpha=alpha, eta=eta, increase=increase, pair_k=pair_k, div_true=div_true)
 
    with tqdm(range(epoch+1)) as pbar:
        for i in pbar:
            optimizer.zero_grad()
            
            zG = zG_0 + initial_guess
            img, _ = generator([zG], return_latents=True, truncation=1, truncation_latent=None, input_is_latent=False, randomize_noise=False)
            with autocast():
                batch = resize_224(img)
                before_fc = vgg16.features(batch)
                before_fc = before_fc.view(-1, 7 * 7 * 512)
                feature_unnorm = vgg16.classifier[:5](before_fc)
                feature = ((feature_unnorm - scaler_min)/data_range).reshape((-1, 4096, 1, 1))
                
                log_p, logdet, zs = nf(feature)
                log_px = (log_p + logdet).squeeze()
                
                
                loss, dists = L(log_px, feature_unnorm, i)
                sum_loss = torch.sum(loss)

            scaler.scale(sum_loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_description(f"log p(x) : {torch.mean(log_px).item():.3f}; loss : {sum_loss.item():.3f}")
            
            if i%save_interval == 0:
                rarity = manifold.rarity(k=k, samples=feature_unnorm.detach().cpu().numpy())[0]
                
                log_pxs.append(log_px.detach().cpu().numpy())
                rarities.append(rarity)
                for j in range(n_sample):
                    if i%10 == 0:
                        utils.save_image(
                            img[j],
                            save_dir + f"{save_name}_{alpha}_{eta}_{j}_{i}_{epoch}_optimized.png",
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1)
                        )
                    else:
                        if (best_loss[j] > loss[j] and 0 < rarity[j] and dists[j] == 0) or i==0:
                            best_loss[j] = loss[j].item()
                            utils.save_image(
                                img[j],
                                save_dir + f"{save_name}_{alpha}_{eta}_{j}_best_optimized.png",
                                nrow=1,
                                normalize=True,
                                value_range=(-1, 1)
                            )
                
            if schedule: 
                scheduler.step()
    return log_pxs, rarities


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    
    # latent code of generator
    zG = np.load(args.zG_path)[:1000]
    
    # penalizing distance
    p_dists = np.load(args.dists_path)[:1000]
    
    # real features and construct real kNN manifold
    real_features = np.load(args.real_feature_path)["feature"]
    manifold = MANIFOLD(real_features=real_features, fake_features=np.zeros_like((1, real_features.shape[1])))
    
    # load pre-trained generator
    generator = Generator(args.img_size, 512, 8, channel_multiplier=args.channel_multiplier).to(device)
    checkpoint = torch.load(args.gen_ckpt)
    generator.load_state_dict(checkpoint["g_ema"])
    generator = generator.eval()
    
    # load pre-trained feature extractor
    vgg16 = models.vgg16(pretrained=True).to(device).eval()
    
    # load normalizing flow model
    n_flow = args.n_flow
    n_block = args.n_block
    affine= args.affine
    no_lu = args.no_lu
    n_group = args.n_group
    n_F = 4096

    nf = Glow(
            1, n_flow, n_block, affine=affine, conv_lu=not no_lu, group=n_group, n_F=n_F
        )
    nf = nn.DataParallel(nf)
    checkpoint = torch.load(args.nf_ckpt)
    nf.load_state_dict(checkpoint, strict=False)
    nf = nf.to(device).eval()

    # load scaler
    scaler = joblib.load(args.scaler_path)
    scaler_min = torch.Tensor(scaler.data_min_).to(device)
    scaler_max = torch.Tensor(scaler.data_max_).to(device)
    data_range = scaler_max - scaler_min
    data_range[data_range == 0] = 1.0
    
    # define resizing function
    resize_224 = partial(F.interpolate, size=(224, 224))
        
    # optimization
    for i, idx in tqdm(enumerate(range(1000))):
        set_seed(idx) # for reproducibility
        
        zG_0 = torch.Tensor(zG[i]).to(device).requires_grad_(False)
        penalize_distance = [p_dists[idx]]
        
        _, _ = optimization(args.n_sample, zG_0, manifold, k=3, epoch=args.epoch, save_dir=args.save_dir, save_name=f'{idx}', save_interval=args.save_interval, lr=args.lr, increase=False, schedule=args.schedule, rand_scale=args.rand_scale, alpha=args.alpha, eta=args.eta, penalize_distance=penalize_distance, pair_k=args.pair_k, div_true=args.div_true)
