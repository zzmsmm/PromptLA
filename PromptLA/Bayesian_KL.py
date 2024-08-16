import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image
import os
import argparse
from utils import *
from diffusers import DiffusionPipeline
import random

from sklearn.neighbors import KernelDensity
from scipy.special import rel_entr

import io
import gc

import warnings
warnings.filterwarnings("ignore")

class JPEGCompression(object):
    def __init__(self, quality):
        self.quality = quality

    def __call__(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        compressed_img = Image.open(buffer)
        return compressed_img

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_attack = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop(96),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(degrees=45),
    # JPEGCompression(quality=85),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)  # small is ok
])

def get_dataloader(folder_path, fp_num, batch_size=20, num_workers=0, attack=False, baseline1=False, baseline2=False):
    if attack:
        dataset = ImageFolderDataset(folder_path, transform_attack)
    else:
        dataset = ImageFolderDataset(folder_path, transform)

    if baseline1:
        sub_test = range(0, int(fp_num))
    elif baseline2:
        sub_test = range(int(fp_num), 2 * int(fp_num))
    else:
        sub_test = random.sample(range(len(dataset)), int(fp_num))
    dataset = torch.utils.data.Subset(dataset, sub_test)
    # print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader

def get_inception_model():
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # 去掉最后的分类层
    model.eval()
    return model

def get_features(dataloader, model, device):
    features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            '''
            latent = vae.encode(batch.to(torch.float16)).latent_dist.sample()
            print(latent.shape)
            latent_flatten = latent.view(latent.size(0), -1)
            print(latent_flatten.shape)
            features.append(latent_flatten.cpu())
            '''
            
            batch_features = model(batch)
            # features.append(batch_features.cpu())
            features.append(batch_features)

            del batch
            gc.collect()

    features = torch.cat(features)
    return features

def compute_statistics(features):
    mean = np.mean(features, axis=0)
    centered_features = features - mean
    covariance = np.cov(centered_features, rowvar=False)

    return mean, covariance

def bayesian_estimate(features, prior_mean, prior_cov, prior_dof, prior_scale):
    n = features.size(0)
    d = features.size(1)
    
    sample_mean = torch.mean(features, dim=0)
    sample_cov = torch.cov(features.T)
    
    post_mean = (prior_scale * prior_mean + n * sample_mean) / (prior_scale + n)
    
    mean_diff = (sample_mean - prior_mean).unsqueeze(1)
    scale_matrix = prior_cov + sample_cov * (n - 1) + (prior_scale * n / (prior_scale + n)) * (mean_diff @ mean_diff.T)
    
    post_dof = prior_dof + n
   
    post_scale = prior_scale + n
    post_cov = scale_matrix / (post_dof + d + 1)
    
    return post_mean, post_cov

def kl_divergence(mean1, cov1, mean2, cov2, device):
    eps = 1e-5
    cov1 += torch.eye(cov1.shape[0], device=device) * eps
    cov2 += torch.eye(cov2.shape[0], device=device) * eps

    cov2_inv = torch.inverse(cov2)
    diff = mean2 - mean1

    # term_trace = np.trace(np.dot(cov2_inv, cov1))
    term_trace = torch.trace(torch.matmul(cov2_inv, cov1))
    # term_diff = np.dot(np.dot(diff[np.newaxis, :], cov2_inv), diff[:, np.newaxis])
    term_diff = torch.matmul(torch.matmul(diff.unsqueeze(0), cov2_inv), diff.unsqueeze(1))
    # term_det = np.log(np.linalg.det(cov2) + eps) - np.log(np.linalg.det(cov1) + eps)
    # term_det = torch.log(torch.det(cov2) + eps) - torch.log(torch.det(cov1) + eps)

    cov1_cpu = cov1.cpu()
    cov2_cpu = cov2.cpu()
    
    term_det = torch.log(torch.det(cov2_cpu) + eps) - torch.log(torch.det(cov1_cpu) + eps)
    term_det = term_det.to(device)

    k = mean1.shape[0]

    kl_div = 0.5 * (term_trace + term_diff - k + term_det)

    if kl_div < 0:
        return 0.0

    return kl_div.item()

def symmetric_kl_divergence(mean1, cov1, mean2, cov2):
    kl1 = kl_divergence(mean1, cov1, mean2, cov2)
    kl2 = kl_divergence(mean2, cov2, mean1, cov1)
    return 0.5 * (kl1 + kl2)

# 计算Jensen-Shannon散度
def jensen_shannon_divergence(mean1, cov1, mean2, cov2):
    mean_m = 0.5 * (mean1 + mean2)
    cov_m = 0.5 * (cov1 + cov2)
    kl1 = kl_divergence(mean1, cov1, mean_m, cov_m)
    kl2 = kl_divergence(mean2, cov2, mean_m, cov_m)
    return 0.5 * (kl1 + kl2)

# 计算标准化KL散度
def normalized_kl_divergence(mean1, cov1, mean2, cov2):
    kl1 = kl_divergence(mean1, cov1, mean2, cov2)
    kl2 = kl_divergence(mean2, cov2, mean1, cov1)
    return kl1 / max(kl1, kl2)

# 计算每维归一化的KL散度
def per_dimension_kl_divergence(mean1, cov1, mean2, cov2):
    kl = kl_divergence(mean1, cov1, mean2, cov2)
    k = mean1.size(0)
    return kl / k

def main():
    parser = argparse.ArgumentParser(description='Calculate the KL.')
    parser.add_argument('--prompt_id', default='1')
    parser.add_argument('--cuda', default='cuda:3')
    parser.add_argument('--fp_num', default=50, type=int)
    parser.add_argument('--attack', default='db_1')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()

    cwd = os.getcwd()
    log_dir = os.path.join(cwd, 'log', 'v1.5', 'KL', f'prompt{args.prompt_id}')
    os.makedirs(log_dir, exist_ok=True)
    if args.baseline:
        logfile = os.path.join(log_dir, f'baseline_result_{args.fp_num}.txt')
    else:
        logfile = os.path.join(log_dir, f'{args.attack}_result_{args.fp_num}.txt')
    set_up_logger(logfile)

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    model = get_inception_model().to(device)
    
    folder_path1 = f'./data/trigger/v1.5/prompt{args.prompt_id}/test_true/v1.5'
    dataloader1 = get_dataloader(folder_path1, args.fp_num)
    features1 = get_features(dataloader1, model, device)
    # features1 = features1[:,1][:, np.newaxis]
    
    folder_path2 = f'./data/trigger/v1.5/prompt{args.prompt_id}/train_true/v1.5'
    dataloader2 = get_dataloader(folder_path2, args.fp_num)
    features2 = get_features(dataloader2, model, device)
    # features2 = features2[:,1][:, np.newaxis]

    prior_mean = torch.zeros(features1.size(1)).to(device)
    prior_cov = torch.eye(features1.size(1)).to(device)
    prior_dof = features1.size(1)
    prior_scale = 1.0

    mean1, cov1 = bayesian_estimate(features1, prior_mean, prior_cov, prior_dof, prior_scale)
    mean2, cov2 = bayesian_estimate(features2, prior_mean, prior_cov, prior_dof, prior_scale)

    # mean1, cov1 = compute_statistics(features1.numpy())
    # mean2, cov2 = compute_statistics(features2.numpy())

    kl_0 = kl_divergence(mean1, cov1, mean2, cov2, device)
    logging.info("v1.5-v1.5 KL Divergence: %.2f" % kl_0)

    if args.baseline:
        folder_path3 = f'./data/trigger/v1.5/prompt{args.prompt_id}/train_true/v1.5'
        dataloader3 = get_dataloader(folder_path3, args.fp_num)
        features3 = get_features(dataloader3, model, device)
        mean3, cov3 = bayesian_estimate(features3, prior_mean, prior_cov, prior_dof, prior_scale)
        kl_1 = kl_divergence(mean1, cov1, mean3, cov3, device)
        logging.info("v1.5-v1.5 KL Divergence: %.2f" % kl_1)
        logging.info("baseline relative KL Divergence: %.2f" % ((kl_1-kl_0)/kl_0))
    else:
        folder_path3 = f'./data/trigger/v1.5/prompt{args.prompt_id}/test_false/{args.attack}'
        dataloader3 = get_dataloader(folder_path3, args.fp_num)
        features3 = get_features(dataloader3, model, device)
        mean3, cov3 = bayesian_estimate(features3, prior_mean, prior_cov, prior_dof, prior_scale)
        kl_1 = kl_divergence(mean1, cov1, mean3, cov3, device)
        logging.info("v1.5-%s KL Divergence: %.2f" % (args.attack, kl_1))
        logging.info("%s relative KL Divergence: %.2f" % (args.attack, (kl_1-kl_0)/kl_0))

if __name__ == "__main__":
    main()
