###############################################################################
# simcbctgenerator
#
# Copyright 2025 Lukas Zimmermann and Michael Rauter
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3, vit_l_16, ViT_L_16_Weights

import numpy as np
from scipy import linalg
import os
import SimpleITK as sitk
from tqdm import tqdm


class MedicalImageDataset(Dataset):
    """Dataset for loading medical images."""
    def __init__(self, image_dir, real=False, half=None):
        self.image_dir = image_dir
        if half is None:
            self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('_0000.nii.gz'))]
        else:
            all_files = [f for f in os.listdir(image_dir) if f.endswith(('_0000.nii.gz'))]
            if half:
                self.image_files = all_files[:len(all_files)//2]
            else:
                self.image_files = all_files[len(all_files)//2:]

        self.mean = 0.
        self.std = 500.
        self.real = real

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])

        # For DICOM files, you would use a library like pydicom instead
        ct = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
        mask = ct > -1024
        count = mask.sum(1).sum(1)
        idx = np.where(count != 0)
        mini, maxi = idx[0].min(), idx[0].max()
        image = np.repeat(ct[mini+8:maxi-8,None], 3, axis=1)
        # if self.real:
        #     image = image[:, :, 25:-25,25:-25]

        image = (image-self.mean)/self.std#image.mean())/image.std()

        return image


class InceptionV3FeatureExtractor(nn.Module):
    """Modified InceptionV3 for feature extraction."""
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        inception = inception_v3(pretrained=True)
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d,
            inception.Mixed_6e, inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        x = self.blocks(x)
        return x.squeeze()


class ViTFeatureExtractor(nn.Module):
    """Vision Transformer for feature extraction from medical images."""
    def __init__(self, weights=ViT_L_16_Weights.DEFAULT):
        super(ViTFeatureExtractor, self).__init__()
        # Load pretrained ViT-L/16 model
        self.model = vit_l_16(weights=weights)
        self.model.eval()

        # Remove the classification head to get features
        self.model.heads = nn.Identity()

    def preprocess(self, images):
        """Preprocess images for ViT (apply ImageNet normalization)."""
        # Images come as torch tensors (N, 3, H, W) from medical dataloader
        # Resize to 224x224 for ViT
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalization is intentionally skipped here because the upstream
        # medical dataloader already standardizes the input volumes.
        return images

    def forward(self, x):
        """Extract ViT features from images."""
        x = self.preprocess(x)
        # Forward through ViT encoder (returns features before classification head)
        features = self.model(x)
        return features


def calculate_activation_statistics(images, model, batch_size=1, device='cuda'):
    """Calculate mean and covariance statistics of features."""
    model.eval()
    model.to(device)
    #model = nn.DataParallel(model, device_ids=[0,1])

    dataloader = DataLoader(images, batch_size=batch_size)
    features = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch[0].to(device)
            feature = model(batch)
            features.append(feature.cpu().numpy())

    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    return mu, sigma


def extract_clip_features(images, model, batch_size=1, device='cuda'):
    """Extract CLIP features for CMMD calculation."""
    model.eval()
    model.to(device)

    dataloader = DataLoader(images, batch_size=batch_size)
    features = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # batch is a tuple (image, ), we need the image tensor
            batch_images = batch[0].to(device)

            feature = model(batch_images)

            features.append(feature.cpu().numpy())

    features = np.concatenate(features, axis=0)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Fréchet distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "FID calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_mmd(x, y, sigma=10.0, scale=1000.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two sets of embeddings.

    This implements the biased/minimum-variance MMD estimator using RBF kernel.
    Based on: https://github.com/sayakpaul/cmmd-pytorch

    Args:
        x: Real image embeddings of shape (n, embedding_dim)
        y: Generated image embeddings of shape (m, embedding_dim)
        sigma: RBF kernel bandwidth parameter (default: 10.0)
        scale: Scaling factor for final MMD score (default: 1000.0)

    Returns:
        MMD distance scaled by the scale factor
    """
    # Convert to torch tensors if needed
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    # Compute squared norms
    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    # RBF kernel parameter
    gamma = 1.0 / (2 * sigma ** 2)

    # Compute kernel gram matrices using RBF kernel
    # K(x,y) = exp(-gamma * ||x-y||^2)
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*<x,y>

    # K_xx: kernel between real samples
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) +
                           torch.unsqueeze(x_sqnorms, 1) +
                           torch.unsqueeze(x_sqnorms, 0)))
    )

    # K_xy: kernel between real and generated samples
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) +
                           torch.unsqueeze(x_sqnorms, 1) +
                           torch.unsqueeze(y_sqnorms, 0)))
    )

    # K_yy: kernel between generated samples
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) +
                           torch.unsqueeze(y_sqnorms, 1) +
                           torch.unsqueeze(y_sqnorms, 0)))
    )

    # MMD^2 = E[K(x,x)] + E[K(y,y)] - 2*E[K(x,y)]
    mmd = scale * (k_xx + k_yy - 2 * k_xy)

    return mmd.item()


def calculate_fid(real_image_dir, generated_image_dir, batch_size=1, device='cuda', internal=False):
    """Calculate FID between real and generated images."""


    # Load datasets
    if internal:
        real_dataset = MedicalImageDataset(real_image_dir, half=True)
        generated_dataset = MedicalImageDataset(real_image_dir, half=False)
    else:
        real_dataset = MedicalImageDataset(real_image_dir)
        generated_dataset = MedicalImageDataset(generated_image_dir)

    # Initialize model
    model = InceptionV3FeatureExtractor()

    # Calculate statistics
    mu_real, sigma_real = calculate_activation_statistics(real_dataset, model, batch_size, device)
    np.save('mu_real.npy', mu_real)
    np.save('sigma_real.npy', sigma_real)
    mu_gen, sigma_gen = calculate_activation_statistics(generated_dataset, model, batch_size, device)
    np.save('mu_gen.npy', mu_gen)
    np.save('sigma_gen.npy', sigma_gen)

    # Calculate FID
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    return fid_value


def calculate_cmmd(real_image_dir, generated_image_dir, batch_size=1, device='cuda', internal=False, sigma=10.0, scale=1000.0):
    """
    Calculate CMMD (CLIP Maximum Mean Discrepancy) between real and generated images.

    This metric uses CLIP embeddings and Maximum Mean Discrepancy to compare distributions.
    It provides an alternative to FID that doesn't assume Gaussian distributions.

    Args:
        real_image_dir: Path to directory containing real images
        generated_image_dir: Path to directory containing generated images
        batch_size: Batch size for processing (default: 1)
        device: Device to run on ('cuda' or 'cpu')
        internal: If True, split real_image_dir in half for comparison (default: False)
        sigma: RBF kernel bandwidth parameter (default: 10.0)
        scale: Scaling factor for MMD score (default: 1000.0)

    Returns:
        CMMD score (lower is better)
    """
    # Load datasets
    if internal:
        real_dataset = MedicalImageDataset(real_image_dir, half=True)
        generated_dataset = MedicalImageDataset(real_image_dir, half=False)
    else:
        real_dataset = MedicalImageDataset(real_image_dir)
        generated_dataset = MedicalImageDataset(generated_image_dir)

    # Initialize CLIP model
    model = ViTFeatureExtractor()

    # Extract features
    print("Extracting CLIP features from real images...")
    real_features = extract_clip_features(real_dataset, model, batch_size, device)
    np.save('clip_features_real.npy', real_features)

    print("Extracting CLIP features from generated images...")
    gen_features = extract_clip_features(generated_dataset, model, batch_size, device)
    np.save('clip_features_gen.npy', gen_features)

    # Calculate CMMD using MMD
    cmmd_value = compute_mmd(real_features, gen_features, sigma=sigma, scale=scale)

    return cmmd_value


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID and CMMD between real and generated medical images.")
    parser.add_argument('--internal', action='store_true',
                        help='If set, split the real images directory in half for comparison.')
    parser.add_argument('--real_path', type=str, default='',
                        help='Path to the directory containing real images.')
    parser.add_argument('--gen_path', type=str, default='',
                        help='Path to the directory containing generated images.')
    parser.add_argument('--metric', type=str, choices=['fid', 'cmmd', 'both'], default='both',
                        help='Metric to calculate: fid, cmmd, or both.')
    args = parser.parse_args()
    internal = args.internal
    real_images_path = args.real_path
    generated_images_path = args.gen_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Calculate FID (using InceptionV3 features)
    if args.metric in ['fid', 'both']:
        print("=" * 50)
        print("Calculating FID...")
        print("=" * 50)
        fid = calculate_fid(real_images_path, generated_images_path, device=device, internal=internal)
        print(f"FID score: {fid:.2f}")

    # Calculate CMMD (using CLIP features)
    if args.metric in ['cmmd', 'both']:
        print("\n" + "=" * 50)
        print("Calculating CMMD...")
        print("=" * 50)
        cmmd = calculate_cmmd(real_images_path, generated_images_path, device=device, internal=internal)
        print(f"CMMD score: {cmmd:.2f}")

    # print("\n" + "=" * 50)
    # print("Summary:")
    # print("=" * 50)
    # #print(f"FID:  {fid:.2f}")
    # print(f"CMMD: {cmmd:.2f}")
    print("\nNote: Lower scores indicate better quality/similarity for both metrics.")
