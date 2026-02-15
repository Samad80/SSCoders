"""
Optimized Segmentation Training Script
Trains a segmentation head on top of DINOv2 backbone
Optimized for hackathon performance
""" 

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import random
from tqdm import tqdm

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    try:
        img = np.array(img)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = np.moveaxis(img, 0, -1)
        img = (img * std + mean) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(filename, img[:, :, ::-1])
    except Exception as e:
        print(f"Error saving image {filename}: {e}")


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)

class_names = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 
               'Dry Bushes', 'Ground Clutter', 'Logs', 'Rocks', 
               'Landscape', 'Sky']


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.masks_dir):
            raise ValueError(f"Mask directory not found: {self.masks_dir}")
        
        self.data_ids = sorted(os.listdir(self.image_dir))
        print(f"Found {len(self.data_ids)} images in {data_dir}")

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255

        return image, mask.long()


# ============================================================================
# Model: Improved Segmentation Head
# ============================================================================

class ImprovedDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, tokenH, tokenW):
        super().__init__()
        self.H = tokenH
        self.W = tokenW

        # Initial projection
        self.proj = nn.Conv2d(in_channels, 512, 1)
        self.dropout = nn.Dropout2d(0.1)

        # Block 1: 512 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        # Upsample 1: 512 -> 256
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Block 2: 256 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Upsample 2: 256 -> 128
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Block 3: 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Upsample 3: 128 -> 64
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)

        x = self.proj(x)
        x = self.dropout(x)
        
        # Residual connection for block1
        identity = x
        x = self.block1(x)
        x = x + identity
        
        x = self.up1(x)
        x = self.block2(x)
        x = self.up2(x)
        x = self.block3(x)
        x = self.up3(x)
        x = self.refine(x)

        return self.final(x)


# ============================================================================
# Combined Loss Function
# ============================================================================

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=n_classes).permute(0,3,1,2).float()
        intersection = (pred * target_onehot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(self, pred, target):
        return self.ce(pred, target) + 0.5 * self.dice_loss(pred, target)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            if target_inds.sum() > 0:
                iou_per_class.append(0.0)
        else:
            iou_per_class.append((intersection / (union + 1e-6)).cpu().numpy())

    return np.nanmean(iou_per_class) if iou_per_class else 0.0


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)

        if target_inds.sum() > 0:
            dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class) if dice_per_class else 0.0


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    correct = (pred_classes == target)
    return correct.float().mean().cpu().numpy()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, show_progress=True):
    """Evaluate all metrics on a dataset."""
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []

    model.eval()
    backbone.eval()
    
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") if show_progress else data_loader
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            output = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(output.to(device))
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels = labels.squeeze(dim=1).long()

            iou = compute_iou(outputs, labels, num_classes=num_classes)
            dice = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)

            if not np.isnan(iou):
                iou_scores.append(iou)
            if not np.isnan(dice):
                dice_scores.append(dice)
            if not np.isnan(pixel_acc):
                pixel_accuracies.append(pixel_acc)

    model.train()
    return (np.mean(iou_scores) if iou_scores else 0.0,
            np.mean(dice_scores) if dice_scores else 0.0,
            np.mean(pixel_accuracies) if pixel_accuracies else 0.0)


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    dpi = 150
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Loss and Accuracy
    plt.figure(figsize=(12, 5), dpi=dpi)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, history['val_loss'], label='val', linewidth=2, marker='s', markersize=4)
    plt.title('Loss vs Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_pixel_acc'], label='train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, history['val_pixel_acc'], label='val', linewidth=2, marker='s', markersize=4)
    plt.title('Pixel Accuracy vs Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    # Plot 2: IoU
    plt.figure(figsize=(12, 5), dpi=dpi)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_iou'], label='Train IoU', linewidth=2, marker='o', markersize=4)
    plt.title('Train IoU vs Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('IoU', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_iou'], label='Val IoU', linewidth=2, marker='s', markersize=4, color='orange')
    plt.title('Validation IoU vs Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('IoU', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    # Plot 3: Dice
    plt.figure(figsize=(12, 5), dpi=dpi)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_dice'], label='Train Dice', linewidth=2, marker='o', markersize=4)
    plt.title('Train Dice vs Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_dice'], label='Val Dice', linewidth=2, marker='s', markersize=4, color='orange')
    plt.title('Validation Dice vs Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    # Plot 4: Combined metrics
    plt.figure(figsize=(14, 10), dpi=dpi)

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, history['val_loss'], label='val', linewidth=2, marker='s', markersize=4)
    plt.title('Loss vs Epoch', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_iou'], label='train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, history['val_iou'], label='val', linewidth=2, marker='s', markersize=4)
    plt.title('IoU vs Epoch', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_dice'], label='train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, history['val_dice'], label='val', linewidth=2, marker='s', markersize=4)
    plt.title('Dice Score vs Epoch', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_pixel_acc'], label='train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, history['val_pixel_acc'], label='val', linewidth=2, marker='s', markersize=4)
    plt.title('Pixel Accuracy vs Epoch', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved training plots to '{output_dir}/'")


def save_history_to_file(history, output_dir):
    """Save training history to a text file."""
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SEGMENTATION TRAINING RESULTS\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Total Epochs Trained: {len(history['train_loss'])}\n")
        f.write("=" * 70 + "\n\n")

        f.write("FINAL METRICS:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 70 + "\n\n")

        f.write("BEST RESULTS:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 70 + "\n\n")

        f.write("PER-EPOCH HISTORY:\n")
        f.write("-" * 110 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 110 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_iou'][i],
                history['val_iou'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_pixel_acc'][i],
                history['val_pixel_acc'][i]
            ))

    print(f"✓ Saved evaluation metrics to '{filepath}'")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    print("\n" + "="*80)
    print("OPTIMIZED SEGMENTATION TRAINING - HACKATHON VERSION")
    print("="*80 + "\n")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed_all(42)

    # ========== OPTIMIZED HYPERPARAMETERS ==========
    batch_size = 8  # Increased from 2
    w = int(((960) // 14) * 14)  # 952 pixels (removed /2)
    h = int(((540) // 14) * 14)  # 532 pixels (removed /2)
    lr = 5e-4  # Increased from 1e-4
    n_epochs = 30  # Increased from 10
    
    print(f"\nHyperparameters:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Resolution: {h}x{w}")
    print(f"  Learning Rate: {lr}")
    print(f"  Epochs: {n_epochs}")

    # Output directory
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Running in Jupyter notebook
        script_dir = os.getcwd()
    
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # ========== TRANSFORMS WITH AUGMENTATION ==========
    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])

    # Dataset paths
    data_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val')

    # Create datasets
    print("\nLoading datasets...")
    
    # Set DataLoader settings based on GPU availability
    use_cuda = torch.cuda.is_available()
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {'num_workers': 0}
    
    trainset = MaskDataset(data_dir=data_dir, transform=train_transform, mask_transform=mask_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, **loader_kwargs)

    valset = MaskDataset(data_dir=val_dir, transform=val_transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # Load DINOv2 backbone
    print("\nLoading DINOv2 backbone...")
    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    print("✓ Backbone loaded successfully")

    # Get embedding dimension
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"✓ Embedding dimension: {n_embedding}")
    print(f"✓ Patch tokens shape: {output.shape}")

    # ========== IMPROVED DECODER ==========
    classifier = ImprovedDecoder(
        in_channels=n_embedding,
        num_classes=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    )
    classifier = classifier.to(device)
    print(f"✓ Decoder created with {sum(p.numel() for p in classifier.parameters())/1e6:.2f}M parameters")

    # ========== COMBINED LOSS ==========
    loss_fct = CombinedLoss()
    print("✓ Using Combined Loss (CrossEntropy + Dice)")

    # ========== OPTIMIZER WITH SCHEDULER ==========
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    print("✓ Optimizer: AdamW with ReduceLROnPlateau scheduler")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_dice': [],
        'val_dice': [],
        'train_pixel_acc': [],
        'val_pixel_acc': []
    }

    # ========== EARLY STOPPING ==========
    best_val_iou = 0.0
    patience = 10
    patience_counter = 0
    best_model_path = os.path.join(script_dir, "best_segmentation_head.pth")

    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    for epoch in range(n_epochs):
        # ===== TRAINING PHASE =====
        classifier.train()
        backbone_model.eval()
        train_losses = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

            logits = classifier(output)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            labels = labels.squeeze(dim=1).long()

            loss = loss_fct(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ===== VALIDATION PHASE =====
        classifier.eval()
        val_losses = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False)
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(device), labels.to(device)

                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(output)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                labels = labels.squeeze(dim=1).long()

                loss = loss_fct(outputs, labels)
                val_losses.append(loss.item())
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ===== CALCULATE METRICS =====
        train_iou, train_dice, train_pixel_acc = evaluate_metrics(
            classifier, backbone_model, train_loader, device, num_classes=n_classes, show_progress=False
        )
        val_iou, val_dice, val_pixel_acc = evaluate_metrics(
            classifier, backbone_model, val_loader, device, num_classes=n_classes, show_progress=False
        )

        # Store history
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['val_pixel_acc'].append(val_pixel_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{n_epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(f"  Train IoU:  {train_iou:.4f} | Val IoU:  {val_iou:.4f}")
        print(f"  Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")
        print(f"  Train Acc:  {train_pixel_acc:.4f} | Val Acc:  {val_pixel_acc:.4f}")

        # ===== EARLY STOPPING & MODEL SAVING =====
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            torch.save(classifier.state_dict(), best_model_path)
            print(f"  ★ New best model saved! (Val IoU: {val_iou:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break

        # ===== LEARNING RATE SCHEDULING =====
        scheduler.step(val_iou)

    # Save final results
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80 + "\n")
    
    print("Generating plots and saving results...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)

    # Save final model
    final_model_path = os.path.join(script_dir, "final_segmentation_head.pth")
    torch.save(classifier.state_dict(), final_model_path)
    print(f"✓ Saved final model to '{final_model_path}'")

    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS:")
    print("="*80)
    print(f"Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})")
    print(f"Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})")
    print(f"Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})")
    print(f"Final Val IoU:     {history['val_iou'][-1]:.4f}")
    print(f"Final Val Dice:    {history['val_dice'][-1]:.4f}")
    print(f"Final Val Accuracy:{history['val_pixel_acc'][-1]:.4f}")
    print("="*80 + "\n")
    
    print(f"🏆 Best model saved at: {best_model_path}")
    print(f"📊 Training stats saved in: {output_dir}/")
    print("\nYou're ready for the hackathon! Good luck! 🚀\n")


if __name__ == "__main__":
    main()
    