import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from util import TwoCropTransform  # implement as in SimCLR repo
from losses import NTXentLoss       # standard SimCLR loss implementation

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class OODModel(nn.Module):
    def __init__(self, dino_model="dinov2_vits14", proj_dim=128, cls_hidden=256):
        super().__init__()
        # Load DINO backbone
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", dino_model
        )
        feat_dim = self.backbone.head.in_features  # dimension of pooled output

        # Projection head for SimCLR
        self.proj = ProjectionHead(feat_dim, hidden_dim=512, out_dim=proj_dim)
        # Classification head for OOD binary prediction
        self.cls = nn.Sequential(
            nn.Linear(feat_dim, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cls_hidden, 1)
        )

    def forward_backbone(self, x):
        # x: normalized image batch
        feats = self.backbone.forward_features(x)
        # Pool patch tokens: mean over tokens
        tokens = feats['x_norm_patchtokens']  # [B, N, D]
        pooled = tokens.mean(dim=1)           # [B, D]
        return pooled

    def forward(self, x1, x2=None):
        # compute features for one or two views
        p1 = self.forward_backbone(x1)
        z1 = self.proj(p1)
        if x2 is None:
            return z1, p1
        p2 = self.forward_backbone(x2)
        z2 = self.proj(p2)
        return (z1, z2), (p1, p2)

    def classify(self, features):
        # features: [B, D]
        return torch.sigmoid(self.cls(features)).squeeze(1)


class LabeledImageFolder(datasets.ImageFolder):
    """
    Extends ImageFolder to return two augmented views and the binary label.
    """
    def __init__(self, root, transform_twice):
        super().__init__(root, transform=transform_twice)

    def __getitem__(self, index):
        (img1, img2), _ = super().__getitem__(index)
        # folder names: assume class0=normal, class1=ood
        label = 1.0 if self.targets[index] == 1 else 0.0
        return img1, img2, label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='root with subfolders 0 (normal), 1 (ood)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for NT-Xent')
    parser.add_argument('--wd', type=float, default=1e-6)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Transforms: two crops for SimCLR
    simclr_transform = TwoCropTransform(
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])
    )

    # Dataset & loader
    dataset = LabeledImageFolder(args.data_dir, simclr_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OODModel().to(device)
    ntxent = NTXentLoss(batch_size=args.batch_size, temperature=args.temp)
    bce = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.wd)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_c_loss = 0.0
        total_bce = 0.0
        for img1, img2, label in loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            # Forward
            (z1, z2), (p1, p2) = model(img1, img2)
            # Contrastive loss
            c_loss = ntxent(z1, z2)
            # Classification loss on one view's pooled features
            preds = model.classify(p1)
            cls_loss = bce(preds, label)

            loss = c_loss + cls_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_c_loss += c_loss.item()
            total_bce += cls_loss.item()

        print(f"Epoch {epoch}/{args.epochs} ",
              f"Contrastive: {total_c_loss/len(loader):.4f}",
              f"Cls: {total_bce/len(loader):.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"epoch{epoch}.pt"))


if __name__ == '__main__':
    main()
