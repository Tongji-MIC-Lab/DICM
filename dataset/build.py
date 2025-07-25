import dataset
import os
from torchvision import transforms
import torchvision
from torchvision.transforms import Compose
from PIL import ImageFilter
import random
from .ContrastiveCrop import ContrastiveCrop
from .imagenet import ImageFolderCCrop

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class MultiViewTransform:
    """Create multiple views of the same image"""
    def __init__(self, transform, num_views=2):
        if not isinstance(transform, (list, tuple)):
            transform = [transform for _ in range(num_views)]
        self.transforms = transform

    def __call__(self, x):
        views = [t(x) for t in self.transforms]
        return views

class CCompose(Compose):
    def __call__(self, x):  # x: [sample, box]
        img = self.transforms[0](*x)
        for t in self.transforms[1:]:
            img = t(img)
        return img

def imagenet_pretrain_rcrop(args):
    trans_list = [
        transforms.RandomResizedCrop(size=args.aug_resize, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ]
    transform = transforms.Compose(trans_list)
    transform = MultiViewTransform(transform, num_views=2)
    return transform

def imagenet_pretrain_ccrop(args):
    trans_list = [
        ContrastiveCrop(alpha=args.alpha, size=args.aug_resize, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ]
    transform = CCompose(trans_list)
    transform = MultiViewTransform(transform, num_views=2)
    return transform

def imagenet_eval_boxes(args):
    trans = transforms.Compose([
        transforms.Resize((args.aug_resize, args.aug_resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ])
    return trans

def build_dataset(args):
    # build transform
    transform = imagenet_eval_boxes(args)

    # build dataset
    ds=torchvision.datasets.ImageFolder(root=os.path.join(args.data,'train'),transform=transform)
    return ds


def build_dataset_ccrop(args):

    # build transform
    transform_rcrop = imagenet_pretrain_rcrop(args)
    transform_ccrop = imagenet_pretrain_ccrop(args)

    # build dataset
    ds=ImageFolderCCrop(root=args.data,transform_ccrop=transform_ccrop,transform_rcrop=transform_rcrop)
    return ds
