import copy
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from zoo import RandomGradientEstimator, RandomGradEstimateMethod

from hook import BNStatisticsHook

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


def get_imagenet_exp_dl(batch_size):
    imagenet_dir = "./data/dataset/imagenet/"
    train_ds = datasets.ImageNet(imagenet_dir, split="train", transform=transform)
    val_ds = datasets.ImageNet(imagenet_dir, split="val", transform=transform)
    train_dl = DataLoader(train_ds, batch_size, num_workers=8, pin_memory=True, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, num_workers=8, pin_memory=True, shuffle=False)
    return train_dl, val_dl


def get_dataloader(name, batch_size, shuffle=False, train=False):
    if name == "imagenet":
        split = "train" if train else "val"
        dataset = datasets.ImageNet("./data/dataset/imagenet/", split=split, transform=transform)
    elif name == "caltech256":
        dataset = datasets.Caltech256("./data/dataset/", transform=transform, download=False)
    elif name == "cifar100":
        dataset = datasets.CIFAR100("./data/dataset/cifar100", transform=transform, download=True, train=train)
    return DataLoader(dataset, batch_size, drop_last=True, shuffle=shuffle)


def get_grad_dl(model: nn.Module, dataloader: DataLoader, device,seeds: list[int]):
    model = copy.deepcopy(model).to(device)
    
    #print(model)
    # 检查模型最后一层输入维度
    #dummy = torch.randn(1, 3, 224, 224).to(device)
    # with torch.no_grad():
        #features = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.relu(model.bn1(model.conv1(dummy))))))))
        #print("features.shape after avgpool =", features.shape)

    criterion = nn.CrossEntropyLoss()
    zo_estimator = RandomGradientEstimator(
        parameters=model.parameters(),
        mu=0.001,
        num_pert=8,
        grad_estimate_method=RandomGradEstimateMethod.rge_central,
        normalize_perturbation=True,
        device=device,
        torch_dtype=torch.float32,
        paramwise_perturb=False,
    )

    hook = BNStatisticsHook(model, train=False)
    for i,(x, y) in enumerate(dataloader):
        seed=seeds[i]
        model.zero_grad()
        hook.clear()
        x, y = x.to(device), y.to(device)
        def loss_fn(x_input, y_input):
            y_pred = model(x_input)
            return criterion(y_pred, y_input)
        # y = torch.arange(len(x)).to(device)
        grad = zo_estimator.compute_grad(x, y, loss_fn, seed=seed)
        #grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters() if p.requires_grad]

        mean_var_list = hook.mean_var_list
        yield x, y, grad, mean_var_list
