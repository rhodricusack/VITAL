import torch
from utils.imagenet_dataset import ImageNet
import torchvision.models as models
import os
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
import argparse
from collections import defaultdict


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def get_activation_map(model, layer_name):
    activations = {}

    def hook_fn(module, input, output):
        activations['feat'] = output

    print(dict([*model.named_modules()]).keys())
    layer = dict([*model.named_modules()])[layer_name]
    layer.register_forward_hook(hook_fn)
    return activations


def generate_dataloader(batch_size=64, split_N=None):
    if split_N is None:
        dataset_real = ImageNet(split='train')
    else:
        dataset_real = ImageNet(split='train_subset', split_N=split_N)

    dloader_real = torch.utils.data.DataLoader(
        dataset_real,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    return dloader_real


def generate_model(arch='resnet50'):
    if 'resnet50_BV' in arch:
        epoch=60
        modeldict = {
            f"resnet50_BV_g0For60_E{epoch}": f"/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_0_for_60_epoch{epoch}.pth.tar",
            f"resnet50_BV_g0pt5For60_E{epoch}": f"/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0pt5_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_0pt5_for_60_epoch_epoch{epoch}.pth.tar",
            f"resnet50_BV_g1For60_E{epoch}": f"/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_1_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_1_for_60_epoch_epoch{epoch}.pth.tar",
            f"resnet50_BV_g1pt5For60_E{epoch}": f"/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_1pt5_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_1pt5_for_60_epoch_epoch{epoch}.pth.tar",
            f"resnet50_BV_g2For60_E{epoch}": f"/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_2_for_60_epoch_ker_13/outmodel/checkpoint_supervised_resnet50_gauss_2_for_60_epoch_ker_13_epoch{epoch}.pth.tar",
            f"resnet50_BV_g3For60_E{epoch}": f"/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_3_for_60_epoch_mmk/outmodel/checkpoint_supervised_resnet50_gauss_3_for_60_epoch_mmk_epoch{epoch}.pth.tar",
            f"resnet50_BV_g4For60_E{epoch}": f"/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_4_for_60_epoch_mmk/outmodel/checkpoint_supervised_resnet50_gauss_4_for_60_epoch_mmk_epoch{epoch}.pth.tar",
            f"resnet50_BV_g6For60_E{epoch}": f"/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_6_for_60_epoch_mmk/outmodel/checkpoint_supervised_resnet50_gauss_6_for_60_epoch_mmk_epoch{epoch}.pth.tar"
        }

        modelpth = modeldict[arch]
        
        from collections import OrderedDict

        # 1) Load the state dict
        state_dict = torch.load(modelpth, map_location="cpu")  # or your path/device
        state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict

        # 2) Strip 'module.' prefix from keys
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # remove only the first occurrence of 'module.'
            new_key = k.replace("module.", "", 1)
            new_state_dict[new_key] = v

        # 3) Create a standard resnet50
        net = models.resnet50(weights=None)  # or pretrained=False in older torchvision

        # 4) Load weights
        net.load_state_dict(new_state_dict)   # use strict=False if fc shape mismatches
        net = net.cuda().eval()

        # # Alternative loading method
        # net = models.resnet50()
        # checkpoint = torch.load(modelpth)
        # net = torch.nn.DataParallel(net)
        
        # if 'state_dict' in checkpoint:
        #     net.load_state_dict(checkpoint['state_dict'])
        # net = net.cuda().eval()

    else:
        net = models.__dict__[arch](pretrained=True).cuda().eval()

    # for name, module in net.named_modules():
    #     print(f"Layer Name: {name}, Layer Type: {type(module)}")

    return net


def extract_topk_image_indices_all_channels(dataloader, model, layer_name, topk_images):
    act_map = get_activation_map(model, layer_name)
    filenames = []
    all_scores = []

    for img_batch, _ in tqdm(dataloader):
        img_batch = img_batch.cuda(non_blocking=True)
        with torch.no_grad():
            _ = model(img_batch)
        activation = act_map['feat']  # [B, C, H, W]

        B, C, H, W = activation.shape
        scores = activation.view(B, C, -1).max(dim=2).values  # [B, C]
        all_scores.append(scores.cpu())

        batch_filenames = [s[0] for s in dataloader.dataset.data.samples[len(filenames):len(filenames)+B]]
        filenames.extend(batch_filenames)

    all_scores = torch.cat(all_scores, dim=0)  # [N, C]
    all_scores_t = all_scores.transpose(0, 1)  # [C, N]

    top_indices = {}
    for ch in range(all_scores_t.size(0)):
        vals, idx = torch.topk(all_scores_t[ch], k=topk_images)
        top_indices[ch] = [(filenames[i], vals[j].item()) for j, i in enumerate(idx.tolist())]

    return top_indices, all_scores_t.size(0)


def save_txt(top_indices, arch, target_layer):

    for ch, img_list in tqdm(top_indices.items(), desc="Extracting patches"): 
        save_dir = f'patch_results/{arch}/neuron_{target_layer}/{ch}/'
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, 'files_all.txt'), 'w') as f:

            for path,_ in img_list:
                f.write(f"{path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_layer', type=str, default='layer4')
    parser.add_argument('--arch', type=str, default='resnet50_gauss_6')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--topk_images', type=int, default=10000)
    parser.add_argument('--split_N', type=int, default=None)
    args = parser.parse_args()
    print(args)
    
    dataloader = generate_dataloader(batch_size=args.batch_size, split_N=args.split_N)
    model = generate_model(args.arch)

    top_indices, num_channels = extract_topk_image_indices_all_channels(
        dataloader=dataloader,
        model=model,
        layer_name=args.target_layer,
        topk_images=args.topk_images
    )

    save_txt(
        top_indices=top_indices,
        arch=args.arch,
        target_layer=args.target_layer
    )
