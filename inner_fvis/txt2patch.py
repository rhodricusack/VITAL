import torch
import torchvision.models as models
import os
import torchvision.utils as vutils
import argparse
from tqdm import tqdm


def list_of_ints(arg):
    return list(map(int, arg.split(',')))

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

def get_activation_map(model, layer_name):
    activations = {}

    def hook_fn(module, input, output):
        activations['feat'] = output

    layer = dict([*model.named_modules()])[layer_name]
    layer.register_forward_hook(hook_fn)
    return activations

def extract_top_patches(model, layer_name, patch_size, channel, topk_patches, arch, overwrite=False):
    from PIL import Image
    from torchvision import transforms
    import torch.nn.functional as F
    import numpy as np
    
    save_dir = f'{arch}/neuron_{layer_name}/{channel}/'
    
    if overwrite or not os.path.exists(save_dir+'patches.png'):
        with open(save_dir+'files_all.txt', 'r') as f:
            img_list = [line.strip() for line in f]

        def patchify(inputs, patch_size=64):
            stride = int(patch_size * 0.50)
            patches = F.unfold(inputs, kernel_size=patch_size, stride=stride)
            patches = patches.transpose(1, 2).contiguous().view(-1, inputs.shape[1], patch_size, patch_size)
            return patches, stride

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        act_map = get_activation_map(model, layer_name)

        patches, patch_scores = [], []

        for path in tqdm(img_list):
            img = transform(Image.open(path).convert('RGB')).unsqueeze(0).cuda()
            with torch.no_grad():
                img.requires_grad = False
                patch_batch, _ = patchify(img, patch_size=patch_size)
                _ = model(patch_batch)
                ch_activations = act_map['feat'].mean((2, 3))
                ch_activations = ch_activations[:, channel].detach().cpu()

            best_idx = torch.argmax(ch_activations).item()
            best_score = ch_activations[best_idx].item()

            patches.append(patch_batch[best_idx].cpu())
            patch_scores.append(best_score)

            del img, patch_batch
            torch.cuda.empty_cache()
            
        patch_scores = np.array(patch_scores)
        topk_idx = np.argsort(patch_scores)[::-1][:topk_patches]
        os.makedirs(save_dir, exist_ok=True)

        vutils.save_image(torch.stack([patches[i] for i in topk_idx[:16]]),
                            os.path.join(save_dir, 'patches.png'),
                            normalize=True, scale_each=True, nrow=4)
        
        # Save top-k filenames
        with open(os.path.join(save_dir, 'topk_files.txt'), 'w') as f:
            for i in topk_idx:
                f.write(f"{img_list[i]}\n")
    else:
        print(f"Skipping extraction for channel {channel} as {save_dir}/patches.png already exists and overwrite is False.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_layer', type=str, default='layer4')
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--topk_patches', type=int, default=50)
    parser.add_argument('--channel', type=int, default=1935)
    parser.add_argument('--overwrite', type=int, default=0)
    args = parser.parse_args()
    print(args)
    
    model = generate_model(args.arch)

    extract_top_patches(
        model=model,
        layer_name=args.target_layer,
        patch_size=args.patch_size,
        channel=args.channel,
        topk_patches=args.topk_patches,
        arch=args.arch,
        overwrite=bool(args.overwrite)
    )
