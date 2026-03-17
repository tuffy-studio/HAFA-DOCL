import argparse
import os
import torch
from src.models.HiViT import HiViT_FT
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


parser = argparse.ArgumentParser(description='Finetune Stage')

parser.add_argument('--weights_path', default="./weights/model.pth", type=str, help='path to model weights')
parser.add_argument('--data_root', required=True, type=str, help='path to val data')

args = parser.parse_args()


# MC dropout次数
MC_TIMES = 10

# 构造模型并加载预训练权重
ft_model = HiViT_FT(use_hierarchical=True,encoder_embed_dim=768,moe=False) 

# init model
mdl_weight = torch.load(args.weights_path, map_location='cpu')
miss, unexpected = ft_model.load_state_dict(mdl_weight, strict=False)
print('now load pretrain model from {:s}, missing keys: {:d}, unexpected keys: {:d}'.format(args.weights_path, len(miss), len(unexpected)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ft_model = ft_model.to(device)

# 先把需要检测的图片路径读入
test_imgs = [os.path.join(args.data_root, f) for f in os.listdir(args.data_root) if f.endswith('.jpg') or f.endswith('.png')]
test_imgs.sort()  # 按字母顺序排序

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.Resize((224, 224)),
    #T.CenterCrop((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])


# !!! 保持dropout开启
ft_model.train()

# 清空submission文件
open("submission.txt", "w").close()

with torch.no_grad():
    for img_path in test_imgs:

        img = Image.open(img_path).convert('RGB')

        probs_list = []

        # 两种输入：原图 + 水平翻转
        for flip in [False, True]:

            if flip:
                img_input = TF.hflip(img)
            else:
                img_input = img

            img_tensor = transform(img_input)  # transform 在这里
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Monte Carlo forward
            for _ in range(MC_TIMES):

                outputs, _ , _ , _ , _  = ft_model(img_tensor)

                probs = torch.softmax(outputs, dim=-1)

                prob_fake = probs[0,1]

                probs_list.append(prob_fake)

        # ensemble平均
        prob = torch.stack(probs_list).mean().item()

        with open('submission.txt', 'a') as f:
            f.write(f"{prob}\n")

print("Inference completed. Predictions saved to submission.txt")

