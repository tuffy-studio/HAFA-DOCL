import csv
import random
import io
import cv2
import torch
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import torchvision.transforms as transforms

# Fine-tuning dataset
# Input: a CSV file containing two columns: image path and label
# Processing: read the image, apply data augmentation, and return the image tensor and label
# Output: image tensor [3, H, W], label tensor [1]
class FineTuneDataset(Dataset):
    def __init__(self, csv_file, data_augment=True, mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225], if_normalize=True):
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)   # 跳过 header
            for row in reader:
                self.data.append((row[0], int(row[1])))  # (image_path, label)

        self.data_augment = data_augment
        print(f"now using data augmentation: {self.data_augment}")

        self.if_normalize = if_normalize
        print(f"now using normalize: {self.if_normalize}")

        self.transform = ImageAugment(im_res=224, visual_augment=self.data_augment, if_normalize=if_normalize, p=0.5, mean=mean, std=std)
        self.transform_origin = ImageTransform(im_res=224, if_normalize=if_normalize, mean=mean, std=std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print("bad image:", img_path)

            # 随机换一张
            new_idx = torch.randint(0, len(self.data), (1,)).item()
            return self.__getitem__(new_idx)
        
        degraded_img = self.transform(img)
        img_origin = self.transform_origin(img)
        label = torch.tensor(int(label), dtype=torch.long)

        return degraded_img, img_origin, label

# transform the original image to tensor
class ImageTransform:
    def __init__(
        self,
        im_res=224,
        if_normalize=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ):
        self.im_res = im_res
        self.if_normalize = if_normalize

        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img):
        """
        img: PIL.Image
        return: Tensor [3,H,W]
        """

        # resize
        img = F.resize(img, (self.im_res, self.im_res))

        # -------------------
        # Flip
        # -------------------
        if random.random() < 0.5:
            img = F.hflip(img)

        img = F.to_tensor(img)
        if self.if_normalize:
            img = self.normalize(img)
        return img

# degrade the original image
class ImageAugment:
    def __init__(
        self,
        im_res=224,
        visual_augment=True,
        if_normalize=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        p=0.5,
        min_size=32
    ):

        self.im_res = im_res
        self.visual_augment = visual_augment
        self.if_normalize = if_normalize

        self.mean = mean
        self.std = std

        self.p = p
        self.min_size = min_size

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    # --------------------------------------------------
    # utility
    # --------------------------------------------------

    def safe_img(self, img):

        img = np.nan_to_num(img)
        img = np.clip(img, -255, 255)

        return img

    # --------------------------------------------------
    # geometric
    # --------------------------------------------------

    def shift_with_black_border(self, img, level):

        h, w = img.shape[:2]

        ratio = np.interp(level, [1,5], [0.02, 0.20])

        dx = int(random.uniform(-ratio, ratio) * w)
        dy = int(random.uniform(-ratio, ratio) * h)

        M = np.float32([[1,0,dx],[0,1,dy]])

        # 随机黑边或白边
        if random.random() < 0.5:
            v = random.randint(0,20)        # 黑边
        else:
            v = random.randint(235,255)     # 白边

        border = v

        img = cv2.warpAffine(
            img,
            M,
            (w,h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(border,border,border)
        )

        return img

    def resize(self, img, level):

        h, w = img.shape[:2]

        scale = np.interp(level,[1,5],[0.9,0.3])

        if random.random() < 0.5:
            scale = random.uniform(scale,1)
        else:
            scale = random.uniform(1,1.8)

        new_w = max(self.min_size,int(w*scale))
        new_h = max(self.min_size,int(h*scale))

        method = random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA
        ])

        img = cv2.resize(img,(new_w,new_h),interpolation=method)

        return img

    # --------------------------------------------------
    # blur
    # --------------------------------------------------

    def blur(self, img, level):

        sigma = np.interp(level,[1,5],[0.5,6])

        k = int(6*sigma+1)
        k = max(3,min(31,k|1))

        img = cv2.GaussianBlur(img,(k,k),sigma)

        return img

    def mean_blur(self, img, level):

        k = int(np.interp(level,[1,5],[3,11]))
        k = k | 1

        img = cv2.blur(img,(k,k))

        return img

    def defocus_blur(self, img, level):

        radius = int(np.interp(level,[1,5],[2,10]))
        k = radius*2+1

        kernel = np.zeros((k,k))
        cv2.circle(kernel,(radius,radius),radius,1,-1)

        kernel /= kernel.sum()

        img = cv2.filter2D(img,-1,kernel)

        return img

    # --------------------------------------------------
    # grayscale
    # --------------------------------------------------

    def grayscale(self, img, level):

        gray = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY)

        img = np.stack([gray,gray,gray],axis=2)

        return img

    # --------------------------------------------------
    # color
    # --------------------------------------------------

    def saturation(self, img, level):

        img = img.astype(np.uint8)

        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        factor = np.interp(level,[1,5],[0.5,1.8])

        hsv[:,:,1] = np.clip(hsv[:,:,1]*factor,0,255)

        img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        return img

    def color_shift(self, img, level):

        img = img.astype(np.float32)

        shift = np.interp(level,[1,5],[5,40])

        for c in range(3):
            img[:,:,c]+=random.uniform(-shift,shift)

        return img

    def color_quantization(self, img, level):

        img = img.astype(np.uint8)

        k = int(np.interp(level,[1,5],[64,8]))

        Z = img.reshape((-1,3))
        Z = np.float32(Z)

        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0
        )

        _,label,center = cv2.kmeans(
            Z,
            k,
            None,
            criteria,
            5,
            cv2.KMEANS_RANDOM_CENTERS
        )

        center = np.uint8(center)

        res = center[label.flatten()]
        img = res.reshape(img.shape)

        return img

    # --------------------------------------------------
    # noise
    # --------------------------------------------------

    def gaussian_noise(self, img, level):

        img = img.astype(np.float32)

        sigma = np.interp(level,[1,5],[5,35])

        noise = np.random.normal(0,sigma,img.shape)

        img = img + noise

        return img

    def salt_pepper_noise(self, img, level):

        img = img.astype(np.float32)

        amount = np.interp(level,[1,5],[0.002,0.02])

        h,w,c = img.shape

        num = int(amount*h*w)

        coords = (
            np.random.randint(0,h,num),
            np.random.randint(0,w,num)
        )

        img[coords]=255

        coords = (
            np.random.randint(0,h,num),
            np.random.randint(0,w,num)
        )

        img[coords]=0

        return img

    def speckle_noise(self, img, level):

        img = img.astype(np.float32)

        sigma = np.interp(level,[1,5],[0.05,0.3])

        noise = np.random.normal(0,sigma,img.shape)

        img = img + img*noise

        return img

    def poisson_noise(self, img, level):

        img = np.clip(img,0,None)

        vals = len(np.unique(img))
        vals = max(vals,2)

        vals = 2**np.ceil(np.log2(vals))

        img = np.random.poisson(img*vals)/float(vals)

        return img

    # --------------------------------------------------
    # jpeg
    # --------------------------------------------------

    def jpeg_compress(self, img, level):

        img = np.clip(img,0,255).astype(np.uint8)

        q = int(np.interp(level,[1,5],[95,10]))

        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),q]

        success,enc=cv2.imencode(".jpg",img,encode_param)

        if success:
            img=cv2.imdecode(enc,1)

        return img

    # --------------------------------------------------
    # brightness / contrast
    # --------------------------------------------------

    def brightness_increase(self, img, level):

        img=np.clip(img,0,255).astype(np.uint8)

        img=Image.fromarray(img)

        factor=np.interp(level,[1,5],[1.1,1.8])

        enhancer=ImageEnhance.Brightness(img)

        img=enhancer.enhance(factor)

        return np.array(img)

    def brightness_decrease(self, img, level):

        img=np.clip(img,0,255).astype(np.uint8)

        img=Image.fromarray(img)

        factor=np.interp(level,[1,5],[0.9,0.4])

        enhancer=ImageEnhance.Brightness(img)

        img=enhancer.enhance(factor)

        return np.array(img)

    def contrast_adjust(self, img, level):

        img=np.clip(img,0,255).astype(np.uint8)

        img=Image.fromarray(img)

        factor=np.interp(level,[1,5],[0.6,1.6])

        enhancer=ImageEnhance.Contrast(img)

        img=enhancer.enhance(factor)

        return np.array(img)

    # --------------------------------------------------
    # occlusion
    # --------------------------------------------------

    def distractor(self, img, level):

        h, w = img.shape[:2]

        if h < 20 or w < 20:
            return img

        if random.random() < 0.5:

            # 文本大小随level变化
            font_scale = np.interp(level, [1, 5], [0.5, 1.5])
            thickness = random.randint(1, 3)

            text = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))

            # 安全随机范围
            max_x = max(1, w - 50)
            max_y = max(10, h - 10)

            pos = (
                random.randint(0, max_x),
                random.randint(10, max_y)
            )

            cv2.putText(
                img,
                text,
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                ),
                thickness
            )

        return img

    # --------------------------------------------------
    # main
    # --------------------------------------------------

    def __call__(self, img):

        if random.random()<0.5:
            img=F.hflip(img)

        if isinstance(img,Image.Image):
            img=np.array(img)

        img=img.astype(np.float32)

        if self.visual_augment:

            degradations=[

                self.shift_with_black_border,
                self.resize,

                self.blur,
                self.mean_blur,
                self.defocus_blur,

                self.brightness_increase,
                self.brightness_decrease,
                self.contrast_adjust,

                self.grayscale,
                self.saturation,
                self.color_shift,
                self.color_quantization,

                self.gaussian_noise,
                self.salt_pepper_noise,
                self.speckle_noise,
                self.poisson_noise,

                self.jpeg_compress,

                self.distractor
            ]

        else:
            degradations=[]

        num_aug=random.randint(1,4)

        ops=random.sample(degradations,num_aug)

        applied=False

        for d in ops:

            if random.random()<self.p or not applied:

                level=random.randint(1,5)

                img=d(img,level)

                img=self.safe_img(img)

                applied=True

        img=np.clip(img,0,255).astype(np.uint8)

        img=cv2.resize(img,(self.im_res,self.im_res))

        img=Image.fromarray(img)

        img=self.to_tensor(img)

        if self.if_normalize:
            img=self.normalize(img)

        return img

