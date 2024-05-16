from time import time
from pathlib import Path
import argparse
import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image, ImageDraw
import sys
import os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from src.core.yaml_config import YAMLConfig


class ImageReader:
    def __init__(self, resize=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            # transforms.Resize((resize, resize)) if isinstance(resize, int) else transforms.Resize(
            #     (resize[0], resize[1])),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])
        self.resize = resize
        self.pil_img = None   # Save latest read image

    def __call__(self, image_path, *args, **kwargs):
        """
        Read image
        """
        self.pil_img = Image.open(image_path).convert('RGB').resize((self.resize, self.resize))
        return self.transform(self.pil_img).unsqueeze(0)


class Model(nn.Module):
    def __init__(self, confg=None, ckpt="") -> None:
        super().__init__()
        self.cfg = YAMLConfig(confg, resume=ckpt)
        if ckpt:
            checkpoint = torch.load(ckpt, map_location='cpu')
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
        else:
            raise AttributeError('only support resume to load model.state_dict by now.')

        # NOTE load train mode state -> convert to deploy mode
        self.cfg.model.load_state_dict(state)

        self.model = self.cfg.model.deploy()
        self.postprocessor = self.cfg.postprocessor.deploy()
        # print(self.postprocessor.deploy_mode)

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--source", type=str)
    parser.add_argument("--output_dir", type=str)

    return parser


def main(args):
    from tqdm import tqdm

    src_path = Path(args.source)

    # if src_path is a directory, get all the image files in it
    if src_path.is_dir():
        src_path = src_path.glob('*.jpg')
        src_images = list(src_path)
    elif src_path.is_file() and src_path.suffix == '.jpg':
        src_images = [src_path]

    device = torch.device('cuda')
    reader = ImageReader(resize=640)
    model = Model(confg=args.config, ckpt=args.ckpt)
    model.to(device=device)

    warmup_iters = 5
    count = 0
    elapse_seconds = 0
    for i, img_path in tqdm(enumerate(src_images), total=len(src_images)):
        img = reader(img_path).to(device)
        size = torch.tensor([[img.shape[2], img.shape[3]]]).to(device)

        t0 = time()
        output = model(img, size)
        if i >= warmup_iters:
            elapse_seconds += time() - t0
            count += 1

        if args.output_dir:
            labels, boxes, scores = output
            im = reader.pil_img
            draw = ImageDraw.Draw(im)
            thrh = 0.6

            for i in range(img.shape[0]):
                scr = scores[i]
                lab = labels[i][scr > thrh]
                box = boxes[i][scr > thrh]

                for b in box:
                    draw.rectangle(list(b), outline='red', )
                    draw.text((b[0], b[1]), text=str(lab[i]), fill='blue', )

            save_path = Path(args.output_dir) / img_path.name
            im.save(save_path)

    print(f"Mean inference time:{elapse_seconds / count * 1000:.1f} ms")


if __name__ == "__main__":
    parser = parse_arguments()
    main(parser.parse_args())
