import logging, argparse
import os, torch, cv2, mrcfile
from tqdm import tqdm
import numpy as np
from PIL import Image

from genem.transform import icebreaker_transform
from genem.utils import mkdirs

logger = logging.getLogger(__name__)

def add_args(parser):
    # input
    parser.add_argument("--apix", type=float, default=1.0, help="apix of input micrographs")
    parser.add_argument("--input_dir", type=str, required=True, help="input micrograph diectory")
    # ouput
    parser.add_argument("--save_dir", type=str, required=True, help="weight map save directory")
    parser.add_argument("--output_len", type=int, default=1024, help="Output weight map size. [output_len, output_len]")
    # general
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser

def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

def warnexists(out):
    if os.path.exists(out):
        logger.warning("Warning: {} already exists. Overwriting.".format(out))
        
def main(args):
    mic_dir = args.input_dir
    save_dir = args.save_dir
    map_len = args.output_len 
    device = args.device
    
    icebreaker_params = {
        'apix': args.apix,
        'x_patch_num': 16, 
        'y_patch_num': 16, 
        'cluster_num': 6
    }
    
    image_list = os.listdir(mic_dir)
    image_list.sort()
    
    mkbasedir(save_dir)
    warnexists(save_dir)
    mkdirs(save_dir)
    
    dataset_id = os.path.basename(os.path.dirname(mic_dir))
    
    for image_name in tqdm(image_list, "Calculate weight map..."):
        src = os.path.join(mic_dir, image_name)
        image_name = f'{dataset_id}_{image_name[:-4]}.png'
        tgt = os.path.join(save_dir, image_name)
        if src.endswith(".mrc"):
            img = mrcfile.open(src, permissive=True).data.copy()
        else:
            img = Image.open(src).convert("L")
            img = np.array(img)
        img = torch.from_numpy(img).to(device)
        img = img[None, None, :, :]
        img = (img - img.min()) / (img.max() - img.min())
        weight_map = icebreaker_transform(img, icebreaker_params, return_map=True, map_len=map_len)
        weight_map = weight_map[0].cpu().numpy()
        size_k = map_len // 5
        if size_k % 2 == 0: size_k += 1
        weight_map = cv2.GaussianBlur(weight_map, (size_k, size_k), 0)
        weight_map = (weight_map * 255).astype(np.uint8)
        cv2.imwrite(tgt, weight_map) 
        
    logger.info(f"Save weight map at {save_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
