import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image

base_output_dir = "gifs"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_names", required=True, help="exp_names")
parser.add_argument("--results_dir", default="./results/", type=str, help="results_dir")
parser.add_argument("--epoch", default="latest", help="which epoch")
parser.add_argument("--phase", default="test", help="which phase")
parser.add_argument("--dataroot", required=True, help="path to images")
parser.add_argument("--interval", default=0.05, type=float, help="time interval")
parser.add_argument("--duration", default=100, type=int, help="duration per frame in ms")
opt, _ = parser.parse_known_args()

exp_list = opt.exp_names.split(",")

for exp_name in exp_list:
    current_output_dir = os.path.join(opt.results_dir, base_output_dir, exp_name)
    os.makedirs(current_output_dir, exist_ok=True)

    test_img_dir = os.path.join(opt.dataroot, "test", "img")
    if not os.path.exists(test_img_dir):
        print(f"Directory not found: {test_img_dir}")
        continue

    for sample_idx in os.listdir(test_img_dir):
        filenames = []
        images = []
        num_str = sample_idx

        for i in range(int(1 / opt.interval)):
            c_name = os.path.join(
                opt.results_dir,
                exp_name,
                f"{opt.phase}_{opt.epoch}",
                "images",
                f"{num_str}_fake_B_list{i}.png",
            )

            if os.path.exists(c_name):
                filenames.append(c_name)
            else:
                print(f"File not found: {c_name}")
                continue

        if not filenames:
            print(f"No files found for sample {sample_idx}")
            continue

        for filename in filenames:
            try:
                img = imageio.imread(filename)
                if isinstance(img, np.ndarray):
                    images.append(img)
                else:
                    print(f"Unexpected image type in {filename}: {type(img)}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

        if not images:
            print(f"No valid images for sample {sample_idx}")
            continue

        output_path = os.path.join(current_output_dir, f"{sample_idx}_{opt.epoch}.gif")

        try:
            pil_images = [Image.fromarray(img) for img in images]
            pil_images[0].save(
                output_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=opt.duration,
                loop=0,
                optimize=True,
            )
            print(f"Successfully saved with PIL: {output_path}")
        except Exception as e:
            print(f"Error saving GIF with PIL {output_path}: {e}")
