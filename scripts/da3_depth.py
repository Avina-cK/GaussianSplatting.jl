import argparse
import os

# OpenCV ships its OpenEXR codec disabled, and refuses `.exr` without this. It
# is read at import time, so it has to be set before `cv2` comes in.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2 as cv
import imageio
import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Depth Anything 3 on a directory of images.")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_dir", type=str, help="Directory to write depth maps to.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Images per inference call. Depth Anything 3 reasons across the "
        "views it is given at once, so a batch comes out on one consistent "
        "scale; one image at a time leaves every frame free to drift. Matters "
        "most with --metric, where that drift survives into the output.",
    )
    parser.add_argument(
        "--metric",
        action="store_true",
        help="Write the model's depth unchanged, in metres, as float EXR. The "
        "default normalizes each frame to its own range and writes 16-bit PNG, "
        "which is fine for a relative prior but discards both the scale and any "
        "consistency between frames.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="Pretrained model name or path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    valid_exts = {ext.lstrip("*").lower() for ext in IMAGE_EXTENSIONS}
    images = sorted(
        entry.path
        for entry in os.scandir(args.input_dir)
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_exts
    )
    if not images:
        raise ValueError(f"No images found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device=device)

    batch_size = max(1, args.batch_size)
    warned = False
    focals = []

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        prediction = model.inference(batch)

        if args.metric and not prediction.is_metric and not warned:
            print(
                "WARNING: the model reports this prediction is not metric, but "
                "--metric writes it out as though it were. The values will be "
                "in some arbitrary unit.",
                flush=True,
            )
            warned = True

        for offset, image_path in enumerate(batch):
            depth = prediction.depth[offset]

            # The model processes images at a fixed resolution, so upsample the
            # depth map back to the resolution of the input image.
            with Image.open(image_path) as img:
                width, height = img.size

            # Predicted intrinsics describe the *processed* frame, which is the
            # longest side resized to `process_res`. Reported as-is they look
            # like a wildly wrong camera; scaled to the source they are usable.
            if prediction.intrinsics is not None:
                scale = width / depth.shape[1]
                focals.append(float(np.asarray(prediction.intrinsics)[offset][0, 0]) * scale)

            if depth.shape[:2] != (height, width):
                depth = cv.resize(depth, (width, height), interpolation=cv.INTER_LINEAR)

            out_name = os.path.splitext(os.path.basename(image_path))[0]
            if args.metric:
                # Straight through: normalizing is exactly what a metric prior
                # must not have done to it. `cv.imwrite` stores float32 input as
                # EXR FLOAT channels already, so no precision is given up here.
                cv.imwrite(
                    os.path.join(args.output_dir, out_name + ".exr"),
                    depth.astype(np.float32),
                )
                continue

            valid_mask = depth > 0
            if valid_mask.sum() > 0:
                depth_min = depth[valid_mask].min()
                depth_max = depth[valid_mask].max()
            else:
                depth_min, depth_max = 0.0, 1.0
            depth_norm = ((depth - depth_min) / (depth_max - depth_min + 1e-6)).clip(0, 1)
            depth_u16 = (depth_norm * 65535).astype(np.uint16)

            imageio.imwrite(os.path.join(args.output_dir, out_name + ".png"), depth_u16)

    # The model estimates the camera it thinks took these, which is worth more
    # than a guessed field of view when something downstream needs a focal.
    if focals:
        median = float(np.median(focals))
        with Image.open(images[0]) as img:
            source_width = img.size[0]
        fov = 2.0 * np.degrees(np.arctan(0.5 * source_width / median))
        print(f"estimated focal: median {median:.1f} px at the source resolution "
              f"({np.min(focals):.1f} - {np.max(focals):.1f} over {len(focals)} "
              f"views), i.e. a horizontal field of view of {fov:.1f} degrees")


if __name__ == "__main__":
    main()
