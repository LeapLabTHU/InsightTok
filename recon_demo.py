import os
import argparse
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from resize_rec import smart_padding, restore_original
from models.insight_tok import InsightTok


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
BACKGROUND_COLOR = "white"
TEXT_COLOR = (20, 20, 20)
TILE_PADDING = 24
IMAGE_GAP = 16
GRID_GAP = 24
LABEL_HEIGHT = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Insigh tTok image reconstruction on one image or a directory."
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default='',
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("assets/valset"),
        help="Input image path or a directory of images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/recon/all_compares.png"),
        help="Path to the single comparison grid image.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Resize inputs to this square size before reconstruction. Use 0 to keep size.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run on, for example 'cuda', 'cuda:0', or 'cpu'.",
    )
    return parser.parse_args()


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    for image_path in sorted(path.rglob("*")):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path





def load_image(path: Path, image_size: int) -> tuple[torch.Tensor, Image.Image]:
    ori_image = Image.open(path).convert("RGB")
    image, meta = smart_padding(ori_image, target_size=(image_size, image_size))

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.mul(2.0).sub(1.0)
    return tensor, ori_image, meta


def tensor_to_image(tensor: torch.Tensor, meta) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    tensor = tensor.add(1.0).mul(127.5).round().byte()
    array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    rec_img = Image.fromarray(array, mode="RGB")
    return restore_original(rec_img, meta)


def draw_centered_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    y: int,
    font: ImageFont.ImageFont,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    draw.text((center_x - text_width // 2, y), text, fill=TEXT_COLOR, font=font)


def make_compare_tile(input_image: Image.Image, recon_image: Image.Image, image_size: int) -> Image.Image:
    input_image, recon_image = input_image.resize((image_size, image_size)), recon_image.resize((image_size, image_size))
    font = ImageFont.load_default(size=45)
    image_width = input_image.width
    image_height = input_image.height
    tile_width = TILE_PADDING * 2 + image_width * 2 + IMAGE_GAP
    tile_height = TILE_PADDING * 2 + LABEL_HEIGHT + image_height
    tile = Image.new("RGB", (tile_width, tile_height), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(tile)

    original_x = TILE_PADDING
    recon_x = TILE_PADDING + image_width + IMAGE_GAP
    label_y = TILE_PADDING
    image_y = TILE_PADDING + LABEL_HEIGHT

    draw_centered_label(draw, "Original", original_x + image_width // 2, label_y, font)
    draw_centered_label(draw, "Reconstruction", recon_x + image_width // 2, label_y, font)

    tile.paste(input_image, (original_x, image_y))
    tile.paste(recon_image, (recon_x, image_y))
    return tile


def save_compare_grid(tiles: list[Image.Image], path: Path, columns: int = 2) -> None:
    if not tiles:
        raise ValueError("No comparison tiles to save.")

    tile_width, tile_height = tiles[0].size
    rows = (len(tiles) + columns - 1) // columns
    grid_width = TILE_PADDING * 2 + columns * tile_width + (columns - 1) * GRID_GAP
    grid_height = TILE_PADDING * 2 + rows * tile_height + (rows - 1) * GRID_GAP
    grid = Image.new("RGB", (grid_width, grid_height), color=BACKGROUND_COLOR)

    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        x = TILE_PADDING + column * (tile_width + GRID_GAP)
        y = TILE_PADDING + row * (tile_height + GRID_GAP)
        grid.paste(tile, (x, y))

    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float32

    model = InsightTok(
        dd_config_key='default',
        codebook_size=16384,
        codebook_embed_dim=256,
        commit_loss_beta=0.02,
        codebook_l2_norm=True,
        codebook_show_usage=True,
        quant_disable_autocast=False,
    )
    if os.path.isfile(args.ckpt_path):
        print(f"Loading checkpoint from {args.ckpt_path}...")
        ckpt_path = args.ckpt_path
    else:
        print("No checkpoint found, download from Hugging Face")
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
                repo_id="yueyang2000/InsightTok",
                filename="insight_tok.pt",
                repo_type="model"     
        )
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")['model']
    model.load_state_dict(checkpoint)
    model.eval().to(device=device, dtype=dtype)
    image_paths = list(iter_images(args.input))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.input}")

    compare_tiles = []
    with torch.inference_mode():
        for image_path in image_paths:
            image_tensor, input_image, meta = load_image(image_path, args.image_size)
            image_tensor = image_tensor.to(device=device, dtype=dtype)
            latent, _, [_, _, indices] = model.encode(image_tensor)
            recon_tensor = model.decode(latent) 
            recon_image = tensor_to_image(recon_tensor, meta)
            compare_tiles.append(make_compare_tile(input_image, recon_image, args.image_size))

    save_compare_grid(compare_tiles, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
