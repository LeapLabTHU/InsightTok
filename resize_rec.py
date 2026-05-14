import os
import sys
import torch
from tqdm import tqdm
import argparse
from PIL import Image
import  numpy as np
import math



def smart_padding(img, target_size=(512, 512), background_color=(0, 0, 0)):
    """
    padding to target size, keep the image ori ratio and return the metainfo
    """
    original_width, original_height = img.size
    target_width, target_height = target_size
    
    ratio = min(target_width / original_width, target_height / original_height)
    new_size = (math.ceil(original_width * ratio), math.ceil(original_height * ratio))
    resized_img = img.resize(new_size, Image.LANCZOS)
    
    padded_img = Image.new("RGB", target_size, background_color)
    
    paste_position = (
        (target_width - new_size[0]) // 2,
        (target_height - new_size[1]) // 2
    )
    
    padded_img.paste(resized_img, paste_position)
    
    meta = {
        "original_size": (original_width, original_height),
        "scaled_size": new_size,
        "paste_position": paste_position,
        "target_size": target_size
    }
    
    return padded_img, meta

def restore_original(padded_img, meta):
    """
    resize and crop to ori size
    """
    scaled_img = padded_img.crop((
        meta["paste_position"][0],
        meta["paste_position"][1],
        meta["paste_position"][0] + meta["scaled_size"][0],
        meta["paste_position"][1] + meta["scaled_size"][1]
    ))
    restored_img = scaled_img.resize(
        meta["original_size"], 
        Image.LANCZOS
    )
    return restored_img
