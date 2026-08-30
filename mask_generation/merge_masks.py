import os
import argparse
import glob
import cv2
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple binary masks into a single mask using logical OR.")
    parser.add_argument("--mask_folders", nargs="+", required=True, help="List of folders containing the masks to merge")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the merged masks")
    parser.add_argument("--overlay_dir", type=str, default=None, help="(Optional) Directory to save colored overlays if you also provide --rgb_folder")
    parser.add_argument("--rgb_folder", type=str, default=None, help="(Optional) Original RGB images folder, required if saving overlays")
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    if args.overlay_dir:
        assert args.rgb_folder is not None, "--rgb_folder is required to generate overlays"
        os.makedirs(args.overlay_dir, exist_ok=True)
        
    if len(args.mask_folders) < 2:
        print("Warning: Only 1 mask folder provided. This will just copy the masks.")
        
    base_folder = args.mask_folders[0]
    mask_files = glob.glob(os.path.join(base_folder, "*.png")) + glob.glob(os.path.join(base_folder, "*.jpg"))
    try:
        mask_files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        mask_files.sort()
        
    if not mask_files:
        print(f"No mask files found in {base_folder}")
        return
        
    print(f"Found {len(mask_files)} files. Merging from {len(args.mask_folders)} folders...")
    
    for base_file in tqdm(mask_files, desc="Merging masks"):
        filename = os.path.basename(base_file)
        
        merged_mask = None
        for folder in args.mask_folders:
            filepath = os.path.join(folder, filename)
            if not os.path.exists(filepath):
                print(f"\nWarning: File {filename} not found in {folder}. Skipping for this folder.")
                continue
                
            mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            # Binarize just to be safe (in case they were saved as 255)
            mask = (mask > 0).astype(np.uint8)
            
            if merged_mask is None:
                merged_mask = mask
            else:
                # Logical OR to merge
                merged_mask = np.logical_or(merged_mask, mask).astype(np.uint8)
                
        if merged_mask is None:
            print(f"Could not load any valid masks for {filename}")
            continue
            
        save_path = os.path.join(args.save_dir, filename)
        cv2.imwrite(save_path, merged_mask * 255)
        
        # Overlay if requested
        if args.overlay_dir and args.rgb_folder:
            rgb_path = os.path.join(args.rgb_folder, filename)
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(args.rgb_folder, os.path.splitext(filename)[0] + ".jpg")
                
            if os.path.exists(rgb_path):
                orig_img = cv2.imread(rgb_path)
                if orig_img is not None:
                    color_mask = np.zeros_like(orig_img)
                    color_mask[merged_mask == 1] = [0, 0, 255] # Red mask
                    overlay = cv2.addWeighted(orig_img, 0.5, color_mask, 0.5, 0)
                    cv2.imwrite(os.path.join(args.overlay_dir, filename), overlay)

if __name__ == "__main__":
    main()
