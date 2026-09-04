import os
import argparse
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import sam3

from sam3.sam3 import build_sam3_image_model
from sam3.sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.sam3.eval.postprocessors import PostProcessImage
from sam3.sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3.sam3.train.data.collator import collate_fn_api as collate
from sam3.sam3.model.utils.misc import copy_data_to_device

# Utilities for building datapoints
GLOBAL_COUNTER = 1

def create_empty_datapoint():
    return Datapoint(find_queries=[], images=[])

def set_image(datapoint, pil_image):
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

def add_text_prompt(datapoint, text_query):
    global GLOBAL_COUNTER
    assert len(datapoint.images) == 1, "please set the image first"
    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[], 
            is_exhaustive=True, 
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True, help="Directory containing images")
    parser.add_argument("--text_prompt", type=str, required=True, help="Object class to segment (e.g., 'person')")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the masks")
    parser.add_argument("--overlay_dir", type=str, default=None, help="Optional directory to save mask visualizations")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)
        
    image_files = glob.glob(os.path.join(args.image_folder, "*.jpg")) + glob.glob(os.path.join(args.image_folder, "*.png"))
    try:
        image_files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        image_files.sort()
        
    if not image_files:
        print(f"No images found in {args.image_folder}")
        return

    # Turn on optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Initializing SAM3 Image Model...")
    bpe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    
    # Use bfloat16 for efficiency
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        model = build_sam3_image_model(bpe_path=bpe_path)
        model = model.cuda()
        model.eval()

        transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=0.5,
            to_cpu=True,
        )

        # Batch process images
        print(f"Processing {len(image_files)} images in batches of {args.batch_size}...")
        for i in tqdm(range(0, len(image_files), args.batch_size), desc="Inferring masks"):
            batch_files = image_files[i:i+args.batch_size]
            
            datapoints = []
            for img_path in batch_files:
                pil_image = Image.open(img_path).convert("RGB")
                dp = create_empty_datapoint()
                set_image(dp, pil_image)
                add_text_prompt(dp, args.text_prompt)
                dp = transform(dp)
                datapoints.append(dp)
                
            # Collate and move to GPU
            batch = collate(datapoints, dict_key="dummy")["dummy"]
            batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
            
            # Forward
            output = model(batch)
            
            # Postprocess
            processed_results = postprocessor.process_results(output, batch.find_metadatas)
            
            # Save results
            for idx, filename in enumerate(batch_files):
                filename = os.path.basename(filename)
                mask_save_name = os.path.splitext(filename)[0] + ".png"
                mask_save_path = os.path.join(args.save_dir, mask_save_name)
                
                # batch.find_metadatas is a list of stages. We have 1 stage.
                img_id = batch.find_metadatas[0].coco_image_id[idx].item()
                if img_id not in processed_results:
                    # No masks detected at all
                    h, w = batch.find_metadatas[0].original_size[idx]
                    final_mask = np.zeros((int(h), int(w)), dtype=np.uint8)
                else:
                    query_result = processed_results[img_id]
                    masks = query_result["masks"] # boolean tensor [N, H, W]
                    
                    if masks.shape[0] > 0:
                        # Merge all detected instances of the class into one binary mask
                        final_mask = (masks.any(dim=0).squeeze().cpu().numpy() * 255).astype(np.uint8)
                    else:
                        h, w = batch.find_metadatas[0].original_size[idx]
                        final_mask = np.zeros((int(h), int(w)), dtype=np.uint8)

                cv2.imwrite(mask_save_path, final_mask)
                
                if args.overlay_dir:
                    orig_img = cv2.imread(batch_files[idx])
                    color_mask = np.zeros_like(orig_img)
                    color_mask[final_mask == 255] = [0, 0, 255] # Red mask
                    overlay = cv2.addWeighted(orig_img, 0.5, color_mask, 0.5, 0)
                    cv2.imwrite(os.path.join(args.overlay_dir, filename), overlay)

if __name__ == "__main__":
    main()
