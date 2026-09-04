#!/bin/bash

# Configuration
BATCH_SIZE=18
BASE_DIR="./datasets/TUM_RGBD"
GEN_SCRIPT="mask_generation/generate_masks.py"
MERGE_SCRIPT="mask_generation/merge_masks.py"

# Function to generate and merge masks for multi-class sequences
process_multi_class() {
    local seq_name=$1
    shift
    local classes=("$@")
    
    echo "================================================="
    echo "Processing $seq_name with classes: ${classes[*]}"
    echo "================================================="
    
    local seq_dir="$BASE_DIR/$seq_name"
    local rgb_dir="$seq_dir/rgb"
    local mask_folders=()
    
    # Generate mask for each class
    for class in "${classes[@]}"; do
        local safe_class="${class// /_}"
        local save_dir="$seq_dir/mask_${safe_class}"
        echo "--> Generating masks for class: $class"
        python $GEN_SCRIPT \
            --image_folder "$rgb_dir" \
            --text_prompt "$class" \
            --save_dir "$save_dir" \
            --batch_size $BATCH_SIZE
        mask_folders+=("$save_dir")
    done
    
    # Merge masks
    echo "--> Merging masks for $seq_name"
    python $MERGE_SCRIPT \
        --mask_folders "${mask_folders[@]}" \
        --save_dir "$seq_dir/mask_final" \
        --rgb_folder "$rgb_dir" \
        --overlay_dir "$seq_dir/mask_vis_final"
        
    echo "--> Copying masks to $BASE_DIR/masks/$seq_name"
    mkdir -p "$BASE_DIR/masks/$seq_name"
    cp -r "$seq_dir/mask_final/"* "$BASE_DIR/masks/$seq_name/"
}

# TUM RGB-D dynamic sequences ("person" and "chair")
process_multi_class "rgbd_dataset_freiburg2_desk_with_person" "person" "chair"
process_multi_class "rgbd_dataset_freiburg3_sitting_halfsphere" "person" "chair"
process_multi_class "rgbd_dataset_freiburg3_sitting_rpy" "person" "chair"
process_multi_class "rgbd_dataset_freiburg3_sitting_static" "person" "chair"
process_multi_class "rgbd_dataset_freiburg3_sitting_xyz" "person" "chair"
process_multi_class "rgbd_dataset_freiburg3_walking_halfsphere" "person" "chair"
process_multi_class "rgbd_dataset_freiburg3_walking_rpy" "person" "chair"
process_multi_class "rgbd_dataset_freiburg3_walking_static" "person" "chair"
process_multi_class "rgbd_dataset_freiburg3_walking_xyz" "person" "chair"

echo "All TUM RGB-D sequences processed for multiple classes!"
