import numpy as np
import json
import pandas as pd
import os

datasets = os.listdir('./output')
for dataset in datasets:
    if not os.path.isdir(os.path.join('output', dataset)):
        continue
    dataset_path = os.path.join('output', dataset)
    scenes = sorted(os.listdir(dataset_path))

    data = {scene: [] for scene in scenes}
    averages = []

    row_data = []
    rmses = []
    for scene in scenes:
        exp_folder = os.path.join(dataset_path, scene)
        # metrics_full_traj, metrics_kf_traj, metrics_kf_traj_before_ba
        result_file = os.path.join(exp_folder, "traj/metrics_full_traj.txt")
        if os.path.exists(result_file):
            # Load the JSON file
            with open(result_file, "r") as f:
                output = f.readlines()

            rmse = float(output[8].split(',')[0].replace("{'rmse': ",''))
            
            # Add metrics to the row: multiply by 100, truncate to 2 decimal places (no rounding)
            val_str = f"{rmse * 100:.6f}"
            truncated = val_str[:val_str.index('.') + 3]
            row_data.append(truncated)
            rmses.append(rmse)
        else:
            row_data.append("N/A")  # If file doesn't exist, mark it as N/A
            
    avg_rmse = np.nanmean(rmses) if rmses else np.nan
    if not np.isnan(avg_rmse):
        avg_str = f"{avg_rmse * 100:.6f}"
        averages.append(avg_str[:avg_str.index('.') + 3])
    else:
        averages.append("N/A")
    for scene, value in zip(scenes, row_data):
        data[scene].append(value)

    data['Average'] = averages

    # Convert the data to a Pandas DataFrame
    df = pd.DataFrame(data, index=['ours'])

    # Save the DataFrame as a CSV file
    csv_path = f"./output/{dataset}_eval.csv"
    df.to_csv(csv_path)

    # Output the CSV file path
    print(f"Results saved to {csv_path}")
        