import os
import numpy as np
import re
import pandas as pd
from datetime import datetime

def calculate_mse(trues_path, preds_path):
    trues = np.load(trues_path)
    preds = np.load(preds_path)
    origin_shape = trues.shape
    trues, preds = trues[:, :, :min(18, trues.shape[-1])], preds[:, :, :min(18, preds.shape[-1])]

    trues_reshaped = trues.transpose(1, 0, 2).reshape(3, -1)
    preds_reshaped = preds.transpose(1, 0, 2).reshape(3, -1)

    mse = np.mean((trues_reshaped - preds_reshaped) ** 2, axis=-1)
    return mse, origin_shape

def process_directory(result_dir, log_path):
    results = []
    with open(log_path, "w") as log_file:
        for root, _, files in os.walk(result_dir):
            for file in files:
                if file.endswith("_trues.npy"):
                    prefix = file[:-10]  # 去掉 _trues.npy
                    trues_path = os.path.join(root, file)
                    preds_path = os.path.join(root, prefix + "_pred.npy")

                    if not os.path.exists(preds_path):
                        continue

                    mse, shape = calculate_mse(trues_path, preds_path)
                    log_line = f"True: {prefix}_trues.npy, Shape: {shape}, MSE for each of the targets: {mse.tolist()}"
                    print(log_line)
                    log_file.write(log_line + "\n")

                    results.append((prefix, shape[2], mse))
    return results

def generate_excel(results, output_path):
    regions_map = {
        "Gasoline PricesEast CoastNew England": ["Gasoline Prices", "East Coast", "New England"],
        "Central AtlanticLower AtlanticMidwest": ["Central Atlantic", "Lower Atlantic", "Midwest"],
        "Gulf CoastRocky MountainWest Coast": ["Gulf Coast", "Rocky Mountain", "West Coast"]
    }
    pred_lengths = ["12", "24", "36", "48"]
    region_names = sum(regions_map.values(), [])

    df = pd.DataFrame(index=region_names, columns=pred_lengths)

    for prefix, pred_len, mse in results:
        for group_name, region_list in regions_map.items():
            if prefix.startswith(group_name):
                for region, value in zip(region_list, mse):
                    df.loc[region, str(pred_len)] = value

    df = df.astype(float)
    df["AVG"] = df.mean(axis=1)
    df.loc["AVG"] = df.mean(numeric_only=True)
    df.to_excel(output_path)
    print(f"表格已写入 {output_path}")

def main():
    result_dir = "/data/dyl/ReTATSF/test_results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"target_test_log_{timestamp}.txt"
    excel_path = f"target_mse_results_{timestamp}.xlsx"

    results = process_directory(result_dir, log_path)
    generate_excel(results, excel_path)

if __name__ == "__main__":
    main()
