import os
import csv
import glob

def main():
    base_dir = "output/Bonn"
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    # Find all mapping_metrics.csv files in subdirectories of output/Bonn
    csv_files = glob.glob(os.path.join(base_dir, "*", "mapping_metrics.csv"))
    
    if not csv_files:
        print("No mapping_metrics.csv files found.")
        return

    results = []
    
    for filepath in csv_files:
        seq_name = os.path.basename(os.path.dirname(filepath))
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                row = next(reader)
                psnr, ssim, lpips = float(row[0]), float(row[1]), float(row[2])
                results.append({
                    "Sequence": seq_name,
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "LPIPS": lpips
                })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    if not results:
        print("No valid data parsed.")
        return

    # Sort results by sequence name for clean output
    results.sort(key=lambda x: x["Sequence"])

    # Compute averages
    avg_psnr = sum(r["PSNR"] for r in results) / len(results)
    avg_ssim = sum(r["SSIM"] for r in results) / len(results)
    avg_lpips = sum(r["LPIPS"] for r in results) / len(results)

    # Write to consolidated CSV
    output_csv = "output/Bonn_mapping_eval_consolidated.csv"
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sequence", "PSNR", "SSIM", "LPIPS"])
        
        for r in results:
            writer.writerow([r["Sequence"], f"{r['PSNR']:.4f}", f"{r['SSIM']:.4f}", f"{r['LPIPS']:.4f}"])
            
        # Empty row for spacing
        writer.writerow([])
        # Write average row
        writer.writerow(["Average", f"{avg_psnr:.4f}", f"{avg_ssim:.4f}", f"{avg_lpips:.4f}"])

    print(f"Successfully consolidated {len(results)} sequences into {output_csv}")
    print(f"Averages -> PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")

if __name__ == "__main__":
    main()
