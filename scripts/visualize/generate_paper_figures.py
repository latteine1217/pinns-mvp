#!/usr/bin/env python3
"""
Paper Figure Generator
遵循論文寫作最佳實踐的圖表生成工具：
1. 嚴格限制每張圖的子圖數量 (Max 4)
2. 使用表格呈現純量指標 (Table over Bar Chart)
3. 專注於 Vanilla vs Proposed 的關鍵對比

Usage:
    python scripts/visualize/generate_paper_figures.py \
        --mode field_comparison \
        --checkpoint checkpoints/experiment/best_model.pth \
        --output-dir results/paper_figures_v2

    python scripts/visualize/generate_paper_figures.py \
        --mode metrics_table \
        --results-json results/comparison/results.json
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import json
import logging
from pathlib import Path
import sys
import pandas as pd

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from pinnx.evals.metrics import relative_L2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PaperFigures")

# Style configuration
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

class PaperFigureGenerator:
    def __init__(self, output_dir="results/paper_figures_v2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_field_row(self, coords, truth, pred, error, field_name, save_name):
        """
        Generates a 1x3 comparison: Truth | Pred | Error
        Strictly 3 subplots.
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        
        # Determine strict bounds from truth (ground truth is the reference)
        vmin, vmax = truth.min(), truth.max()
        
        # 1. Truth
        im0 = axes[0].scatter(coords[:, 0], coords[:, 1], c=truth, s=5, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title(f"True {field_name}")
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # 2. Prediction
        im1 = axes[1].scatter(coords[:, 0], coords[:, 1], c=pred, s=5, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Predicted {field_name}")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # 3. Error
        # Use simple 'Reds' or 'viridis' for error, 0 to max_error
        err_vmax = np.percentile(error, 98) # Robust max
        im2 = axes[2].scatter(coords[:, 0], coords[:, 1], c=error, s=5, cmap='Reds', vmin=0, vmax=err_vmax)
        axes[2].set_title(f"Absolute Error")
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path)
        logger.info(f"Saved field comparison to {save_path}")
        plt.close()

    def generate_metrics_table(self, json_paths, output_name="metrics_comparison"):
        """
        Generates a clean comparison table from multiple JSON result files.
        Outputs CSV and Markdown.
        """
        data = []
        for path in json_paths:
            p = Path(path)
            if not p.exists():
                logger.warning(f"File not found: {p}")
                continue
                
            with open(p, 'r') as f:
                res = json.load(f)
            
            # Handle nested structure if necessary
            # Assuming standard structure: {'metrics': {'u': {'l2_relative': ...}}, ...}
            # Or flat structure depending on file
            
            # Example adaptation for the user's specific json format
            # If the json contains multiple experiments (like comparison_results.json)
            if "vanilla" in res and "full" in res:
                for method_name, content in res.items():
                    row = {
                        "Method": method_name.capitalize(),
                        "L2(u)": content['metrics']['u']['l2_relative'],
                        "L2(v)": content['metrics']['v']['l2_relative'],
                        "L2(p)": content['metrics']['p']['l2_relative'],
                        "Div Mean": content['physics']['divergence_mean_abs']
                    }
                    data.append(row)
            else:
                # Single experiment file
                name = p.parent.name # Use folder name as method name
                metrics = res.get('metrics', {}) or res
                # Try to dig out values
                u_l2 = metrics.get('u', {}).get('l2_relative') or metrics.get('relative_l2_u', 0)
                v_l2 = metrics.get('v', {}).get('l2_relative') or metrics.get('relative_l2_v', 0)
                p_l2 = metrics.get('p', {}).get('l2_relative') or metrics.get('relative_l2_p', 0)
                
                row = {
                    "Method": name,
                    "L2(u)": u_l2,
                    "L2(v)": v_l2,
                    "L2(p)": p_l2,
                }
                data.append(row)

        if not data:
            logger.error("No data extracted for table.")
            return

        df = pd.DataFrame(data)
        
        # Formatting
        df = df.round(4)
        
        # Save CSV
        csv_path = self.output_dir / f"{output_name}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save Markdown
        md_path = self.output_dir / f"{output_name}.md"
        with open(md_path, 'w') as f:
            f.write(df.to_markdown(index=False))
            
        logger.info(f"Saved metrics table to {csv_path} and {md_path}")
        print(df.to_markdown(index=False))

    def generate_generic_table(self, json_path, output_name="table", key_name="ID", columns=None):
        """
        Generates a table from a generic JSON dictionary.
        JSON format: { "RowKey": { "Col1": val, "Col2": val }, ... }
        """
        p = Path(json_path)
        if not p.exists():
            logger.error(f"File not found: {p}")
            return

        with open(p, 'r') as f:
            data = json.load(f)

        rows = []
        for key, metrics in data.items():
            row = {key_name: key}
            # If metrics is not a dict (e.g. simple value), handle it? 
            # Assuming dict for now based on use cases.
            if isinstance(metrics, dict):
                row.update(metrics)
            else:
                row["Value"] = metrics
            rows.append(row)

        df = pd.DataFrame(rows)
        
        # Filter/Order columns if specified
        if columns:
            # Ensure key_name is in columns or add it
            if key_name not in columns:
                cols_to_use = [key_name] + columns
            else:
                cols_to_use = columns
            
            # Only keep columns that exist in df
            cols_to_use = [c for c in cols_to_use if c in df.columns]
            df = df[cols_to_use]
        
        # Save
        csv_path = self.output_dir / f"{output_name}.csv"
        df.to_csv(csv_path, index=False)
        
        md_path = self.output_dir / f"{output_name}.md"
        with open(md_path, 'w') as f:
            f.write(df.to_markdown(index=False))
            
        logger.info(f"Saved generic table to {md_path}")
        print(df.to_markdown(index=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['field_comparison', 'metrics_table', 'rans_stats_table', 'generic_table'], required=True)
    parser.add_argument('--checkpoint', help='Path to model checkpoint')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--results-json', nargs='+', help='Path to results JSON file(s)')
    parser.add_argument('--output-dir', default='results/paper_figures_v2')
    parser.add_argument('--output-prefix', default='', help='Prefix for output filenames')
    parser.add_argument('--key-name', default='ID', help='Header for the primary key column (generic_table mode)')
    parser.add_argument('--columns', nargs='+', help='List of columns to include (generic_table mode)')
    
    args = parser.parse_args()
    
    gen = PaperFigureGenerator(args.output_dir)
    
    if args.mode == 'metrics_table':
        if not args.results_json:
            print("Error: --results-json required for metrics_table mode")
            return
        gen.generate_metrics_table(args.results_json)

    elif args.mode == 'rans_stats_table':
        if not args.results_json:
            print("Error: --results-json required for rans_stats_table mode")
            return
        gen.generate_rans_stats_table(args.results_json[0])

    elif args.mode == 'generic_table':
        if not args.results_json:
            print("Error: --results-json required for generic_table mode")
            return
        # Use filename stem as default output name if not specified (we use 'output-prefix' + name)
        name = Path(args.results_json[0]).stem
        if args.output_prefix:
            name = f"{args.output_prefix}_{name}"
            
        gen.generate_generic_table(args.results_json[0], output_name=name, key_name=args.key_name, columns=args.columns)
        
    elif args.mode == 'field_comparison':
        if not args.checkpoint:
            print("Error: --checkpoint required for field_comparison mode")
            return
            
        # Load model and inference (Simplified integration)
        from pinnx.train.factory import create_model, get_device
        device = get_device('auto')
        
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        config = ckpt['config']
        
        model = create_model(config, device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        # Load Data (Assuming Kolmogorov for now, easy to extend)
        # In a real generic script, we'd reuse the Loader classes.
        # Here we do a quick load for demonstration/fixing.
        if config.get('physics', {}).get('type') == 'kolmogorov_flow_2d':
            import h5py
            data_path = config['data']['kolmogorov_config']['data_path']
            print(f"Loading data from {data_path}")
            with h5py.File(data_path, 'r') as f:
                # Pick t=35.0 or similar
                t_idx = 35 
                u_true = f['u'][t_idx]
                v_true = f['v'][t_idx]
                
                nx, ny = u_true.shape
                x = np.linspace(0, 2*np.pi, nx)
                y = np.linspace(0, 2*np.pi, ny)
                X, Y = np.meshgrid(x, y, indexing='ij')
                
                coords = np.stack([np.full_like(X, f['time'][t_idx]), X, Y], axis=-1)
                coords_flat = coords.reshape(-1, 3)
                
                # Inference
                with torch.no_grad():
                    inp = torch.tensor(coords_flat, dtype=torch.float32).to(device)
                    out = model(inp).cpu().numpy()
                    
                u_pred = out[:, 0].reshape(nx, ny)
                v_pred = out[:, 1].reshape(nx, ny)
                
                prefix = f"{args.output_prefix}_" if args.output_prefix else ""
                
                # Plot U
                gen.plot_field_row(coords.reshape(-1, 3)[:, 1:], u_true.flatten(), u_pred.flatten(), 
                                   np.abs(u_true - u_pred).flatten(), "Velocity U", f"{prefix}Fig_U_Comparison")
                                   
                # Plot V
                gen.plot_field_row(coords.reshape(-1, 3)[:, 1:], v_true.flatten(), v_pred.flatten(), 
                                   np.abs(v_true - v_pred).flatten(), "Velocity V", f"{prefix}Fig_V_Comparison")
                                   
        else:
            print("Only Kolmogorov 2D currently supported for quick figure gen.")

if __name__ == "__main__":
    main()
