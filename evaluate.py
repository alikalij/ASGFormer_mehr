# evaluate.py

import argparse
from configs.env_config import CONFIG
from data.dataset import H5Dataset, PointCloudProcessor, read_file_list
from torch_geometric.loader import DataLoader
from evaluator import Evaluator # ✅ وارد کردن کلاس Evaluator
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the ASGFormer model.")
    parser.add_argument('--mode', type=str, required=True, choices=['test', 'plot', 'visualize'],
                        help="Evaluation mode: 'test' for metrics, 'plot' for loss curve, 'visualize' for 3D visualization.")
    parser.add_argument('--num_samples', type=int, default=3, help="Number of samples to visualize (for visualize mode).")
    parser.add_argument('--dataset_split', type=str, default='val', choices=['train', 'val', 'test'], 
                        help="Which dataset split to use for testing/visualization.")
    args = parser.parse_args()

    # --- ساخت Evaluator ---
    print("Initializing Evaluator...")
    evaluator = Evaluator(config=CONFIG)

    # --- اجرای حالت انتخاب شده ---
    if args.mode == 'plot':
        evaluator.plot_loss()
    
    else: 
        # --- آماده‌سازی دیتا لودر برای تست یا بصری‌سازی ---
        split_file = f"{args.dataset_split}5.txt" # e.g., val5.txt
        file_list = read_file_list(os.path.join(CONFIG['dataset_path'], "list", split_file))
        processor = PointCloudProcessor(num_points=CONFIG['num_points'])
        dataset = H5Dataset(file_list, processor, CONFIG['dataset_path'])
        
        # 💡 نکته: برای تست دقیق، batch_size=1 بهتر است تا هر نمونه جداگانه پردازش شود.
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2) 
        
        if args.mode == 'test':
            evaluator.test(dataloader)
        
        elif args.mode == 'visualize':
            evaluator.visualize(dataloader, args.num_samples)

    print("Evaluation finished.")