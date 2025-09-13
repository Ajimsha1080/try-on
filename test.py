import os
import argparse
import torch
from cp_dataset import CPDataset  # your dataset
from networks import GMM  # your GMM model
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='data/test', help='root path of test data')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/GMM/gmm_final.pth', help='path to checkpoint')
    parser.add_argument('--test_pairs', type=str, default='test_pairs.txt', help='test pairs file')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of data loader workers')
    parser.add_argument('--result_dir', type=str, default='result', help='where to save results')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
    opt = parser.parse_args()

    # Create result directory
    os.makedirs(opt.result_dir, exist_ok=True)

    # Load dataset
    dataset = CPDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    # Load GMM model
    model = GMM()  # ✅ No argument needed
    if torch.cuda.is_available():
        torch.cuda.set_device(int(opt.gpu_ids))
        model = model.cuda()

    # Load checkpoint safely
    if os.path.exists(opt.checkpoint):
        checkpoint = torch.load(opt.checkpoint, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint, strict=False)  # allow mismatch-safe loading
        print(f'Loaded checkpoint from {opt.checkpoint}')
    else:
        print(f'Warning: checkpoint {opt.checkpoint} not found!')

    model.eval()

    # Inference loop
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if torch.cuda.is_available():
                for k in batch:
                    if torch.is_tensor(batch[k]):
                        batch[k] = batch[k].cuda()
            output = model(batch)
            # Save or process output as needed
            print(f'Processed batch {i+1}/{len(dataloader)}')

if __name__ == '__main__':
    main()
