import os
import argparse
import torch
from cp_dataset import CPDataset
from networks import GMM  # make sure this is your fixed GMM model
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='data/test', help='root path of test data')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/GMM/gmm_final.pth', help='GMM checkpoint')
    parser.add_argument('--test_pairs', type=str, default='test_pairs.txt', help='test pairs file')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--result_dir', type=str, default='result')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, e.g. 0 or 0,1')

    opt = parser.parse_args()

    os.makedirs(opt.result_dir, exist_ok=True)

    # Use GPU if available
    device = torch.device(f"cuda:{opt.gpu_ids}" if torch.cuda.is_available() else "cpu")
    
    # Dataset & Loader
    dataset = CPDataset(opt)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    # Load GMM model
    model = GMM()  # make sure GMM init does not require extra args
    if os.path.exists(opt.checkpoint):
        checkpoint = torch.load(opt.checkpoint, map_location=device)
        model.load_state_dict(checkpoint, strict=False)  # mismatch-safe
        print(f"[Info] Loaded checkpoint {opt.checkpoint}")
    else:
        print(f"[Warning] Checkpoint {opt.checkpoint} not found. Using uninitialized model.")

    model.to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # Example: move inputs to GPU
            for k in batch:
                batch[k] = batch[k].to(device)
            
            output = model(batch)
            
            # Save results
            for j, out_img in enumerate(output):
                save_path = os.path.join(opt.result_dir, f"{i*opt.batch_size + j}.png")
                # assuming output is tensor, convert to image
                img = out_img.cpu().float().clamp(0, 1)
                img = (img * 255).byte().permute(1, 2, 0).numpy()
                from PIL import Image
                Image.fromarray(img).save(save_path)

    print("[Info] Test finished. Results saved to:", opt.result_dir)

if __name__ == "__main__":
    main()
