# This is a training script adapted from https://github.com/Hsankesara/VoxelMorph-PyTorch
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import argparse
from MRIDataset import PairedImageDataset
from networks import VxmDense
from tqdm import tqdm
import Loss as L
import nibabel as nib
import layers
from functools import partial

class RegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ncc = L.NCC(win=21)
        self.l1 = torch.nn.SmoothL1Loss()
        
    def forward(self, moving_warped, fixed, df, df_with_grid):
        sim_loss = self.ncc(moving_warped, fixed)
        Jdet = L.neg_Jdet_loss(df_with_grid, margin = 0, normalized_coord = True)
        Jdet_loss = (10.*Jdet + Jdet**2).mean()
        v_loss = (10.*df.abs()  + df**2).mean()
        
        return sim_loss, Jdet_loss, v_loss, sim_loss + 0.01* Jdet_loss + 0.001*v_loss
        
def main(args):
    # make output directory
    os.makedirs(args.ckptepath, exist_ok = True)
    os.makedirs(args.resultpath, exist_ok = True)
    
    # Data loader
    training_set = PairedImageDataset(data_dir = args.datadir)
    training_loader = data.DataLoader(training_set, batch_size= args.bsize,
              shuffle = True,
              num_workers = args.num_worker,
              worker_init_fn = np.random.seed(42))
    
    # Model
    model = VxmDense(inshape=(160, 192, 144),
                 integrator = layers.VecInt if args.integrator =='single' else partial(layers.MultiVecInt, num_V=args.integrator_multiply),
                 integrator_multiply = 1 if args.integrator =='single' else args.integrator_multiply)
    model = model.to(args.gpu_id)
    
    # Loss function 
    rLoss = RegLoss().to(args.gpu_id)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Loop over epochs
    for epoch in range(args.epochs):
        progress_bar = tqdm(training_loader)
        idx = 0
        for batch in progress_bar:
            fixed, moving = batch["fixed"], batch["moving"]
            fixed = fixed.to(args.gpu_id).unsqueeze(1)
            moving = moving.to(args.gpu_id).unsqueeze(1)
            moving_warped, df, df_with_grid = model(source = moving, target = fixed, registration = True)
            sim_loss, Jdet_loss, v_loss, loss = rLoss(moving_warped, fixed, df, df_with_grid)
            opt.zero_grad()
            loss.backward()
            opt.step()
            progress_bar.set_description(f"Loss: {sim_loss.item():.3g}, {Jdet_loss.item():.3g}, {v_loss.item():.3g}, {loss.item():.3g}")
			
        # save model after each epoch
        model.save(osp.join(args.resultpath, f"model_{epoch}.pth"))
        
        # Save the fixed and moving for visualization
        nib.save(nib.Nifti1Image(fixed.squeeze(0).squeeze(0).detach().cpu().numpy(), np.eye(4)), 
                 osp.join(args.resultpath, "fixed.nii.gz"))
        nib.save(nib.Nifti1Image(moving.squeeze(0).squeeze(0).detach().cpu().numpy(), np.eye(4)), 
                 osp.join(args.resultpath, "moving.nii.gz"))
        nib.save(nib.Nifti1Image(moving_warped.squeeze(0).squeeze(0).detach().cpu().numpy(), np.eye(4)), 
                 osp.join(args.resultpath, "moving_warped.nii.gz"))
        
        
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Voxelmorph')
    
    # Data options
    parser.add_argument('--datadir', default='./data',
                        help='Data path for Kinetics')
    
    parser.add_argument('--ckptepath', type=str, default='./checkpoints2',
                        help='Path for checkpoints and logs')
    
    parser.add_argument('--resultpath', type=str, default='./results2',
                        help='Path for checkpoints and logs')
    
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint file to resume')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--bsize', type=int, default=1,
                        help='batch size for training (default: 1)')
    parser.add_argument('--num_worker', type=int, default=8,
                        help='number of dataloader threads')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='which gpu to use')
    
    parser.add_argument('--integrator', type=str, default='multi',
                        help='which integrator to use. Either single or multi')
    
    parser.add_argument('--integrator_multiply', type=int, default=3,
                        help='how many velocity field to use in the multi integrator')
    
    args = parser.parse_args()
    
    
    
    main(args)
