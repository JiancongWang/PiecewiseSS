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

def dice(array1, array2, labels, eps = 1e-4):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    dicem = []
    for idx, label in enumerate(labels):
        top = 2. * torch.logical_and(array1 == label, array2 == label).float().sum()
        bottom = (array1 == label).float().sum() + (array2 == label).float().sum()
        bottom = max(bottom.item(), eps)  # add epsilon
        dicem.append((top / bottom).item())
    return dicem

def main(args):
    # make output directory
    os.makedirs(args.valdir, exist_ok = True)
    
    # Data loader
    val_set = PairedImageDataset(data_dir = args.datadir, istrain = False)
    val_loader = data.DataLoader(val_set, batch_size= args.bsize,
              shuffle = False,
              num_workers = args.num_worker,
              worker_init_fn = np.random.seed(42))
    
    # Model
    model = VxmDense(inshape=(160, 192, 144),
                 integrator = layers.VecInt if args.integrator =='single' else partial(layers.MultiVecInt, num_V=args.integrator_multiply),
                 integrator_multiply = 1 if args.integrator =='single' else args.integrator_multiply)
    model = model.to(args.gpu_id)
    model.load(args.resume, args.gpu_id) # load checkpoints
    
    
    ST_seg = layers.SpatialTransformer((160, 192, 144), mode='nearest').to(args.gpu_id)
    
	
    # Loop over epochs
    progress_bar = tqdm(val_loader)
    idx = 0
    for batch in progress_bar:
        fixed, moving, fixed_seg, moving_seg = batch["fixed"], batch["moving"], batch["fmask"], batch["mmask"]
        
        fixed = fixed.to(args.gpu_id).unsqueeze(1)
        moving = moving.to(args.gpu_id).unsqueeze(1)

        fixed_seg = fixed_seg.to(args.gpu_id).unsqueeze(1)
        moving_seg = moving_seg.to(args.gpu_id).unsqueeze(1)
        
        with torch.no_grad():
            moving_warped, df, df_with_grid = model(source = moving, target = fixed, registration = True)
        
            # Calculate the negdet
            Jdet = L.neg_Jdet_loss(df_with_grid, margin = 0, normalized_coord = True)
            
            Jdmean = Jdet.mean().item() # mean jdet
            Jdpercent = ((Jdet>0).float().sum()/Jdet.numel()).item() # percent jdet
            
            # Calculate dice
            labels = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60]
            
            warped_seg = ST_seg(moving_seg.float(), df, return_phi=False).int()
            
            dicem = dice(fixed_seg, warped_seg, labels, eps = 1e-4)
            
            np.save(osp.join(args.valdir, batch["fixed_dir"][0] + "_" + batch["moving_dir"][0] + ".npy"),
                    np.array([Jdmean, Jdpercent] + dicem))            
            
            progress_bar.set_description(f"Jmean: {Jdmean:.3g}, Jp: {Jdpercent:.3g}, dice: {np.array(dicem).mean() : .3g}")
			
		# save model after each epoc
        if idx%100==0:
            output_path = osp.join(args.valdir, batch["fixed_dir"][0] + "_" + batch["moving_dir"][0])
            
            os.makedirs(output_path, exist_ok = True)
            
            # Save the fixed and moving for visualization
            nib.save(nib.Nifti1Image(fixed.squeeze(0).squeeze(0).detach().cpu().numpy(), np.eye(4)), 
                     osp.join(output_path, "fixed.nii.gz"))
            nib.save(nib.Nifti1Image(moving.squeeze(0).squeeze(0).detach().cpu().numpy(), np.eye(4)), 
                     osp.join(output_path, "moving.nii.gz"))
            nib.save(nib.Nifti1Image(moving_warped.squeeze(0).squeeze(0).detach().cpu().numpy(), np.eye(4)), 
                     osp.join(output_path, "moving_warped.nii.gz"))
        idx+=1
        
        
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Voxelmorph')
    
    # Data options
    parser.add_argument('--datadir', default='./data',
                        help='Data path for data')
    parser.add_argument('--valdir', default='./val',
                        help='Data path for validation result')
    
    parser.add_argument('--resume', type=str, default="./ckpt/model.pth",
                        help='Checkpoint file to resume')
    
    # Training options
    parser.add_argument('--bsize', type=int, default=1,
                        help='batch size for training (default: 1)')
    parser.add_argument('--num_worker', type=int, default=8,
                        help='number of dataloader threads')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='which gpu to use')
    
    parser.add_argument('--integrator', type=str, default='multi',
                        help='which integrator to use. Either single or multi')
    
    parser.add_argument('--integrator_multiply', type=int, default=3,
                        help='how many velocity field to use in the multi integrator')
    
    args = parser.parse_args()
    
    
    
    main(args)
