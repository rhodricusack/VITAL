#!/usr/bin/env python
"""
Select the most representative neuron for each model and launch SLURM jobs
to run reverse_engineer_neurons.py for reconstruction.
"""

import os
import sys
import subprocess
from pathlib import Path

# Import from compare_models
from compare_models import select_representative_neurons

def launch_reconstruction_jobs(model_dirs, target_layer, model_names=None, max_neurons=999, 
                               partition='debug', time_limit='24:00:00'):
    """
    Select the most representative neuron for each model and launch SLURM jobs.
    
    Args:
        model_dirs: List of model directories (e.g., ['resnet50_BV_g0For60_E60/neuron_layer4.2', ...])
        target_layer: Target layer name (e.g., 'layer4.2')
        model_names: Optional list of model names (will be extracted from paths if None)
        max_neurons: Maximum number of neurons to analyze for selection
        partition: SLURM partition to use
        time_limit: Time limit for each job
    """
    
    if model_names is None:
        # Extract model names from directory paths
        model_names = [Path(d).parent.name for d in model_dirs]
    
    print(f"Selecting most representative neuron for each of {len(model_dirs)} models...")
    print(f"Target layer: {target_layer}\n")
    
    job_ids = []
    
    for model_dir, model_name in zip(model_dirs, model_names):
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")
        
        # Select representative neurons (we'll take the top one)
        try:
            rep_neurons = select_representative_neurons(model_dir, n_neurons=1, max_neurons=max_neurons)
            
            if not rep_neurons:
                print(f"ERROR: No representative neurons found for {model_name}")
                continue
            
            channel = rep_neurons[0]
            print(f"\nMost representative neuron: {channel}")
            
            # Prepare SLURM job
            job_name = f"recon_{model_name}_n{channel}"
            output_file = f"vital_vis/recon_{model_name}_n{channel}_%j.out"
            error_file = f"vital_vis/recon_{model_name}_n{channel}_%j.err"
            
            # Create the sbatch command
            sbatch_cmd = [
                'sbatch',
                f'--job-name={job_name}',
                f'--output={output_file}',
                f'--error={error_file}',
                f'--time={time_limit}',
                '--ntasks=1',
                '--cpus-per-task=4',
                '--mem=32G',
                '--gres=gpu:1',
                f'--partition={partition}',
                f'--export=ALL,ARCH={model_name},TARGET_LAYER={target_layer},CHANNEL={channel}',
                'vital_vis_reconstruction.sh'
            ]
            
            print(f"\nLaunching SLURM job...")
            print(f"Command: {' '.join(sbatch_cmd)}")
            
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                job_ids.append((model_name, channel, job_id))
                print(f"✓ Job submitted: {job_id}")
            else:
                print(f"✗ Error submitting job: {result.stderr}")
                
        except Exception as e:
            print(f"ERROR processing {model_name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Submitted {len(job_ids)} jobs")
    print(f"{'='*60}")
    for model_name, channel, job_id in job_ids:
        print(f"{model_name:40s} channel {channel:4d} -> job {job_id}")
    
    return job_ids


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Select representative neurons and launch reconstruction jobs'
    )
    parser.add_argument('--target_layer', type=str, default='layer4.2',
                       help='Target layer (e.g., layer3.2, layer4.2)')
    parser.add_argument('--max_neurons', type=int, default=9999,
                       help='Maximum number of neurons to analyze for selection')
    parser.add_argument('--partition', type=str, default='debug',
                       help='SLURM partition to use')
    parser.add_argument('--time_limit', type=str, default='24:00:00',
                       help='Time limit for jobs (HH:MM:SS)')
    
    args = parser.parse_args()
    
    # Define models to process
    glist = ['0', '0pt5', '1', '2', '3', '4', '6']
    
    model_dirs = [
        f'patch_results/resnet50_BV_g{g}For60_E60/neuron_{args.target_layer}' 
        for g in glist
    ]
    
    model_names = [f'resnet50_BV_g{g}For60_E60' for g in glist]
    
    # Launch jobs
    launch_reconstruction_jobs(
        model_dirs=model_dirs,
        target_layer=args.target_layer,
        model_names=model_names,
        max_neurons=args.max_neurons,
        partition=args.partition,
        time_limit=args.time_limit
    )
