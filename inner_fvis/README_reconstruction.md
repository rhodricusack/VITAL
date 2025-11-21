# Representative Neuron Reconstruction

This set of scripts selects the most representative neuron for each model (based on frequency spectrum similarity to the model average) and launches SLURM jobs to reconstruct those neurons.

## Files

1. **`launch_representative_reconstruction.py`** - Main launcher script
2. **`vital_vis_reconstruction.sh`** - SLURM batch script for reconstruction

## Usage

### Basic usage:
```bash
python launch_representative_reconstruction.py
```

### With options:
```bash
python launch_representative_reconstruction.py \
    --target_layer layer4.2 \
    --max_neurons 50 \
    --partition gpu \
    --time_limit 24:00:00
```

## Parameters

- `--target_layer`: Target layer to analyze (default: `layer4.2`)
  - Options: `layer3.2`, `layer4.2`, etc.
  
- `--max_neurons`: Maximum neurons to analyze when selecting representative (default: 50)
  - Lower = faster selection, but may miss some neurons
  
- `--partition`: SLURM partition to use (default: `debug`)
  - Adjust based on your cluster setup
  
- `--time_limit`: Time limit per job in HH:MM:SS format (default: `24:00:00`)

## What it does

1. For each model (g=0, 0.5, 1, 2, 3, 4, 6):
   - Analyzes frequency spectra of neuron patches
   - Compares each neuron to the model average
   - Selects the neuron most similar to average (highest correlation)

2. Launches a SLURM job for each selected neuron:
   - Runs `reverse_engineer_neurons.py`
   - Reconstructs the optimal stimulus for that channel
   - Outputs to `vital_vis/recon_{model}_{neuron}_{jobid}.out`

## Example output

```
Processing resnet50_BV_g0For60_E60
Most representative neuron: 42
✓ Job submitted: 12345

Processing resnet50_BV_g0pt5For60_E60
Most representative neuron: 38
✓ Job submitted: 12346

...

SUMMARY: Submitted 7 jobs
resnet50_BV_g0For60_E60          channel   42 -> job 12345
resnet50_BV_g0pt5For60_E60       channel   38 -> job 12346
...
```

## Checking job status

```bash
squeue -u $USER
```

## Viewing output

```bash
tail -f vital_vis/recon_resnet50_BV_g0For60_E60_n42_*.out
```
