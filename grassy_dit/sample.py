"""
Sampling script for GRASSY-DiT.
Generates molecules conditioned on scattering moments.
"""
import argparse
import numpy as np
import torch
from grassy_dit.train import ScatteringGraphDIT


def main():
    parser = argparse.ArgumentParser(description='Generate molecules with GRASSY-DiT')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--scattering', required=True, help='Path to target scattering .npy file')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per scattering')
    parser.add_argument('--num_nodes', type=int, default=None, help='Number of atoms (None = sample from training dist)')
    parser.add_argument('--output', default='generated.txt', help='Output file')
    args = parser.parse_args()
    
    # Load model
    model = ScatteringGraphDIT()
    model.load_from_local(args.checkpoint)
    
    # Load scattering
    scattering = np.load(args.scattering)
    if scattering.ndim == 2:
        scattering = scattering[0]  # Take first if multiple
    
    # Generate
    smiles_list = model.generate(
        scattering=scattering,
        num_nodes=args.num_nodes,
        batch_size=args.num_samples,
    )
    
    # Save results
    valid_smiles = [s for s in smiles_list if s is not None]
    print(f"Generated {len(valid_smiles)}/{len(smiles_list)} valid molecules")
    
    with open(args.output, 'w') as f:
        for smi in valid_smiles:
            f.write(smi + '\n')
    
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
    
# Example:
# python -m grassy_dit.sample --checkpoint grassy_dit_checkpoint.pt --scattering target.npy --num_samples 10