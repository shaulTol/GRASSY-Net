"""
Scaffold-constrained generation using torch-molecule's discrete diffusion.

Hooks into the trained ScatteringGraphDIT model and extends sampling
with optional scaffold constraints.
"""
import torch
import numpy as np
from typing import List, Optional, Union
from rdkit import Chem

from grassy_dit.train import ScatteringGraphDIT


def smiles_to_scaffold_tensors(smiles: str, max_n_nodes: int, atom_types: List[str], bond_types: List[str]):
    """
    Convert scaffold SMILES to one-hot tensors.
    
    Args:
        smiles: Scaffold SMILES string
        max_n_nodes: Maximum nodes (for padding)
        atom_types: List of atom symbols in order (e.g., ['C', 'N', 'O', ...])
        bond_types: List of bond types (e.g., ['NONE', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
    
    Returns:
        X: [max_n_nodes, num_atom_types] one-hot atom features
        E: [max_n_nodes, max_n_nodes, num_bond_types] one-hot bond features
        atom_indices: List of atom indices in the scaffold
        num_atoms: Number of atoms in scaffold
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    num_atoms = mol.GetNumAtoms()
    num_atom_types = len(atom_types)
    num_bond_types = len(bond_types)
    
    # Initialize tensors
    X = torch.zeros(max_n_nodes, num_atom_types)
    E = torch.zeros(max_n_nodes, max_n_nodes, num_bond_types)
    
    # Fill atom features
    atom_to_idx = {a: i for i, a in enumerate(atom_types)}
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol in atom_to_idx:
            X[i, atom_to_idx[symbol]] = 1.0
    
    # Fill bond features (index 0 = no bond typically)
    bond_type_map = {
        Chem.BondType.SINGLE: bond_types.index('SINGLE') if 'SINGLE' in bond_types else 1,
        Chem.BondType.DOUBLE: bond_types.index('DOUBLE') if 'DOUBLE' in bond_types else 2,
        Chem.BondType.TRIPLE: bond_types.index('TRIPLE') if 'TRIPLE' in bond_types else 3,
        Chem.BondType.AROMATIC: bond_types.index('AROMATIC') if 'AROMATIC' in bond_types else 4,
    }
    
    # Set no-bond for all pairs first
    E[:, :, 0] = 1.0
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        if bt in bond_type_map:
            bond_idx = bond_type_map[bt]
            E[i, j, :] = 0
            E[j, i, :] = 0
            E[i, j, bond_idx] = 1.0
            E[j, i, bond_idx] = 1.0
    
    atom_indices = list(range(num_atoms))
    return X, E, atom_indices, num_atoms


def create_scaffold_mask(scaffold_atoms: List[int], max_n_nodes: int, device=None):
    """
    Create boolean masks for scaffold atoms and edges.
    
    Args:
        scaffold_atoms: List of atom indices that are part of scaffold
        max_n_nodes: Maximum number of nodes
        device: Torch device
    
    Returns:
        dict with 'nodes' [max_n_nodes] and 'edges' [max_n_nodes, max_n_nodes] boolean masks
    """
    node_mask = torch.zeros(max_n_nodes, dtype=torch.bool, device=device)
    node_mask[scaffold_atoms] = True
    
    # Edge mask: edges between scaffold atoms
    edge_mask = node_mask[:, None] & node_mask[None, :]
    
    return {'nodes': node_mask, 'edges': edge_mask}


class ScaffoldScatteringSampler:
    """
    Sampler that uses torch-molecule's discrete diffusion with scaffold constraints.
    
    Hooks into the trained ScatteringGraphDIT model and modifies sampling
    to keep scaffold atoms/bonds fixed throughout denoising.
    
    Usage:
        # Load trained model
        model = ScatteringGraphDIT.load_from_local('grassy_dit_checkpoint.pt')
        
        # Create sampler
        sampler = ScaffoldScatteringSampler(model)
        
        # Sample unconditionally (just scattering conditioning)
        molecules = sampler.sample(scattering=target_scatter, num_samples=10)
        
        # Sample with scaffold constraint
        molecules = sampler.sample(
            scattering=target_scatter,
            num_samples=10,
            scaffold_smiles='c1ccccc1',  # benzene scaffold
            total_atoms=20,
        )
    """
    
    def __init__(self, model: ScatteringGraphDIT):
        """
        Args:
            model: Trained ScatteringGraphDIT model
        """
        self.model = model
        self.device = model.device
        
        # Get model config
        self.max_n_nodes = model.max_node
        self.num_atom_types = model.input_dim_X
        self.num_bond_types = model.input_dim_E
        
        # Get diffusion config from model
        self.num_steps = model.diffusion_steps
        self.noise_schedule = model.noise_schedule
        
    def _get_atom_bond_types(self):
        """Get atom and bond type lists from model config."""
        # Default types - adjust based on your dataset
        atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H'][:self.num_atom_types]
        bond_types = ['NONE', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'][:self.num_bond_types]
        return atom_types, bond_types
    
    @torch.no_grad()
    def sample(
        self,
        scattering: Union[torch.Tensor, np.ndarray],
        num_samples: int = 1,
        guidance_scale: float = 2.0,
        scaffold_smiles: Optional[str] = None,
        scaffold_atoms: Optional[List[int]] = None,
        scaffold_X: Optional[torch.Tensor] = None,
        scaffold_E: Optional[torch.Tensor] = None,
        total_atoms: Optional[int] = None,
        return_smiles: bool = True,
    ):
        """
        Generate molecules conditioned on scattering moments with optional scaffold.
        
        Args:
            scattering: [440] or [num_samples, 440] scattering moments
            num_samples: Number of molecules to generate
            guidance_scale: CFG guidance strength (1.0 = no guidance)
            scaffold_smiles: SMILES of scaffold (alternative to scaffold_X/E)
            scaffold_atoms: List of atom indices for scaffold (if using scaffold_X/E)
            scaffold_X: [max_n_nodes, Xdim] scaffold atom features
            scaffold_E: [max_n_nodes, max_n_nodes, Edim] scaffold bond features
            total_atoms: Total atoms in final molecule (scaffold + generated)
            return_smiles: If True, return SMILES strings; else return tensors
        
        Returns:
            List of SMILES strings or (X, E) tensor tuple
        """
        self.model.model.eval()
        
        # Prepare scattering
        if isinstance(scattering, np.ndarray):
            scattering = torch.from_numpy(scattering).float()
        if scattering.dim() == 1:
            scattering = scattering.unsqueeze(0).expand(num_samples, -1)
        scattering = scattering.to(self.device)
        
        # Process scaffold
        use_scaffold = scaffold_smiles is not None or scaffold_X is not None
        scaffold_mask = None
        
        if scaffold_smiles is not None:
            atom_types, bond_types = self._get_atom_bond_types()
            scaffold_X, scaffold_E, scaffold_atoms, scaffold_size = smiles_to_scaffold_tensors(
                scaffold_smiles, self.max_n_nodes, atom_types, bond_types
            )
            if total_atoms is None:
                total_atoms = scaffold_size  # Just scaffold if not specified
        
        if use_scaffold:
            scaffold_X = scaffold_X.unsqueeze(0).expand(num_samples, -1, -1).to(self.device)
            scaffold_E = scaffold_E.unsqueeze(0).expand(num_samples, -1, -1, -1).to(self.device)
            scaffold_mask = create_scaffold_mask(scaffold_atoms, self.max_n_nodes, self.device)
            scaffold_mask = {k: v.unsqueeze(0).expand(num_samples, *v.shape) for k, v in scaffold_mask.items()}
        
        # Determine number of atoms
        if total_atoms is None:
            # Sample from training distribution or use default
            total_atoms = min(self.max_n_nodes, 20)  # Default
        
        # Create node mask
        node_mask = torch.zeros(num_samples, self.max_n_nodes, dtype=torch.bool, device=self.device)
        node_mask[:, :total_atoms] = True
        
        # Initialize from prior distribution (categorical, not Gaussian)
        X_t, E_t = self._sample_prior(num_samples, node_mask)
        
        # Inject scaffold into initial state
        if use_scaffold:
            X_t = self._inject_scaffold(X_t, scaffold_X, scaffold_mask['nodes'])
            E_t = self._inject_scaffold_edges(E_t, scaffold_E, scaffold_mask['edges'])
        
        # Reverse diffusion
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            # Predict x_0 with CFG
            X_pred, E_pred = self._predict_with_cfg(
                X_t, E_t, node_mask, t_tensor, scattering, guidance_scale
            )
            
            # Sample x_{t-1} from posterior (discrete diffusion)
            if t > 0:
                X_t, E_t = self._sample_posterior(X_t, E_t, X_pred, E_pred, t, node_mask)
            else:
                # Final step: use argmax
                X_t = torch.zeros_like(X_pred).scatter_(-1, X_pred.argmax(-1, keepdim=True), 1.0)
                E_t = torch.zeros_like(E_pred).scatter_(-1, E_pred.argmax(-1, keepdim=True), 1.0)
            
            # Re-inject scaffold (keep it clean)
            if use_scaffold:
                X_t = self._inject_scaffold(X_t, scaffold_X, scaffold_mask['nodes'])
                E_t = self._inject_scaffold_edges(E_t, scaffold_E, scaffold_mask['edges'])
        
        # Convert to molecules
        if return_smiles:
            return self._tensors_to_smiles(X_t, E_t, node_mask)
        else:
            return X_t, E_t
    
    def _sample_prior(self, num_samples: int, node_mask: torch.Tensor):
        """Sample from prior distribution (marginal atom/bond type distributions)."""
        # Get marginals from model if available, else uniform
        if hasattr(self.model, 'marginal_X'):
            marginal_X = self.model.marginal_X.to(self.device)
            marginal_E = self.model.marginal_E.to(self.device)
        else:
            # Uniform prior
            marginal_X = torch.ones(self.num_atom_types, device=self.device) / self.num_atom_types
            marginal_E = torch.ones(self.num_bond_types, device=self.device) / self.num_bond_types
        
        # Sample atoms
        X = torch.zeros(num_samples, self.max_n_nodes, self.num_atom_types, device=self.device)
        atom_samples = torch.multinomial(marginal_X.expand(num_samples * self.max_n_nodes, -1), 1)
        atom_samples = atom_samples.view(num_samples, self.max_n_nodes)
        X.scatter_(-1, atom_samples.unsqueeze(-1), 1.0)
        
        # Sample edges
        E = torch.zeros(num_samples, self.max_n_nodes, self.max_n_nodes, self.num_bond_types, device=self.device)
        num_edges = self.max_n_nodes * self.max_n_nodes
        edge_samples = torch.multinomial(marginal_E.expand(num_samples * num_edges, -1), 1)
        edge_samples = edge_samples.view(num_samples, self.max_n_nodes, self.max_n_nodes)
        E.scatter_(-1, edge_samples.unsqueeze(-1), 1.0)
        
        # Symmetrize edges
        E = (E + E.transpose(1, 2)) / 2
        E = E / E.sum(-1, keepdim=True).clamp(min=1e-8)
        
        # Mask padding
        X = X * node_mask.unsqueeze(-1)
        mask_2d = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        E = E * mask_2d.unsqueeze(-1)
        
        return X, E
    
    def _predict_with_cfg(self, X_t, E_t, node_mask, t, scattering, guidance_scale):
        """Predict x_0 with classifier-free guidance."""
        # Prepare noisy_data dict for adapter
        noisy_data = {
            'X_t': X_t,
            'E_t': E_t,
            'node_mask': node_mask,
            't': t,
            'y_t': scattering,
        }
        
        # Conditional prediction
        pred_cond = self.model.model(noisy_data, unconditioned=False)
        X_cond, E_cond = pred_cond.X, pred_cond.E
        
        if guidance_scale == 1.0:
            return X_cond, E_cond
        
        # Unconditional prediction
        pred_uncond = self.model.model(noisy_data, unconditioned=True)
        X_uncond, E_uncond = pred_uncond.X, pred_uncond.E
        
        # CFG combination
        X_pred = X_uncond + guidance_scale * (X_cond - X_uncond)
        E_pred = E_uncond + guidance_scale * (E_cond - E_uncond)
        
        return X_pred, E_pred
    
    def _sample_posterior(self, X_t, E_t, X_pred, E_pred, t, node_mask):
        """Sample from posterior q(x_{t-1} | x_t, x_0_pred) for discrete diffusion."""
        # Get noise schedule parameters
        alpha_t = self._get_alpha(t)
        alpha_t_minus_1 = self._get_alpha(t - 1)
        
        # Compute posterior probabilities
        # p(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x_0)
        
        # Softmax predictions to get probabilities
        X_prob = torch.softmax(X_pred, dim=-1)
        E_prob = torch.softmax(E_pred, dim=-1)
        
        # Add noise for sampling (simplified - proper implementation uses transition matrices)
        noise_scale = (1 - alpha_t_minus_1) / (1 - alpha_t + 1e-8)
        noise_scale = min(noise_scale, 0.99)
        
        # Mix prediction with uniform noise
        uniform_X = torch.ones_like(X_prob) / self.num_atom_types
        uniform_E = torch.ones_like(E_prob) / self.num_bond_types
        
        X_posterior = (1 - noise_scale) * X_prob + noise_scale * uniform_X
        E_posterior = (1 - noise_scale) * E_prob + noise_scale * uniform_E
        
        # Sample categorically
        X_samples = torch.multinomial(X_posterior.view(-1, self.num_atom_types), 1)
        X_samples = X_samples.view(X_t.shape[0], self.max_n_nodes)
        X_new = torch.zeros_like(X_t).scatter_(-1, X_samples.unsqueeze(-1), 1.0)
        
        E_samples = torch.multinomial(E_posterior.view(-1, self.num_bond_types), 1)
        E_samples = E_samples.view(E_t.shape[0], self.max_n_nodes, self.max_n_nodes)
        E_new = torch.zeros_like(E_t).scatter_(-1, E_samples.unsqueeze(-1), 1.0)
        
        # Symmetrize edges
        E_new = (E_new + E_new.transpose(1, 2)) / 2
        E_new = (E_new > 0.5).float()
        E_new = E_new / E_new.sum(-1, keepdim=True).clamp(min=1e-8)
        
        # Mask padding
        X_new = X_new * node_mask.unsqueeze(-1)
        mask_2d = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        E_new = E_new * mask_2d.unsqueeze(-1)
        
        return X_new, E_new
    
    def _get_alpha(self, t):
        """Get cumulative alpha for timestep t."""
        if t < 0:
            return 1.0
        # Cosine schedule
        s = 0.008
        T = self.num_steps
        return np.cos(0.5 * np.pi * (t / T + s) / (1 + s)) ** 2
    
    def _inject_scaffold(self, X, scaffold_X, node_mask):
        """Replace scaffold positions with clean scaffold atoms."""
        X = X.clone()
        X[node_mask] = scaffold_X[node_mask]
        return X
    
    def _inject_scaffold_edges(self, E, scaffold_E, edge_mask):
        """Replace scaffold edges with clean scaffold bonds."""
        E = E.clone()
        E[edge_mask] = scaffold_E[edge_mask]
        return E
    
    def _tensors_to_smiles(self, X, E, node_mask):
        """Convert predicted tensors to SMILES strings."""
        atom_types, bond_types = self._get_atom_bond_types()
        smiles_list = []
        
        for b in range(X.shape[0]):
            try:
                mol = Chem.RWMol()
                num_atoms = node_mask[b].sum().item()
                
                # Add atoms
                atom_indices = X[b, :num_atoms].argmax(-1).cpu().numpy()
                atom_map = {}
                for i, idx in enumerate(atom_indices):
                    if idx < len(atom_types):
                        atom = Chem.Atom(atom_types[idx])
                        atom_map[i] = mol.AddAtom(atom)
                
                # Add bonds
                bond_type_map = {
                    1: Chem.BondType.SINGLE,
                    2: Chem.BondType.DOUBLE,
                    3: Chem.BondType.TRIPLE,
                    4: Chem.BondType.AROMATIC,
                }
                
                edge_indices = E[b, :num_atoms, :num_atoms].argmax(-1).cpu().numpy()
                for i in range(num_atoms):
                    for j in range(i + 1, num_atoms):
                        bond_idx = edge_indices[i, j]
                        if bond_idx > 0 and bond_idx in bond_type_map:
                            if i in atom_map and j in atom_map:
                                mol.AddBond(atom_map[i], atom_map[j], bond_type_map[bond_idx])
                
                # Sanitize and get SMILES
                try:
                    Chem.SanitizeMol(mol)
                    smiles = Chem.MolToSmiles(mol)
                except:
                    smiles = None
                    
                smiles_list.append(smiles)
                
            except Exception as e:
                smiles_list.append(None)
        
        return smiles_list


# Convenience function for quick sampling
def sample_with_scaffold(
    checkpoint_path: str,
    scattering: Union[torch.Tensor, np.ndarray],
    scaffold_smiles: Optional[str] = None,
    total_atoms: int = 20,
    num_samples: int = 1,
    guidance_scale: float = 2.0,
):
    """
    Quick sampling function.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        scattering: [440] target scattering moments
        scaffold_smiles: Optional scaffold SMILES
        total_atoms: Total atoms in generated molecule
        num_samples: Number of samples to generate
        guidance_scale: CFG guidance strength
    
    Returns:
        List of SMILES strings
    """
    model = ScatteringGraphDIT.load_from_local(checkpoint_path)
    sampler = ScaffoldScatteringSampler(model)
    
    return sampler.sample(
        scattering=scattering,
        num_samples=num_samples,
        guidance_scale=guidance_scale,
        scaffold_smiles=scaffold_smiles,
        total_atoms=total_atoms,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--scattering', required=True, help='Path to target scattering .npy file')
    parser.add_argument('--scaffold', default=None, help='Scaffold SMILES (optional)')
    parser.add_argument('--total_atoms', type=int, default=20, help='Total atoms in molecule')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--guidance_scale', type=float, default=2.0, help='CFG guidance scale')
    parser.add_argument('--output', default='generated.txt', help='Output file')
    args = parser.parse_args()
    
    # Load scattering
    scattering = np.load(args.scattering)
    if scattering.ndim == 2:
        scattering = scattering[0]  # Take first if multiple
    
    # Generate
    smiles_list = sample_with_scaffold(
        checkpoint_path=args.checkpoint,
        scattering=scattering,
        scaffold_smiles=args.scaffold,
        total_atoms=args.total_atoms,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
    )
    
    # Save results
    valid_smiles = [s for s in smiles_list if s is not None]
    print(f"Generated {len(valid_smiles)}/{len(smiles_list)} valid molecules")
    
    with open(args.output, 'w') as f:
        for smi in valid_smiles:
            f.write(smi + '\n')
    
    print(f"Saved to {args.output}")
    
    # Example usage:
    # python -m grassy_dit.sample --checkpoint grassy_dit_checkpoint.pt --scattering target.npy --scaffold 'c1ccccc1' --total_atoms 25 --num_samples 10