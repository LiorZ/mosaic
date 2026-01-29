# Sequence Design Method for 1DLI Enzyme Redesign

This document describes how the enzyme sequence is designed in `redesign_protocol.py`.

## Overview

The goal is to redesign residues 229-402 of UDP-glucose dehydrogenase (PDB: 1DLI) to bind glucose-1-phosphate (G1P) instead of glucose-1-UDP, while preserving:
- The NAD-binding lobe (residues 1-228)
- The catalytic cysteine (CYS260)
- The overall protein fold

The design uses **gradient-based optimization** of a soft sequence representation, guided by a structure prediction model (Boltz2) and multiple loss terms.

---

## Design Pipeline

### Step 1: Feature Preparation

1. **Load template structure**: The PDB file `1dli_g1p.pdb` contains the protein with G1P already positioned in the binding site (replacing UDP)
2. **Generate Boltz2 features**: A YAML specification defines the protein sequence, NAD ligand, G1P ligand, and template structure
3. **Compute wildtype distogram**: Run Boltz2 on the wildtype sequence to get the predicted inter-residue distance distribution, used as a scaffold preservation target

### Step 2: Define Variable Positions

```
Position:     1    ...    228    229    ...    259    260    ...    402
             |---- NAD lobe ----|-------- G1P lobe --------|
Constraint:  [     FIXED       ][ VAR ]...[ VAR ][ FIX ]...[ VAR ]
```

- **Fixed positions (229 total)**: Residues 1-228 (NAD lobe) + CYS260 (catalytic)
- **Variable positions (173 total)**: Residues 229-259 and 261-402

The `SetPositions` wrapper ensures the optimizer only modifies the 173 variable positions.

### Step 3: Continuous Optimization (Phase 1)

**Method**: Accelerated Projected Gradient Method (APGM) on the probability simplex

**Input**: A soft sequence representation — a `(173, 20)` matrix where each row is a probability distribution over 20 amino acids

**Initialization**: Gumbel-softmax random initialization
```python
x = softmax(0.5 * gumbel_noise)
```

**Optimization**:
```python
simplex_APGM(
    loss_function=design_loss,
    x=initial_pssm,
    n_steps=100,
    stepsize=0.1,
    momentum=0.0,
)
```

At each step:
1. Compute loss and gradients via Boltz2 forward/backward pass
2. Take a gradient step
3. Project back onto the probability simplex (ensure each row sums to 1, all values ≥ 0)

**Output**: A "soft" sequence where each position has a probability distribution over amino acids

### Step 4: Sharpening (Phase 2)

The soft sequence is progressively sharpened toward a discrete one-hot encoding:

```python
# Round 1: mild sharpening
simplex_APGM(..., scale=1.1, n_steps=25)

# Round 2: strong sharpening
simplex_APGM(..., scale=1.5, n_steps=25)
```

The `scale` parameter applies entropy regularization that pushes probabilities toward 0 or 1.

**Output**: A nearly-discrete PSSM where `argmax` gives the designed sequence

### Step 5: MCMC Refinement (Phase 3, Optional)

Gradient-guided Markov Chain Monte Carlo for discrete sequence polishing:

```python
gradient_MCMC(
    design_loss,
    discrete_sequence,
    temp=0.001,        # Low temperature for greedy search
    proposal_temp=0.00001,
    steps=50,
)
```

At each step:
1. Compute gradients to identify promising mutations
2. Propose a single-residue mutation based on gradient information
3. Accept/reject based on Metropolis criterion

**Output**: A refined discrete sequence

---

## Loss Function

The design is guided by minimizing a composite loss function:

### Structure Prediction Losses (via Boltz2)

| Term | Weight | Description |
|------|--------|-------------|
| `PLDDTLoss` | 1.0 | Maximize predicted Local Distance Difference Test score (confidence) |
| `WithinBinderContact` | 1.0 | Encourage intra-protein contacts (fold compactness) |
| `WithinBinderPAE` | 0.5 | Minimize predicted aligned error within protein |
| `BinderTargetContact` (NAD) | 1.0 | Maintain contacts between NAD lobe (res 1-228) and NAD ligand |
| `BinderTargetContact` (G1P) | 2.0 | Encourage contacts between G1P lobe (res 229-402) and G1P ligand |
| `BinderTargetPAE` | 0.1 | Minimize predicted error at protein→ligand interface |
| `TargetBinderPAE` | 0.1 | Minimize predicted error at ligand→protein interface |
| `DistogramCE` | 2.0 | Match wildtype inter-residue distance distribution (scaffold preservation) |

### Sequence-Level Losses

| Term | Weight | Description |
|------|--------|-------------|
| `StabilityModel` | 0.5 | Minimize predicted ΔG (encourage stable sequences) |

### Loss Clipping

`SoftClip` is applied to `DistogramCE` and `StabilityModel` to prevent over-optimization:
```python
SoftClip(loss, lower=2.5, upper=3.0)
```
Once the loss drops below the threshold, gradients are attenuated. This prevents the optimizer from producing unrealistic sequences that score well on one metric but poorly on others.

---

## Key Design Choices

### Why gradient-based optimization?

- **Efficiency**: Gradients provide direction for improvement, much faster than random search
- **Differentiable structure prediction**: Boltz2 (like AlphaFold) is fully differentiable, enabling end-to-end gradient flow from structure to sequence

### Why soft sequences?

- Discrete sequences are not differentiable (can't compute gradients through argmax)
- Soft sequences (probability distributions) enable gradient-based optimization
- Sharpening progressively converts soft → discrete

### Why multiple loss terms?

Each term addresses a different aspect of the design:
- **Confidence losses** (pLDDT, PAE): Ensure the structure prediction is reliable
- **Contact losses**: Ensure proper protein-ligand interactions
- **Scaffold loss** (DistogramCE): Preserve the overall fold
- **Stability loss**: Ensure the designed sequence is physically realizable

### Why fix the NAD lobe?

- The NAD lobe (residues 1-228) is responsible for NAD binding and catalysis
- It should remain unchanged to preserve enzymatic function
- Only the substrate-binding lobe needs redesign for the new substrate (G1P)

### Why preserve CYS260?

- CYS260 is the catalytic residue that forms a covalent intermediate during the reaction
- Its position must be maintained exactly for catalytic activity

---

## Output

The protocol produces:
1. **Designed sequence**: 402 residues, with positions 1-228 and 260 identical to wildtype
2. **Predicted structure**: Boltz2 prediction of the designed sequence with ligands
3. **Confidence metrics**: pLDDT and iPTM scores
4. **Validation**: Verification that constraints are satisfied

---

## Computational Requirements

- **GPU memory**: ~20-40 GB recommended (462-token system with gradients)
- **Time**: ~1-2 hours for full optimization on a modern GPU
- **Dependencies**: JAX, Boltz2, mosaic framework
