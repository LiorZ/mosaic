# ProteinMPNN-Based Sequence Design for 1DLI Enzyme Redesign

This document describes the ProteinMPNN-based design protocol in `redesign_protocol_mpnn.py`.

## Overview

This is an alternative to the Boltz2-based gradient optimization protocol. Instead of optimizing a soft sequence representation against structure prediction losses, this protocol uses **ProteinMPNN inverse folding** to directly design sequences that are compatible with the fixed backbone structure.

## Key Differences from Boltz2 Protocol

| Aspect | Boltz2 Protocol | ProteinMPNN Protocol |
|--------|-----------------|---------------------|
| **Method** | Gradient-based optimization of soft sequences | Autoregressive sequence generation |
| **Structure** | Predicted by Boltz2 during optimization | Fixed template structure |
| **Backbone** | Implicitly optimized via structure prediction | Completely fixed |
| **Speed** | Slow (minutes-hours) | Fast (seconds) |
| **Memory** | High (~20-40 GB GPU) | Low (~2-4 GB GPU) |
| **Diversity** | Single optimized sequence | Multiple diverse samples |

## Designable Positions

The protocol allows more flexibility in the NAD lobe compared to the Boltz2 version:

### Fixed Positions
1. **NAD-contacting residues** (within 8Å of NAD): These directly interact with NAD and must be preserved for cofactor binding
2. **CYS260**: Catalytic residue that must be preserved

### Designable Positions
1. **Entire G1P lobe** (residues 229-402, except CYS260): The substrate-binding domain being redesigned
2. **NAD lobe residues >8Å from NAD**: These don't directly contact NAD, so their sequence can be optimized for stability/folding without affecting NAD binding

## Distance Calculation

The minimum distance from each residue's Cα atom to any NAD atom is computed:

```python
# For each residue i and NAD atom j
distance[i, j] = sqrt((CA_i - NAD_j)^2)

# Minimum distance for residue i
min_distance[i] = min(distance[i, :])

# Designable if > 8Å
designable_in_nad_lobe[i] = min_distance[i] > 8.0
```

## ProteinMPNN Design Process

### 1. Structure Encoding

The fixed backbone structure (chain A only) is encoded by ProteinMPNN's encoder:

```python
h_V, h_E, E_idx = mpnn.encode(
    X=coords,        # (N, 4, 3) backbone coordinates
    mask=mask,       # (N,) valid positions
    residue_idx=..., # (N,) residue indices
    chain_encoding=..., # (N,) chain IDs
)
```

This produces hidden representations of the structure that capture local and global geometric features.

### 2. Autoregressive Decoding with Fixed Positions

ProteinMPNN uses autoregressive decoding controlled by `decoding_order`:
- **Low values** → decoded early (designable, generated fresh)
- **High values** → decoded late (fixed, conditioned upon)

```python
decoding_order = jnp.where(
    designable_mask,
    random_values,      # Random order for designable
    10.0,               # High value for fixed (decoded last)
)
```

Fixed positions are effectively "given" to the model as context, while designable positions are generated based on that context.

### 3. Jacobi Iteration

Since ProteinMPNN is autoregressive, we use Jacobi iteration to approximate parallel decoding:

```python
for iteration in range(num_iterations):
    logits = mpnn.decode(current_sequence, ...)
    new_sequence = sample(logits / temperature + gumbel_noise)
    current_sequence = where(designable, new_sequence, wildtype)
```

This iteratively refines the sequence until convergence.

### 4. Temperature-Based Sampling

The `temperature` parameter controls diversity:
- **Low temperature (0.1)**: More deterministic, higher confidence
- **High temperature (1.0)**: More diverse, exploratory

Multiple sequences are generated with different random seeds to provide design diversity.

## Output

The protocol produces:
1. **Multiple designed sequences** (default: 8) in FASTA format
2. **Mutation analysis**: Which positions changed and whether constraints are satisfied
3. **Visualization**: Distance plots and mutation maps

## Computational Requirements

- **GPU memory**: ~2-4 GB (much less than Boltz2)
- **Time**: ~10-30 seconds for 8 designs
- **Dependencies**: JAX, ProteinMPNN weights (included in mosaic)

## When to Use This Protocol

**Use ProteinMPNN when:**
- You want fast sequence generation
- The backbone structure is fixed and trusted
- You need multiple diverse sequence candidates
- GPU memory is limited

**Use Boltz2 when:**
- You want to co-optimize sequence and structure
- The binding geometry might need adjustment
- You have sufficient GPU memory
- You need fine-grained control via multiple loss terms

## Limitations

1. **Fixed backbone assumption**: ProteinMPNN assumes the backbone is correct. If the G1P binding requires backbone changes, this won't be captured.

2. **No explicit ligand modeling**: Unlike the Boltz2 protocol, ProteinMPNN doesn't see the G1P ligand. The design relies on the backbone geometry implicitly encoding the binding site shape.

3. **No stability optimization**: ProteinMPNN optimizes for backbone compatibility, not thermodynamic stability. Consider filtering designs with a stability predictor.

## Recommended Workflow

1. Run ProteinMPNN protocol to generate diverse candidates (fast)
2. Filter by sequence properties (no cysteines, charge, etc.)
3. Validate top candidates with Boltz2 structure prediction
4. Optionally refine best candidates with the Boltz2 optimization protocol
