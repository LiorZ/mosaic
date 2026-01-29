# Hybrid Boltz2 + ProteinMPNN Design Protocol

This document describes the hybrid design protocol in `redesign_protocol_hybrid.py`.

## Overview

This protocol combines **Boltz2 structure prediction** with **ProteinMPNN inverse folding** in a single optimization loop, then uses ProteinMPNN for diverse sequence sampling. It offers the best of both approaches:

1. **Phase 1**: Joint backbone + sequence optimization using Boltz2 with ProteinMPNNLoss
2. **Phase 2**: Diverse sequence sampling using ProteinMPNN on the optimized backbone
3. **Phase 3**: Validation of all sequences with Boltz2

## Key Design Decisions

### Two Separate Designable Masks

To preserve the NAD lobe backbone while allowing sequence changes in residues far from NAD:

| Mask | Purpose | Positions |
|------|---------|-----------|
| `boltz_designable_mask` | Boltz2 optimization (backbone + sequence) | G1P lobe only (229-402, except CYS260) = 173 positions |
| `mpnn_designable_mask` | MPNN sampling (sequence only) | G1P lobe + NAD lobe >8Å from NAD = ~230 positions |

**Why?** Boltz2 predicts structure from sequence. If we allowed NAD lobe sequence changes during Boltz2 optimization, the predicted backbone could drift. By fixing the NAD lobe sequence in Phase 1, the backbone remains template-driven. In Phase 2, ProteinMPNN designs sequences for a **fixed backbone**, so NAD lobe residues far from NAD can be designed without backbone changes.

### ProteinMPNNLoss Integration

The key innovation is adding `ProteinMPNNLoss` to the Boltz2 optimization loop:

```python
inner_loss = (
    # ... structure losses ...
    + 1.0 * ProteinMPNNLoss(
        mpnn=mpnn,
        num_samples=4,
        stop_grad=False,  # Gradients flow through structure
    )
)
```

This creates a **self-consistency loop**:
1. Boltz2 predicts structure from sequence
2. ProteinMPNNLoss evaluates: "Does MPNN think this sequence fits this structure?"
3. Gradients from MPNN loss flow back through Boltz2's structure prediction
4. The optimizer finds sequences where **both models agree**

### Fixed Position Bias for MPNN Sampling

In Phase 2, we ensure fixed positions aren't mutated by MPNN using a bias term:

```python
fixed_position_bias = jnp.zeros((402, 20))
for i in range(402):
    if not mpnn_designable_mask[i]:
        wt_idx = TOKENS.index(PROTEIN_SEQUENCE[i])
        bias = jnp.full(20, -1e6)  # -inf for non-wildtype
        bias = bias.at[wt_idx].set(0.0)  # 0 for wildtype
        fixed_position_bias = fixed_position_bias.at[i].set(bias)
```

The bias is added to MPNN logits, making non-wildtype amino acids impossible at fixed positions.

## Phase 1: Joint Optimization

### Loss Function

| Term | Weight | Description |
|------|--------|-------------|
| PLDDTLoss | 1.0 | Maximize structure prediction confidence |
| WithinBinderContact | 1.0 | Intra-protein contacts (fold compactness) |
| WithinBinderPAE | 0.5 | Intra-protein structural accuracy |
| BinderTargetContact (NAD) | 1.0 | NAD lobe ↔ NAD contacts |
| BinderTargetContact (G1P) | 2.0 | G1P lobe ↔ G1P contacts (primary objective) |
| BinderTargetPAE | 0.1 | Protein → ligand accuracy |
| TargetBinderPAE | 0.1 | Ligand → protein accuracy |
| DistogramCE | 2.0 | Scaffold preservation (SoftClipped) |
| **ProteinMPNNLoss** | 1.0 | **Inverse folding consistency** |

### Optimization

1. **Initialization**: Gumbel-softmax random PSSM over 173 variable positions
2. **Optimization**: `simplex_APGM` for 100 steps with `serial_evaluation=True`
3. **Sharpening**: Two rounds with `scale=1.1` then `scale=1.5` to convert soft → discrete

## Phase 2: Diverse Sequence Sampling

After Phase 1 converges:

1. Get Boltz2 output (structure) for the optimized sequence
2. Use `jacobi_inverse_fold()` with the MPNN designable mask
3. Generate 8 diverse sequences with different random seeds
4. Apply fixed position bias to preserve constraints

```python
seq_tokens = jacobi_inverse_fold(
    mpnn=mpnn,
    binder_length=402,
    output=phase1_output,
    temp=0.1,  # Low temperature for confident designs
    key=key_i,
    jacobi_iterations=20,
    bias=fixed_position_bias,
)
```

## Phase 3: Validation

Each diverse sequence is validated with Boltz2:
- pLDDT (confidence)
- iPTM (interface quality)
- Mutation counts (G1P lobe vs NAD lobe)
- Constraint checks (CYS260, fixed positions)

## Comparison with Other Protocols

| Aspect | redesign_protocol.py | redesign_protocol_mpnn.py | **redesign_protocol_hybrid.py** |
|--------|---------------------|--------------------------|--------------------------------|
| Backbone optimization | Boltz2 (G1P lobe) | None (fixed template) | **Boltz2 (G1P lobe)** |
| Sequence design | Gradient on PSSM | ProteinMPNN | **Both: gradient + MPNN loss** |
| NAD lobe backbone | Fixed (no sequence changes) | Fixed | **Fixed** |
| NAD lobe sequence | All fixed | >8Å from NAD designable | **Phase 1: fixed; Phase 2: >8Å designable** |
| Diversity | 1 sequence | 8+ sequences | **1 optimized + 8+ diverse** |
| Self-consistency | Implicit | Explicit (MPNN) | **Explicit (both models)** |

## When to Use This Protocol

**Use the hybrid protocol when:**
- You want the structural optimization power of Boltz2
- You want sequences that ProteinMPNN agrees with (inverse folding consistency)
- You want diverse sequence candidates for experimental testing
- You need to preserve NAD lobe backbone while allowing sequence changes far from NAD

**Use Boltz2-only (redesign_protocol.py) when:**
- You only need one optimized sequence
- Memory is limited (ProteinMPNNLoss adds overhead)
- You don't need NAD lobe sequence changes

**Use MPNN-only (redesign_protocol_mpnn.py) when:**
- You trust the template backbone completely
- You need fast generation (seconds vs. hour)
- Memory is very limited

## Computational Requirements

- **GPU memory**: ~25-45 GB (Boltz2 + ProteinMPNNLoss)
- **Time**: ~1-2 hours for Phase 1, ~1 minute for Phase 2
- **Dependencies**: JAX, Boltz2, ProteinMPNN

## Output

1. **Phase 1 sequence**: Single optimized sequence with co-designed backbone
2. **Phase 2 sequences**: 8 diverse sequences, all compatible with the Phase 1 backbone
3. **Validation metrics**: pLDDT, iPTM, mutation counts for all sequences
4. **Structures**: Boltz2 predictions for all sequences
5. **FASTA export**: All sequences in a single file
