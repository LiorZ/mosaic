# 1DLI Enzyme Redesign: Implementation Summary

## Context and Goal

**Biological objective**: Redesign UDP-glucose dehydrogenase (PDB: 1DLI) so it binds glucose-1-phosphate (G1P) instead of glucose-1-UDP, converting it into a G1P dehydrogenase. The enzyme has two lobes: an NAD-binding lobe (residues 1-228) and a substrate-binding lobe (residues 229-402). Only the substrate-binding lobe is redesigned. CYS260 is catalytic and must be preserved in its exact position.

**Input files**:
- `1dli_g1p.pdb` — Template PDB with chain A (402-residue protein), chain B (NAD), chain C (G1P ligand already placed in the binding site)
- `INSTRUCTIONS.md` — Original task description

**Output file**:
- `redesign_protocol.py` — A marimo notebook implementing the full redesign pipeline

---

## What Was Done

A marimo notebook (`redesign_protocol.py`) was written that implements end-to-end enzyme redesign using the mosaic framework with Boltz2 as the structure prediction backbone. The notebook has 18 cells (setup + cells 1-17) that go from raw inputs to optimized sequences with validation.

---

## How the Mosaic Framework Maps to This Problem

This is a **partial protein redesign**, not a binder design. The trick is to abuse mosaic's binder-design machinery:

| Mosaic concept | Maps to in this problem |
|---|---|
| "binder" (first N tokens) | Full 402-residue protein (all of it) |
| "target" (remaining tokens) | NAD + G1P ligand tokens |
| `BinderTargetContact(epitope_idx=nad_idx)` | Protein-to-NAD contacts |
| `BinderTargetContact(epitope_idx=g1p_idx)` | Protein-to-G1P contacts |
| `WithinBinderContact` | Intra-protein contacts |
| `DistogramCE` | Scaffold fold adherence |
| `SetPositions` | Fix residues 1-228 + CYS260 |
| `StabilityModel` | Sequence stability (delta-G) |

**Key insight**: `set_binder_sequence` (in `src/mosaic/losses/boltz2.py:196`) replaces the first N tokens in the feature dict. With N=402, all protein tokens get replaced but ligand tokens (NAD, G1P) are unaffected since they come after position 401. `SetPositions` wraps this so the optimizer only sees 173 variable positions.

---

## Cell-by-Cell Implementation Details

### Setup (Cell 0): Imports

Imports are split into standard (jax, numpy, matplotlib, gemmi) and mosaic-specific. Key mosaic imports:
- `load_features_and_structure_writer` from `mosaic.losses.boltz2` — used directly instead of the higher-level `Boltz2.binder_features()` because the helper doesn't support ligands
- `SetPositions` from `mosaic.losses.transformations` — constrains optimization to variable positions only
- `SoftClip` from `mosaic.losses.transformations` — prevents over-optimization of individual loss terms
- `StabilityModel` from `mosaic.losses.stability` — predicts delta-G from ESMC embeddings
- `load_esmc` from `mosaic.losses.esmc` — loads the ESM-C 300M model

**Why `load_features_and_structure_writer` directly?** The `Boltz2.binder_features()` / `target_only_features()` helpers in `src/mosaic/models/boltz2.py` use `chain_yaml()` which only supports protein chains, not ligands. We need to construct YAML manually with `ligand:` entries.

### Cell 1: Sequence and Variable Positions

The 402-residue wildtype sequence was extracted from the PDB file using a Python script that parsed ATOM records for chain A and converted 3-letter codes to 1-letter.

`wildtype_with_x` is a 402-character string where:
- Positions 0-227: wildtype amino acids (fixed)
- Position 259: `C` (CYS260, fixed)
- All other positions 228-401: `X` (variable)

This gives **173 variable positions**. `SetPositions.from_sequence()` reads this string and builds the mapping.

### Cell 2: Model Loading

Three models are loaded:
1. `Boltz2()` — structure prediction (loads checkpoint from `~/.boltz/boltz2_conf.ckpt`)
2. `load_esmc("esmc_300m")` — ESM-C language model (used by StabilityModel)
3. `StabilityModel.from_pretrained(esmc)` — delta-G predictor (loads weights from `stability.eqx`)

### Cell 3: Template PDB to CIF Conversion

Boltz2 requires templates in CIF format with proper entity metadata. The conversion:
1. Read `1dli_g1p.pdb` with gemmi
2. Extract chain A only (protein, no ligands)
3. Build a new `gemmi.Structure` with proper entity metadata:
   - `EntityType.Polymer`, `PolymerType.PeptideL`
   - Subchain assignments
   - `label_seq_id` assignment
4. Write to a temporary CIF file

This follows the pattern in `src/mosaic/models/boltz2.py:66-112` (`build_template_yaml()`). The entity metadata is **critical** — without it, Boltz2's data pipeline silently ignores the template.

### Cell 4: YAML Construction and Feature Generation

The YAML is constructed manually (not via `chain_yaml()`) because we need ligand entries:

```yaml
version: 1
sequences:
  - protein:
      id: [A]
      sequence: <402 residues>
      msa: empty
  - ligand:
      id: [B]
      ccd: NAD
  - ligand:
      id: [C]
      ccd: G1P
templates:
  - cif: <path>
    chain_id: [A]
    template_id: [A]
```

**Why `msa: empty`?** We're fully controlling the sequence via PSSM optimization; MSA features would be misleading for a redesigned sequence.

**Why `G1P` as the CCD code?** This matches the residue name in the PDB file (`HETATM ... G1P C 404`). Boltz2 resolves CCD codes from its cached chemical component dictionary at `~/.boltz/ccd.pkl`.

After calling `load_features_and_structure_writer(yaml_str)`, we assert `template_mask` has non-zero entries to verify the template was loaded.

### Cell 5: Ligand Token Identification

After feature generation, `features["asym_id"]` is a 1D array where each token has an asym_id. The first 402 tokens are protein (asym_id 0). Tokens beyond 402 are ligands:
- NAD tokens: `asym_id == unique_target_asyms[0]` (first ligand)
- G1P tokens: `asym_id == unique_target_asyms[-1]` (second ligand)

The resulting `nad_idx` and `g1p_idx` lists are **0-indexed relative to the target portion** (i.e., relative to `binder_len`). This is what `BinderTargetContact.epitope_idx` expects — see `src/mosaic/losses/structure_prediction.py:236` where it indexes into `output.distogram_logits[:binder_len, binder_len:][:, self.epitope_idx]`.

**Note**: Each CCD ligand is typically 1 token in Boltz2, but the code doesn't hardcode this — it discovers the count from `asym_id`.

### Cell 6: Wildtype Distogram

A forward pass with the wildtype sequence (3 recycling steps) produces `wt_distogram_probs` (softmaxed distogram logits). This serves as the scaffold adherence target for `DistogramCE`.

**Why 3 recycling steps here but 1 during optimization?** More recycling gives a better reference distogram for the wildtype. During optimization, 1 recycling step is used for speed (each optimization step calls the model).

### Cell 7: Wildtype Visualization

Sanity check: predict and display the wildtype structure to verify the template and features are correct.

### Cell 8: Composite Loss (the core of the design)

The loss has two tiers:

**Tier 1 — Structure prediction losses** (wrapped in `boltz2.build_loss()`):
- `1.0 * PLDDTLoss()` — overall confidence
- `1.0 * WithinBinderContact()` — intra-protein contacts (fold compactness)
- `0.5 * WithinBinderPAE()` — intra-protein structural accuracy
- `1.0 * BinderTargetContact(epitope_idx=nad_idx, paratope_idx=range(0,228), contact_distance=12.0)` — NAD lobe must maintain NAD contacts (regularizer for the fixed region)
- `2.0 * BinderTargetContact(epitope_idx=g1p_idx, paratope_idx=range(228,402), contact_distance=12.0)` — **G1P lobe must bind G1P** (the primary design objective, hence weight 2.0)
- `0.1 * BinderTargetPAE()` + `0.1 * TargetBinderPAE()` — light structural accuracy for protein-ligand interface
- `2.0 * SoftClip(DistogramCE(wt_distogram_probs), 2.5, 3.0)` — strong scaffold adherence to preserve overall fold

**Tier 2 — Sequence-level loss** (no structure prediction):
- `0.5 * SoftClip(StabilityModel, 2.5, 3.0)` — encourages stable sequences

**Why separate tiers?** `StabilityModel.__call__` takes a `[N, 20]` PSSM directly and runs it through ESM-C. It doesn't need `Boltz2Output`. It's added outside `boltz_loss` using `+` so `SetPositions` first expands the 173-dim variable PSSM to 402-dim, then passes it to both `boltz_loss` and `stability_term`.

**Call chain**: `optimizer(173-dim)` -> `SetPositions.__call__` -> `SetPositions.sequence()` expands to 402-dim -> both `Boltz2Loss.__call__` (which calls `set_binder_sequence` then inner losses) and `StabilityModel.__call__` receive the full 402-dim PSSM.

**Weight rationale**:
- G1P contact at 2.0: this is the primary objective
- NAD contact at 1.0: regularizer to maintain existing function
- DistogramCE at 2.0 with SoftClip: strong fold preservation, but SoftClip prevents it from dominating once "good enough"
- StabilityModel at 0.5 with SoftClip: gentle stability pressure without overwhelming the structural objectives
- PAE at 0.1: light constraints, PAE losses can be noisy

### Cell 9: Phase 1 — Continuous Optimization

`simplex_APGM` with:
- `x`: Gumbel-initialized soft sequence over 173 positions x 20 amino acids
- `n_steps=100`, `stepsize=0.1`, `momentum=0.0`
- `serial_evaluation=True` (evaluates loss and gradient sequentially to save memory)

**Why Gumbel init?** `jax.nn.softmax(0.5 * jax.random.gumbel(...))` gives a soft categorical distribution that's more "peaky" than uniform but still exploratory. The 0.5 temperature controls initial sharpness.

**Why momentum=0.0?** For this redesign problem, the loss landscape is complex enough that momentum can cause oscillation. Zero momentum is more stable.

### Cell 10: Phase 2 — Sharpening

Two rounds of APGM with increasing `scale` (1.1 then 1.5), 25 steps each. The `scale` parameter in `simplex_APGM` applies weight decay that pushes the PSSM toward one-hot (discrete) sequences. This converts the soft PSSM from Phase 1 into a near-discrete sequence.

### Cell 11: Sequence Recovery

`design_loss.sequence(PSSM_sharp)` maps the 173-dim optimized PSSM back to a full 402-dim PSSM by inserting the fixed wildtype residues. `argmax` gives the discrete sequence.

### Cell 12-13: Prediction and Diagnostics

Predict the designed structure with 3 recycling steps and display:
- 3D structure viewer
- PAE heatmap (should show low PAE within protein and between protein-ligand interfaces)
- pLDDT per-residue plot with redesign boundary marked at position 228
- PSSM visualization (should be near one-hot after sharpening)

### Cell 14: MCMC Refinement

`gradient_MCMC` performs discrete sequence optimization starting from the sharpened result:
- `temp=0.001` — low temperature for greedy search
- `proposal_temp=0.00001` — very focused proposals
- `steps=50` — discrete polishing steps

This can fix edge cases where the continuous optimization settled on a suboptimal amino acid choice.

### Cell 15-16: MCMC Prediction and Export

Predict MCMC-refined structure, display, and provide download buttons for both designed and MCMC-refined PDB files.

### Cell 17: Validation

Asserts:
1. NAD lobe (positions 0-227) is identical to wildtype
2. CYS260 (position 259) is preserved as cysteine
3. Counts mutations in the redesigned region (should be up to 173)

---

## Key Decisions and Their Rationale

### 1. Why treat the full protein as "binder"?

Mosaic's loss terms (`BinderTargetContact`, `PLDDTLoss`, etc.) split the token sequence into "binder" (first N) and "target" (rest) based on `sequence.shape[0]`. By making N=402, all protein tokens are "binder" and ligand tokens are "target". This lets us use `BinderTargetContact` with `epitope_idx` and `paratope_idx` to independently control which protein residues should contact which ligand.

### 2. Why `SetPositions` instead of `FixedPositionsPenalty`?

`SetPositions` (in `src/mosaic/losses/transformations.py:87`) is a hard constraint — the optimizer literally cannot modify fixed positions. `FixedPositionsPenalty` is a soft penalty that adds L2 loss for deviations. Hard constraints are stronger and don't waste optimization capacity on maintaining fixed regions.

### 3. Why `SoftClip` on DistogramCE and StabilityModel?

`SoftClip` (in `src/mosaic/losses/transformations.py:33`) applies ELU clipping: once the loss drops below a threshold, the gradient flattens. This prevents over-optimization of any single term. Without it, the optimizer might produce unrealistic sequences that score well on one metric but poorly on others.

### 4. Why manual YAML instead of `binder_features()`?

`binder_features()` (in `src/mosaic/models/boltz2.py:114`) calls `target_only_features()` which uses `chain_yaml()` — and `chain_yaml()` only handles `protein`, not `ligand` entries. The YAML for ligands must be hand-constructed.

### 5. Why `contact_distance=12.0` for ligand contacts?

Smaller than the default 20.0 in `BinderTargetContact`. Ligand binding sites are compact — 12A captures direct and near-direct contacts without including distant residues that would dilute the signal.

### 6. Why `msa: empty`?

For a redesigned protein, MSA data would come from the wildtype homologs, which could bias the structure prediction toward the wildtype fold. Since we're redesigning the substrate-binding lobe, we want the structure prediction to be driven purely by the designed sequence.

---

## Files Referenced in the Codebase

| File | What was used from it |
|---|---|
| `src/mosaic/losses/boltz2.py` | `load_features_and_structure_writer`, `set_binder_sequence`, `Boltz2Loss`, `Boltz2Output` |
| `src/mosaic/losses/transformations.py` | `SetPositions`, `SoftClip` |
| `src/mosaic/losses/structure_prediction.py` | `PLDDTLoss`, `WithinBinderContact`, `WithinBinderPAE`, `BinderTargetContact`, `BinderTargetPAE`, `TargetBinderPAE`, `DistogramCE` |
| `src/mosaic/losses/stability.py` | `StabilityModel` |
| `src/mosaic/losses/esmc.py` | `load_esmc` |
| `src/mosaic/models/boltz2.py` | `Boltz2` class, `build_template_yaml` (pattern for CIF conversion) |
| `src/mosaic/common.py` | `TOKENS` (the 20 standard amino acid alphabet) |
| `src/mosaic/optimizers.py` | `simplex_APGM` (line 259), `gradient_MCMC` (line 125) |
| `src/mosaic/notebook_utils.py` | `pdb_viewer` |
| `src/mosaic/structure_prediction.py` | `TargetChain`, `StructurePrediction` |
| `examples/example_notebook.py` | Pattern for marimo notebook structure, optimization workflow |

---

## Potential Issues and Things to Watch For

1. **Template mask verification**: Cell 4 asserts `template_mask` is non-zero. If this fails, the CIF entity metadata is likely wrong. Check that `assign_label_seq_id()` was called.

2. **Ligand token count**: Cell 5 discovers NAD/G1P token indices dynamically. If Boltz2's CCD processing changes, the token count might differ. The code handles this by inspecting `asym_id` rather than hardcoding.

3. **Memory**: A 402-residue protein with ligands and structure prediction is large. `serial_evaluation=True` in the optimizer helps, but this still needs a GPU with substantial memory (likely 40GB+).

4. **`__file__` in marimo**: Cell 3 uses `Path(__file__).parent / "1dli_g1p.pdb"`. This works when running the notebook from the `1dli_redesign/` directory. If run from elsewhere, the path resolution may fail.

5. **`NamedTemporaryFile` lifecycle**: The `template_cif_file` and the `TemporaryDirectory` inside `load_features_and_structure_writer` must not be garbage collected during the session. The marimo cell return values keep them alive, but if cells are re-run in a different order, references could be lost.

6. **MCMC with SetPositions**: `gradient_MCMC` expects a `[N]` integer sequence. When wrapped in `SetPositions`, the input dimension is 173 (variable positions only). The MCMC output is also 173-dim and must be expanded via `design_loss.sequence(jax.nn.one_hot(seq_mcmc, 20))` before prediction.

---

## What a Future Agent Should Do

1. **Run the notebook** on a GPU machine to verify it works end-to-end
2. **Check wildtype pLDDT** (Cell 6/7) — should be >0.7 for the protein region
3. **Evaluate designed output** — look for:
   - pLDDT > 0.6 in the redesigned region (228-401)
   - Low PAE between G1P lobe and G1P ligand
   - NAD contacts maintained
   - Overall fold preserved (visual comparison to template)
4. **Tune hyperparameters** if results are poor:
   - Increase G1P contact weight (currently 2.0)
   - Adjust `contact_distance` (currently 12.0)
   - Try more optimization steps
   - Try different random seeds (the Gumbel init uses `np.random.randint`)
5. **Consider adding** inverse folding loss (ProteinMPNN on the template structure) for additional scaffold adherence
6. **Consider adding** a NoCysteine penalty if unwanted cysteines appear in the redesigned region (currently only CYS260 is fixed, but the optimizer could introduce others)
