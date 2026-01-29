import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

with app.setup:
    # GPU memory settings - must be before importing JAX
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    import jax
    jax.config.update("jax_default_matmul_precision", "bfloat16")
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import gemmi
    from pathlib import Path
    from tempfile import NamedTemporaryFile

    from mosaic.optimizers import simplex_APGM
    import mosaic.losses.structure_prediction as sp
    from mosaic.models.boltz2 import Boltz2
    from mosaic.losses.boltz2 import load_features_and_structure_writer
    from mosaic.losses.transformations import SetPositions, SoftClip
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery, jacobi_inverse_fold
    from mosaic.proteinmpnn.mpnn import load_mpnn
    from mosaic.common import TOKENS
    from mosaic.notebook_utils import pdb_viewer


@app.cell
def _():
    """Cell 1: Define protein sequence and load structure for NAD distance calculation."""
    PROTEIN_SEQUENCE = (
        "MKIAVAGSGYVGLSLGVLLSLQNEVTIVDILPSKVDKINNGLSPIQDEYIEYYLKSKQLSIKATLDSKAAYKEAELVIIATPTNYNSRINYFDTQHVETVIKEVLSVNSHATLIIKSTIPIGFITEMRQKFQTDRIIFSPEFLRESKALYDNLYPSRIIVSCEENDSPKVKADAEKFALLLKSAAKKNNVPVLIMGASEAEAVKLFANTYLALRVAYFNELDTYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    )
    assert len(PROTEIN_SEQUENCE) == 402

    # Load structure to compute NAD distances
    pdb_path = Path(__file__).parent / "1dli_g1p.pdb"
    st = gemmi.read_structure(str(pdb_path))
    model = st[0]

    mo.md(f"**Loaded structure**: {pdb_path.name}")
    return PROTEIN_SEQUENCE, model, pdb_path


@app.cell
def _(model):
    """Cell 2: Extract protein and NAD coordinates for distance calculation."""
    chain_a = model.find_chain("A")
    chain_b = model.find_chain("B")  # NAD

    # Get CA coordinates for chain A (protein)
    ca_coords = []
    for residue in chain_a:
        try:
            ca = residue.sole_atom("CA")
            ca_coords.append([ca.pos.x, ca.pos.y, ca.pos.z])
        except:
            ca_coords.append([np.nan, np.nan, np.nan])
    ca_coords = np.array(ca_coords)

    # Get all NAD atom coordinates
    nad_coords = []
    for residue in chain_b:
        for atom in residue:
            nad_coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    nad_coords = np.array(nad_coords)

    mo.md(
        f"**Protein residues**: {len(ca_coords)}  \n"
        f"**NAD atoms**: {len(nad_coords)}"
    )
    return ca_coords, nad_coords


@app.cell
def _(PROTEIN_SEQUENCE, ca_coords, nad_coords):
    """Cell 3: Calculate NAD distances and build designable masks.

    Two masks are created:
    1. boltz_designable_mask: G1P lobe only (for backbone optimization with Boltz2)
    2. mpnn_designable_mask: G1P lobe + NAD lobe >8Å from NAD (for sequence sampling with MPNN)

    This ensures NAD lobe backbone is preserved (fixed in Boltz2) while allowing
    sequence diversity in the far-from-NAD region via MPNN.
    """
    # Compute pairwise distances: (n_residues, n_nad_atoms)
    diff = ca_coords[:, None, :] - nad_coords[None, :, :]
    distances = np.sqrt((diff ** 2).sum(-1))
    min_dist_to_nad = np.nanmin(distances, axis=1)

    NAD_DISTANCE_THRESHOLD = 8.0

    # NAD lobe residues far from NAD (>8Å)
    nad_lobe_mask = np.zeros(402, dtype=bool)
    nad_lobe_mask[:228] = True
    far_from_nad = min_dist_to_nad > NAD_DISTANCE_THRESHOLD
    designable_in_nad_lobe = nad_lobe_mask & far_from_nad

    # Mask 1: Boltz2 optimization (G1P lobe only, preserves NAD lobe backbone)
    boltz_designable_mask = np.zeros(402, dtype=bool)
    boltz_designable_mask[228:402] = True  # G1P lobe
    boltz_designable_mask[259] = False  # Fix CYS260

    # Mask 2: MPNN sampling (G1P lobe + NAD lobe far from NAD)
    mpnn_designable_mask = boltz_designable_mask.copy()
    mpnn_designable_mask[:228] = designable_in_nad_lobe[:228]

    # Build wildtype_with_x for Boltz2 SetPositions (G1P lobe only)
    wildtype_with_x = "".join(
        "X" if boltz_designable_mask[i] else PROTEIN_SEQUENCE[i]
        for i in range(402)
    )

    n_boltz_variable = boltz_designable_mask.sum()
    n_mpnn_variable = mpnn_designable_mask.sum()
    n_nad_lobe_designable = designable_in_nad_lobe.sum()

    mo.md(
        f"### Designable Positions\n\n"
        f"**NAD distance threshold**: {NAD_DISTANCE_THRESHOLD} Å\n\n"
        f"| Region | Boltz2 (backbone) | MPNN (sequence) |\n"
        f"|--------|-------------------|------------------|\n"
        f"| G1P lobe (229-402, excl. CYS260) | {173} | {173} |\n"
        f"| NAD lobe >8Å from NAD | 0 (fixed) | {n_nad_lobe_designable} |\n"
        f"| **Total variable** | **{n_boltz_variable}** | **{n_mpnn_variable}** |\n\n"
        f"*NAD lobe backbone is fixed during Boltz2 optimization. MPNN designs sequences for the fixed backbone.*"
    )
    return (
        NAD_DISTANCE_THRESHOLD,
        boltz_designable_mask,
        designable_in_nad_lobe,
        min_dist_to_nad,
        mpnn_designable_mask,
        n_boltz_variable,
        wildtype_with_x,
    )


@app.cell
def _(
    NAD_DISTANCE_THRESHOLD,
    boltz_designable_mask,
    designable_in_nad_lobe,
    min_dist_to_nad,
    mpnn_designable_mask,
):
    """Cell 4: Visualize designable regions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Distance to NAD for NAD lobe
    ax = axes[0]
    colors = ['green' if d else 'red' for d in designable_in_nad_lobe[:228]]
    ax.bar(range(228), min_dist_to_nad[:228], color=colors)
    ax.axhline(y=NAD_DISTANCE_THRESHOLD, color='black', linestyle='--', label=f'{NAD_DISTANCE_THRESHOLD}Å threshold')
    ax.set_xlabel('Residue index (NAD lobe)')
    ax.set_ylabel('Distance to NAD (Å)')
    ax.set_title('NAD lobe: distance to NAD')
    ax.legend()

    # Plot 2: Boltz2 designable mask
    ax = axes[1]
    ax.bar(range(402), boltz_designable_mask.astype(int), color='blue', alpha=0.7)
    ax.set_xlabel('Residue index')
    ax.set_ylabel('Designable')
    ax.set_title('Boltz2: G1P lobe only (backbone + sequence)')
    ax.axvline(x=228, color='red', linestyle='--', label='Lobe boundary')
    ax.axvline(x=259, color='orange', linestyle='--', label='CYS260')
    ax.legend()

    # Plot 3: MPNN designable mask
    ax = axes[2]
    ax.bar(range(402), mpnn_designable_mask.astype(int), color='green', alpha=0.7)
    ax.set_xlabel('Residue index')
    ax.set_ylabel('Designable')
    ax.set_title('MPNN: G1P lobe + NAD lobe >8Å (sequence only)')
    ax.axvline(x=228, color='red', linestyle='--', label='Lobe boundary')
    ax.axvline(x=259, color='orange', linestyle='--', label='CYS260')
    ax.legend()

    plt.tight_layout()
    fig
    return


@app.cell
def _():
    """Cell 5: Load models."""
    boltz2 = Boltz2()
    mpnn = load_mpnn(backbone_noise=0.0)
    mo.md("**Loaded models**: Boltz2, ProteinMPNN")
    return boltz2, mpnn


@app.cell
def _(pdb_path):
    """Cell 6: Convert template PDB to CIF for Boltz2."""
    st_template = gemmi.read_structure(str(pdb_path))

    # Extract chain A only for the template
    template_st = gemmi.Structure()
    template_model = gemmi.Model("0")
    template_chain_a = st_template[0].find_chain("A")

    new_chain = gemmi.Chain("A")
    for res in template_chain_a:
        new_chain.add_residue(res)

    ent = gemmi.Entity("A")
    ent.entity_type = gemmi.EntityType.Polymer
    ent.polymer_type = gemmi.PolymerType.PeptideL
    ent.subchains = ["A"]
    ent.full_sequence = [r.name for r in new_chain]
    for r in new_chain:
        r.subchain = "A"

    template_model.add_chain(new_chain)
    template_st.add_model(template_model)
    template_st.entities = gemmi.EntityList([ent])
    template_st.assign_subchains()
    template_st.setup_entities()
    template_st.ensure_entities()
    template_st.assign_label_seq_id()

    template_cif_file = NamedTemporaryFile(suffix=".cif", delete=False)
    doc = template_st.make_mmcif_document()
    doc.write_file(template_cif_file.name)

    mo.md(f"Template CIF written to `{template_cif_file.name}`")
    return (template_cif_file,)


@app.cell
def _(PROTEIN_SEQUENCE, template_cif_file):
    """Cell 7: Construct YAML and generate Boltz2 features."""
    import yaml as _yaml

    yaml_dict = {
        "version": 1,
        "sequences": [
            {"protein": {"id": ["A"], "sequence": PROTEIN_SEQUENCE, "msa": "empty"}},
            {"ligand": {"id": ["B"], "ccd": "NAD"}},
            {"ligand": {"id": ["C"], "ccd": "G1P"}},
        ],
        "templates": [
            {
                "cif": template_cif_file.name,
                "chain_id": ["A"],
                "template_id": ["A"],
            }
        ],
    }
    yaml_str = _yaml.dump(yaml_dict, default_flow_style=False, sort_keys=False)
    print("Generated YAML:")
    print(yaml_str)

    features, writer = load_features_and_structure_writer(yaml_str)

    template_mask_sum = np.sum(np.array(features["template_mask"]))
    mo.md(f"**Template mask sum**: {template_mask_sum} (should be > 0)")
    assert template_mask_sum > 0, "Template mask is all zeros — CIF may be malformed"
    return features, writer


@app.cell
def _(features):
    """Cell 8: Identify ligand token indices."""
    asym_id = np.array(features["asym_id"]).squeeze()
    protein_len = 402

    target_asym = asym_id[protein_len:]
    unique_target_asyms = np.unique(target_asym)

    nad_idx = [int(i) for i, a in enumerate(target_asym) if a == unique_target_asyms[0]]
    g1p_idx = [int(i) for i, a in enumerate(target_asym) if a == unique_target_asyms[-1]]

    mo.md(
        f"**Total tokens**: {len(asym_id)}  \n"
        f"**Protein tokens**: {protein_len}  \n"
        f"**NAD token indices** (relative to target): {nad_idx}  \n"
        f"**G1P token indices** (relative to target): {g1p_idx}"
    )
    return g1p_idx, nad_idx


@app.cell
def _(PROTEIN_SEQUENCE, boltz2, features):
    """Cell 9: Get wildtype distogram for scaffold preservation."""
    wt_pssm = jax.nn.one_hot(
        jnp.array([TOKENS.index(aa) for aa in PROTEIN_SEQUENCE]), 20
    )

    wt_output = boltz2.model_output(
        PSSM=wt_pssm,
        features=features,
        recycling_steps=3,
        key=jax.random.key(42),
    )

    wt_distogram_probs = jax.nn.softmax(wt_output.distogram_logits)[:402, :402]
    wt_plddt = wt_output.plddt

    mo.md(
        f"**Wildtype mean pLDDT**: {float(wt_plddt[:402].mean()):.3f}  \n"
        f"**Distogram shape**: {wt_distogram_probs.shape}"
    )
    return (wt_distogram_probs,)


@app.cell
def _(
    boltz2,
    features,
    g1p_idx,
    mpnn,
    nad_idx,
    wildtype_with_x,
    wt_distogram_probs,
):
    """Cell 10: Define composite loss with InverseFoldingSequenceRecovery.

    The loss combines:
    1. Structure prediction losses (pLDDT, contacts, PAE, distogram)
    2. InverseFoldingSequenceRecovery: pushes sequence toward what MPNN would sample

    This term samples sequences from MPNN and computes the inner product with
    the current soft sequence, effectively guiding optimization toward
    MPNN-preferred solutions. Empirically speeds up design and increases
    filter pass rates.
    """
    nad_idx_arr = np.array(nad_idx)
    g1p_idx_arr = np.array(g1p_idx)
    nad_lobe_idx = np.arange(0, 228)
    g1p_lobe_idx = np.arange(228, 402)

    inner_loss = (
        # Confidence
        1.0 * sp.PLDDTLoss()
        # Intra-protein fold quality
        + 1.0 * sp.WithinBinderContact()
        + 0.5 * sp.WithinBinderPAE()
        # NAD binding (fixed lobe maintains NAD contacts)
        + 1.0 * sp.BinderTargetContact(
            epitope_idx=nad_idx_arr,
            paratope_idx=nad_lobe_idx,
            contact_distance=12.0,
        )
        # G1P binding (redesigned lobe must bind G1P)
        + 2.0 * sp.BinderTargetContact(
            epitope_idx=g1p_idx_arr,
            paratope_idx=g1p_lobe_idx,
            contact_distance=12.0,
        )
        # Interface PAE
        + 0.1 * sp.BinderTargetPAE()
        + 0.1 * sp.TargetBinderPAE()
        # Scaffold adherence via distogram cross-entropy
        + 2.0 * SoftClip(
            sp.DistogramCE(wt_distogram_probs, name="scaffold_CE"),
            2.5, 3.0,
        )
        # ProteinMPNN sequence recovery loss
        # Pushes sequence toward what MPNN would sample for the predicted structure
        + 5.0 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.01))
    )

    boltz_loss = boltz2.build_loss(
        loss=inner_loss,
        features=features,
        recycling_steps=1,
        sampling_steps=25,
    )

    # Restrict to variable positions (G1P lobe only for Boltz2)
    design_loss = SetPositions.from_sequence(wildtype_with_x, boltz_loss)

    mo.md(
        "### Composite Loss\n\n"
        "| Term | Weight | Description |\n"
        "|------|--------|-------------|\n"
        "| PLDDTLoss | 1.0 | Maximize confidence |\n"
        "| WithinBinderContact | 1.0 | Intra-protein contacts |\n"
        "| WithinBinderPAE | 0.5 | Intra-protein accuracy |\n"
        "| BinderTargetContact (NAD) | 1.0 | NAD lobe ↔ NAD |\n"
        "| BinderTargetContact (G1P) | 2.0 | G1P lobe ↔ G1P |\n"
        "| PAE losses | 0.1 | Interface accuracy |\n"
        "| DistogramCE | 2.0 | Scaffold preservation |\n"
        "| **InverseFoldingSequenceRecovery** | 5.0 | **Guides toward MPNN-preferred sequences** |"
    )

    jax.clear_caches()
    return (design_loss,)


@app.cell
def _(design_loss, n_boltz_variable):
    """Cell 11: Phase 1 — Joint optimization with Boltz2 + MPNN loss."""
    mo.md("### Phase 1: Joint Backbone + Sequence Optimization")

    _, PSSM_soft = simplex_APGM(
        loss_function=design_loss,
        x=jax.nn.softmax(
            0.5 * jax.random.gumbel(
                key=jax.random.key(np.random.randint(100000)),
                shape=(n_boltz_variable, 20),
            )
        ),
        n_steps=100,
        stepsize=0.1,
        momentum=0.0,
        serial_evaluation=False,
    )
    return (PSSM_soft,)


@app.cell
def _(PSSM_soft, design_loss):
    """Cell 12: Phase 1 — Sharpening to discrete sequence."""
    PSSM_sharp, _ = simplex_APGM(
        loss_function=design_loss,
        n_steps=25,
        x=PSSM_soft,
        stepsize=0.2,
        scale=1.1,
        serial_evaluation=False,
    )
    PSSM_sharp, _ = simplex_APGM(
        loss_function=design_loss,
        n_steps=25,
        x=PSSM_sharp,
        stepsize=0.2,
        scale=1.5,
        serial_evaluation=False,
    )
    return (PSSM_sharp,)


@app.cell
def _(PSSM_sharp, boltz2, design_loss, features, writer):
    """Cell 13: Recover full sequence and predict Phase 1 structure."""
    full_pssm_phase1 = design_loss.sequence(PSSM_sharp)
    designed_tokens_phase1 = full_pssm_phase1.argmax(-1)
    designed_sequence_phase1 = "".join([TOKENS[int(i)] for i in designed_tokens_phase1])

    # Predict structure for Phase 1 result
    phase1_pred = boltz2.predict(
        PSSM=full_pssm_phase1,
        features=features,
        writer=writer,
        recycling_steps=3,
        key=jax.random.key(99),
    )

    mo.md(
        f"### Phase 1 Result\n\n"
        f"**iPTM**: {float(phase1_pred.iptm):.3f}  \n"
        f"**Mean pLDDT**: {float(phase1_pred.plddt.mean()):.3f}  \n\n"
        f"**Designed sequence**:\n\n`{designed_sequence_phase1}`"
    )
    pdb_viewer(phase1_pred.st)
    return designed_sequence_phase1, full_pssm_phase1, phase1_pred


@app.cell
def _(
    PROTEIN_SEQUENCE,
    boltz2,
    features,
    full_pssm_phase1,
    mpnn,
    mpnn_designable_mask,
):
    """Cell 14: Phase 2 — Diverse sequence sampling with ProteinMPNN.

    Use the Boltz2-predicted structure from Phase 1 as the backbone for MPNN sampling.
    MPNN can design both G1P lobe AND NAD lobe residues >8Å from NAD.
    Fixed positions are enforced via bias.
    """
    mo.md("### Phase 2: Diverse Sequence Sampling with ProteinMPNN")

    # Get Boltz2 output for the Phase 1 sequence
    phase1_output = boltz2.model_output(
        PSSM=full_pssm_phase1,
        features=features,
        recycling_steps=3,
        key=jax.random.key(42),
    )

    # Build bias to fix non-designable positions
    # Bias is -1e6 for non-wildtype amino acids at fixed positions
    fixed_position_bias = jnp.zeros((402, 20))
    for pos_idx in range(402):
        if not mpnn_designable_mask[pos_idx]:
            wt_idx = TOKENS.index(PROTEIN_SEQUENCE[pos_idx])
            bias_row = jnp.full(20, -1e6)
            bias_row = bias_row.at[wt_idx].set(0.0)
            fixed_position_bias = fixed_position_bias.at[pos_idx].set(bias_row)

    # Sample diverse sequences
    n_samples = 8
    diverse_sequences = []
    for sample_idx in range(n_samples):
        key_i = jax.random.key(sample_idx * 1000)
        seq_tokens = jacobi_inverse_fold(
            mpnn=mpnn,
            binder_length=402,
            output=phase1_output,
            temp=0.1,
            key=key_i,
            jacobi_iterations=20,
            bias=fixed_position_bias,
        )
        seq_str = "".join([TOKENS[int(t)] for t in seq_tokens])
        diverse_sequences.append(seq_str)

    mo.md(f"**Generated {n_samples} diverse sequences using ProteinMPNN**")
    return (diverse_sequences,)


@app.cell
def _(
    PROTEIN_SEQUENCE,
    boltz2,
    diverse_sequences,
    features,
    mpnn_designable_mask,
    writer,
):
    """Cell 15: Validate diverse sequences with Boltz2."""
    mo.md("### Phase 3: Validation of Diverse Sequences")

    validations = []
    for val_idx, seq in enumerate(diverse_sequences):
        # Convert to PSSM
        pssm = jax.nn.one_hot(
            jnp.array([TOKENS.index(aa) for aa in seq]), 20
        )
        # Predict with Boltz2
        pred = boltz2.predict(
            PSSM=pssm,
            features=features,
            writer=writer,
            recycling_steps=3,
            key=jax.random.key(200 + val_idx),
        )

        # Count mutations
        mutations_g1p = sum(1 for j in range(228, 402) if j != 259 and seq[j] != PROTEIN_SEQUENCE[j])
        mutations_nad = sum(1 for j in range(228) if seq[j] != PROTEIN_SEQUENCE[j])

        # Check constraints
        cys260_ok = seq[259] == 'C'
        fixed_ok = all(seq[j] == PROTEIN_SEQUENCE[j] for j in range(402) if not mpnn_designable_mask[j])

        validations.append({
            'index': val_idx + 1,
            'sequence': seq,
            'plddt': float(pred.plddt.mean()),
            'iptm': float(pred.iptm),
            'mutations_g1p': mutations_g1p,
            'mutations_nad': mutations_nad,
            'cys260_ok': cys260_ok,
            'fixed_ok': fixed_ok,
            'structure': pred.st,
        })

    # Display summary table
    summary_md = "| Design | pLDDT | iPTM | G1P mut | NAD mut | CYS260 | Fixed OK |\n"
    summary_md += "|--------|-------|------|---------|---------|--------|----------|\n"
    for val_entry in validations:
        summary_md += f"| {val_entry['index']} | {val_entry['plddt']:.3f} | {val_entry['iptm']:.3f} | {val_entry['mutations_g1p']} | {val_entry['mutations_nad']} | {'OK' if val_entry['cys260_ok'] else 'X'} | {'OK' if val_entry['fixed_ok'] else 'X'} |\n"

    mo.md(summary_md)
    return (validations,)


@app.cell
def _(PROTEIN_SEQUENCE, designed_sequence_phase1, phase1_pred, validations):
    """Cell 16: Compare Phase 1 vs Phase 2 sequences."""
    # Find best Phase 2 sequence by pLDDT
    best_phase2 = max(validations, key=lambda v: v['plddt'])

    # Count Phase 1 mutations
    phase1_mutations_g1p = sum(1 for j in range(228, 402) if j != 259 and designed_sequence_phase1[j] != PROTEIN_SEQUENCE[j])

    mo.md(
        f"### Comparison\n\n"
        f"**Phase 1 sequence** (Boltz2 + MPNN loss optimization):\n\n`{designed_sequence_phase1}`\n\n"
        f"**Best Phase 2 sequence** (Design {best_phase2['index']}, pLDDT={best_phase2['plddt']:.3f}):\n\n`{best_phase2['sequence']}`\n\n"
        f"**Differences**:\n"
        f"- Phase 1 mutations (G1P lobe only): {phase1_mutations_g1p}\n"
        f"- Phase 2 mutations (G1P lobe): {best_phase2['mutations_g1p']}\n"
        f"- Phase 2 mutations (NAD lobe >8Å): {best_phase2['mutations_nad']}"
    )

    # Show both structures
    mo.ui.tabs({
        "Phase 1": pdb_viewer(phase1_pred.st),
        "Best Phase 2": pdb_viewer(best_phase2['structure']),
    })
    return


@app.cell
def _(designed_sequence_phase1, phase1_pred, validations):
    """Cell 17: Export results to disk and provide download buttons."""
    # Create output directory
    output_dir = Path(__file__).parent / "output_hybrid"
    output_dir.mkdir(exist_ok=True)

    # Build FASTA content
    fasta_lines = [
        f">Phase1_Boltz2_MPNN_optimized",
        designed_sequence_phase1,
    ]
    for fasta_val in validations:
        fasta_lines.append(f">Phase2_MPNN_design_{fasta_val['index']}_pLDDT_{fasta_val['plddt']:.3f}")
        fasta_lines.append(fasta_val['sequence'])

    fasta_content = "\n".join(fasta_lines)

    # Save FASTA to disk
    fasta_path = output_dir / "designed_sequences.fasta"
    fasta_path.write_text(fasta_content)

    # Save Phase 1 PDB to disk
    phase1_pdb_path = output_dir / "phase1_optimized.pdb"
    phase1_pdb_path.write_text(phase1_pred.st.make_pdb_string())

    # Save all Phase 2 PDBs to disk
    for save_val in validations:
        phase2_pdb_path = output_dir / f"phase2_design_{save_val['index']}_pLDDT_{save_val['plddt']:.3f}.pdb"
        phase2_pdb_path.write_text(save_val['structure'].make_pdb_string())

    saved_files = list(output_dir.glob("*.pdb")) + list(output_dir.glob("*.fasta"))
    mo.md(
        f"### Export\n\n"
        f"**Saved {len(saved_files)} files to `{output_dir}/`**:\n"
        + "\n".join(f"- `{f.name}`" for f in sorted(saved_files))
    )

    # Also provide download buttons for interactive use
    mo.hstack([
        mo.download(
            fasta_content,
            filename="designed_sequences.fasta",
            label="Download FASTA (all)",
        ),
        mo.download(
            phase1_pred.st.make_pdb_string(),
            filename="phase1_optimized.pdb",
            label="Download Phase 1 PDB",
        ),
    ])
    return


@app.cell
def _(
    PROTEIN_SEQUENCE,
    designed_sequence_phase1,
    mpnn_designable_mask,
    validations,
):
    """Cell 18: Final validation checks."""
    def validate_sequence(seq, label):
        # NAD lobe fixed positions unchanged
        nad_fixed_ok = all(
            seq[j] == PROTEIN_SEQUENCE[j]
            for j in range(228)
            if not mpnn_designable_mask[j]
        )
        # CYS260 preserved
        cys260_ok = seq[259] == "C"
        # Count mutations
        mutations_total = sum(1 for j in range(402) if seq[j] != PROTEIN_SEQUENCE[j])
        mutations_g1p = sum(1 for j in range(228, 402) if seq[j] != PROTEIN_SEQUENCE[j])
        mutations_nad = sum(1 for j in range(228) if seq[j] != PROTEIN_SEQUENCE[j])

        return {
            'label': label,
            'nad_fixed_ok': nad_fixed_ok,
            'cys260_ok': cys260_ok,
            'mutations_total': mutations_total,
            'mutations_g1p': mutations_g1p,
            'mutations_nad': mutations_nad,
        }

    checks = [validate_sequence(designed_sequence_phase1, "Phase 1 (Boltz2 + MPNN)")]
    for check_val in validations:
        checks.append(validate_sequence(check_val['sequence'], f"Phase 2 Design {check_val['index']}"))

    # Display
    md_output = "### Validation Summary\n\n"
    md_output += "| Sequence | NAD fixed | CYS260 | Total mut | G1P mut | NAD mut |\n"
    md_output += "|----------|-----------|--------|-----------|---------|----------|\n"
    for c in checks:
        md_output += f"| {c['label']} | {'OK' if c['nad_fixed_ok'] else 'FAIL'} | {'OK' if c['cys260_ok'] else 'FAIL'} | {c['mutations_total']} | {c['mutations_g1p']} | {c['mutations_nad']} |\n"

    all_valid = all(c['nad_fixed_ok'] and c['cys260_ok'] for c in checks)
    if all_valid:
        md_output += "\n**All constraints satisfied!**"
    else:
        md_output += "\n**WARNING: Some constraints violated!**"

    mo.md(md_output)
    return


if __name__ == "__main__":
    app.run()
