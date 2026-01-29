import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

with app.setup:
    # GPU memory settings - must be before importing JAX
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import gemmi
    from pathlib import Path

    from mosaic.proteinmpnn.mpnn import ProteinMPNN, load_mpnn, MPNN_ALPHABET
    from mosaic.losses.protein_mpnn import load_chain, boltz_to_mpnn_matrix
    from mosaic.common import TOKENS


@app.cell
def _():
    """Define the protein sequence and load structure."""
    PROTEIN_SEQUENCE = (
        "MKIAVAGSGYVGLSLGVLLSLQNEVTIVDILPSKVDKINNGLSPIQDEYIEYYLKSKQLSIKATLDSKAAYKEAELVIIATPTNYNSRINYFDTQHVETVIKEVLSVNSHATLIIKSTIPIGFITEMRQKFQTDRIIFSPEFLRESKALYDNLYPSRIIVSCEENDSPKVKADAEKFALLLKSAAKKNNVPVLIMGASEAEAVKLFANTYLALRVAYFNELDTYAESRKLNSHMIIQGISYDDRIGMHYNNPSFGYGGYCLPKDTKQLLANYNNIPQTLIEAIVSSNNVRKSYIAKQIINVLKEQESPVKVVGVYRLIMKSNSDNFRESAIKDVIDILKSKDIKIIIYEPMLNKLESEDQSVLVNDLENFKKQANIIVTNRYDNELQDVKNKVYSRDIFGRD"
    )
    assert len(PROTEIN_SEQUENCE) == 402

    # Load structure
    pdb_path = Path(__file__).parent / "1dli_g1p.pdb"
    st = gemmi.read_structure(str(pdb_path))
    model = st[0]

    mo.md(f"**Loaded structure**: {pdb_path.name}")
    return PROTEIN_SEQUENCE, st, model


@app.cell
def _(model):
    """Extract protein chain A coordinates and NAD coordinates."""
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
def _(ca_coords, nad_coords):
    """Calculate minimum distance from each residue to NAD."""
    # Compute pairwise distances: (n_residues, n_nad_atoms)
    diff = ca_coords[:, None, :] - nad_coords[None, :, :]
    distances = np.sqrt((diff ** 2).sum(-1))

    # Minimum distance to any NAD atom for each residue
    min_dist_to_nad = np.nanmin(distances, axis=1)

    # Find NAD lobe residues (1-228, 0-indexed: 0-227) that are >8Å from NAD
    NAD_DISTANCE_THRESHOLD = 8.0
    nad_lobe_mask = np.zeros(402, dtype=bool)
    nad_lobe_mask[:228] = True

    far_from_nad = min_dist_to_nad > NAD_DISTANCE_THRESHOLD
    designable_in_nad_lobe = nad_lobe_mask & far_from_nad

    n_designable_nad_lobe = designable_in_nad_lobe.sum()

    mo.md(
        f"**NAD distance threshold**: {NAD_DISTANCE_THRESHOLD} Å  \n"
        f"**NAD lobe residues >8Å from NAD**: {n_designable_nad_lobe} / 228  \n"
        f"**Closest NAD lobe residue to NAD**: {min_dist_to_nad[:228].min():.1f} Å  \n"
        f"**Farthest NAD lobe residue from NAD**: {min_dist_to_nad[:228].max():.1f} Å"
    )
    return min_dist_to_nad, designable_in_nad_lobe, NAD_DISTANCE_THRESHOLD


@app.cell
def _(designable_in_nad_lobe, min_dist_to_nad, NAD_DISTANCE_THRESHOLD):
    """Build the full designable mask."""
    # Designable positions:
    # 1. G1P lobe (229-402, 0-indexed: 228-401) EXCEPT CYS260 (0-indexed: 259)
    # 2. NAD lobe residues >8Å from NAD

    designable_mask = np.zeros(402, dtype=bool)

    # G1P lobe (all except CYS260)
    designable_mask[228:402] = True
    designable_mask[259] = False  # Fix CYS260

    # NAD lobe residues far from NAD
    designable_mask[:228] = designable_in_nad_lobe[:228]

    # Summary
    n_designable_g1p = designable_mask[228:402].sum()
    n_designable_nad = designable_mask[:228].sum()
    n_fixed = (~designable_mask).sum()

    mo.md(
        f"### Designable positions\n"
        f"- **G1P lobe** (229-402 minus CYS260): {n_designable_g1p}\n"
        f"- **NAD lobe** (>8Å from NAD): {n_designable_nad}\n"
        f"- **Total designable**: {designable_mask.sum()}\n"
        f"- **Fixed positions**: {n_fixed}\n\n"
        f"**Fixed residues include**: NAD lobe residues within 8Å of NAD + CYS260"
    )

    # Plot distance distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.bar(range(228), min_dist_to_nad[:228], color=['green' if d else 'red' for d in designable_in_nad_lobe[:228]])
    ax.axhline(y=NAD_DISTANCE_THRESHOLD, color='black', linestyle='--', label=f'{NAD_DISTANCE_THRESHOLD}Å threshold')
    ax.set_xlabel('Residue index (NAD lobe)')
    ax.set_ylabel('Distance to NAD (Å)')
    ax.set_title('NAD lobe: distance to NAD')
    ax.legend()

    ax = axes[1]
    ax.bar(range(402), designable_mask.astype(int), color='green', alpha=0.7)
    ax.set_xlabel('Residue index')
    ax.set_ylabel('Designable')
    ax.set_title('Designable positions (green=designable)')
    ax.axvline(x=228, color='blue', linestyle='--', label='Lobe boundary')
    ax.axvline(x=259, color='red', linestyle='--', label='CYS260')
    ax.legend()

    plt.tight_layout()
    fig
    return designable_mask, n_designable_g1p, n_designable_nad


@app.cell
def _():
    """Load ProteinMPNN model."""
    mpnn = load_mpnn(backbone_noise=0.0)
    mo.md("**Loaded ProteinMPNN** (v_48_020)")
    return (mpnn,)


@app.cell
def _(PROTEIN_SEQUENCE, model, mpnn):
    """Encode the structure with ProteinMPNN."""
    # Clone and clean structure (protein only for encoding)
    st_clean = model.clone()

    # Extract chain A only
    chain_a = model.find_chain("A")
    sequence, coords = load_chain(chain_a)

    assert sequence == PROTEIN_SEQUENCE, f"Sequence mismatch: {len(sequence)} vs {len(PROTEIN_SEQUENCE)}"

    # Prepare ProteinMPNN inputs
    residue_idx = np.arange(len(sequence))
    chain_encoding = np.zeros(len(sequence), dtype=np.int32)
    mask = jnp.ones(len(sequence), dtype=jnp.int32)

    # Encode structure
    h_V, h_E, E_idx = mpnn.encode(
        X=coords,
        mask=mask,
        residue_idx=residue_idx,
        chain_encoding_all=chain_encoding,
        key=jax.random.key(42),
    )

    mo.md(f"**Encoded structure**: {len(sequence)} residues")
    return sequence, coords, h_V, h_E, E_idx, mask, residue_idx, chain_encoding


@app.cell
def _(PROTEIN_SEQUENCE, designable_mask, mpnn, h_V, h_E, E_idx, mask):
    """Design sequences using ProteinMPNN with fixed positions."""

    def design_sequence(key, temperature=0.1, num_iterations=10):
        """
        Design a sequence using Jacobi iteration with fixed positions.

        Fixed positions are set by giving them a very high decoding order,
        so they are "decoded last" and don't get modified.
        """
        n_residues = len(PROTEIN_SEQUENCE)

        # Initialize with wildtype sequence
        wt_tokens = jnp.array([TOKENS.index(aa) for aa in PROTEIN_SEQUENCE])
        sequence_tokens = wt_tokens.copy()

        # Gumbel noise for sampling
        gumbel = jax.random.gumbel(key, (n_residues, len(MPNN_ALPHABET)))

        def tokens_to_logits(tokens):
            """Get ProteinMPNN logits for current sequence."""
            # Convert to one-hot in MPNN alphabet
            seq_onehot = jax.nn.one_hot(tokens, 20)  # Boltz alphabet
            seq_mpnn = seq_onehot @ boltz_to_mpnn_matrix()

            # Decoding order: designable positions get low values (decoded first)
            # Fixed positions get high values (decoded last, conditioned on)
            decoding_order = jnp.where(
                jnp.array(designable_mask),
                jax.random.uniform(key, (n_residues,)),  # Random order for designable
                jnp.ones(n_residues) * 10.0,  # High value for fixed (decoded last)
            )

            logits = mpnn.decode(
                S=seq_mpnn,
                h_V=h_V,
                h_E=h_E,
                E_idx=E_idx,
                mask=mask,
                decoding_order=decoding_order,
            )[0]

            # Convert back to Boltz alphabet
            return logits @ boltz_to_mpnn_matrix().T

        # Jacobi iteration
        for _ in range(num_iterations):
            logits = tokens_to_logits(sequence_tokens)

            # Sample new tokens (only for designable positions)
            new_tokens = (logits / temperature + gumbel[:, :20]).argmax(-1)

            # Keep fixed positions unchanged
            sequence_tokens = jnp.where(
                jnp.array(designable_mask),
                new_tokens,
                wt_tokens,
            )

        return sequence_tokens

    # Design multiple sequences
    n_designs = 8
    keys = jax.random.split(jax.random.key(0), n_designs)

    designed_sequences = []
    for i, key in enumerate(keys):
        tokens = design_sequence(key, temperature=0.1, num_iterations=20)
        seq = "".join([TOKENS[int(t)] for t in tokens])
        designed_sequences.append(seq)

    mo.md(f"**Designed {n_designs} sequences**")
    return designed_sequences, design_sequence


@app.cell
def _(PROTEIN_SEQUENCE, designed_sequences, designable_mask):
    """Analyze designed sequences."""

    def analyze_sequence(seq, label):
        # Count mutations
        mutations = sum(1 for i, (a, b) in enumerate(zip(PROTEIN_SEQUENCE, seq)) if a != b)
        mutations_designable = sum(
            1 for i, (a, b) in enumerate(zip(PROTEIN_SEQUENCE, seq))
            if a != b and designable_mask[i]
        )
        mutations_fixed = mutations - mutations_designable

        # Check constraints
        cys260_ok = seq[259] == 'C'
        nad_lobe_fixed_ok = all(
            seq[i] == PROTEIN_SEQUENCE[i]
            for i in range(228)
            if not designable_mask[i]
        )

        return {
            'label': label,
            'mutations': mutations,
            'mutations_designable': mutations_designable,
            'mutations_fixed': mutations_fixed,
            'cys260_ok': cys260_ok,
            'nad_lobe_fixed_ok': nad_lobe_fixed_ok,
            'sequence': seq,
        }

    analyses = [analyze_sequence(seq, f"Design {i+1}") for i, seq in enumerate(designed_sequences)]

    # Display summary
    summary_md = "### Designed Sequences Analysis\n\n"
    summary_md += "| Design | Mutations | In designable | In fixed | CYS260 | NAD contacts |\n"
    summary_md += "|--------|-----------|---------------|----------|--------|-------------|\n"
    for a in analyses:
        summary_md += f"| {a['label']} | {a['mutations']} | {a['mutations_designable']} | {a['mutations_fixed']} | {'✓' if a['cys260_ok'] else '✗'} | {'✓' if a['nad_lobe_fixed_ok'] else '✗'} |\n"

    mo.md(summary_md)
    return analyses


@app.cell
def _(analyses, PROTEIN_SEQUENCE, designable_mask):
    """Visualize sequence differences."""
    # Pick the first design for detailed view
    design = analyses[0]
    seq = design['sequence']

    # Create mutation map
    mutation_positions = []
    mutation_labels = []
    for i, (wt, des) in enumerate(zip(PROTEIN_SEQUENCE, seq)):
        if wt != des:
            mutation_positions.append(i)
            mutation_labels.append(f"{wt}{i+1}{des}")

    fig, ax = plt.subplots(figsize=(14, 3))

    # Plot designable regions
    for i in range(402):
        color = 'lightgreen' if designable_mask[i] else 'lightcoral'
        ax.axvspan(i-0.5, i+0.5, color=color, alpha=0.3)

    # Plot mutations
    for pos in mutation_positions:
        ax.axvline(x=pos, color='blue', linewidth=2, alpha=0.7)

    ax.axvline(x=228, color='black', linestyle='--', linewidth=2, label='Lobe boundary')
    ax.axvline(x=259, color='red', linestyle='--', linewidth=2, label='CYS260')

    ax.set_xlim(-1, 403)
    ax.set_xlabel('Residue position')
    ax.set_title(f'Design 1: {len(mutation_positions)} mutations (green=designable, red=fixed)')
    ax.legend()

    plt.tight_layout()

    mo.vstack([
        mo.md(f"**Mutations**: {', '.join(mutation_labels[:20])}{'...' if len(mutation_labels) > 20 else ''}"),
        fig
    ])
    return


@app.cell
def _(analyses):
    """Display sequences for download."""
    output_lines = []
    for a in analyses:
        output_lines.append(f">{a['label']}")
        output_lines.append(a['sequence'])

    fasta_content = "\n".join(output_lines)

    mo.vstack([
        mo.md("### Designed Sequences (FASTA format)"),
        mo.download(
            fasta_content,
            filename="designed_sequences.fasta",
            label="Download FASTA",
        ),
        mo.md(f"```\n{fasta_content[:500]}...\n```" if len(fasta_content) > 500 else f"```\n{fasta_content}\n```"),
    ])
    return


@app.cell
def _(PROTEIN_SEQUENCE, designed_sequences, designable_mask):
    """Validate constraints are satisfied."""
    all_valid = True
    issues = []

    for i, seq in enumerate(designed_sequences):
        # Check CYS260
        if seq[259] != 'C':
            all_valid = False
            issues.append(f"Design {i+1}: CYS260 mutated to {seq[259]}")

        # Check fixed positions in NAD lobe
        for j in range(228):
            if not designable_mask[j] and seq[j] != PROTEIN_SEQUENCE[j]:
                all_valid = False
                issues.append(f"Design {i+1}: Fixed position {j+1} mutated from {PROTEIN_SEQUENCE[j]} to {seq[j]}")

    if all_valid:
        mo.md("### ✓ All constraints satisfied\n\n- CYS260 preserved in all designs\n- Fixed NAD-contacting residues unchanged")
    else:
        mo.md(f"### ✗ Constraint violations\n\n" + "\n".join(f"- {issue}" for issue in issues[:10]))
    return


if __name__ == "__main__":
    app.run()
