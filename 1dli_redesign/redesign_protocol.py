import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

with app.setup:
    # GPU memory settings - must be before importing JAX
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
    os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Uncomment to use GPU 1 only

    import jax
    jax.config.update("jax_default_matmul_precision", "bfloat16")
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import gemmi
    from pathlib import Path
    from tempfile import NamedTemporaryFile

    from mosaic.optimizers import simplex_APGM, gradient_MCMC
    import mosaic.losses.structure_prediction as sp
    from mosaic.models.boltz2 import Boltz2
    from mosaic.losses.boltz2 import load_features_and_structure_writer
    from mosaic.losses.transformations import SetPositions, SoftClip
    from mosaic.losses.stability import StabilityModel
    from mosaic.losses.esmc import load_esmc
    from mosaic.common import TOKENS
    from mosaic.notebook_utils import pdb_viewer


@app.cell
def _():
    PROTEIN_SEQUENCE = (
        "MKIAVAGSGYVGLSLGVLLSLQNEVTIVDILPSKVDKINNGLSPIQDEYIEYYLKSKQLSIKATLDSKAAYKEAELVIIATPTNYNSRINYFDTQHVETVIKEVLSVNSHATLIIKSTIPIGFITEMRQKFQTDRIIFSPEFLRESKALYDNLYPSRIIVSCEENDSPKVKADAEKFALLLKSAAKKNNVPVLIMGASEAEAVKLFANTYLALRVAYFNELDTYAESRKLNSHMIIQGISYDDRIGMHYNNPSFGYGGYCLPKDTKQLLANYNNIPQTLIEAIVSSNNVRKSYIAKQIINVLKEQESPVKVVGVYRLIMKSNSDNFRESAIKDVIDILKSKDIKIIIYEPMLNKLESEDQSVLVNDLENFKKQANIIVTNRYDNELQDVKNKVYSRDIFGRD"
    )
    assert len(PROTEIN_SEQUENCE) == 402

    # Build wildtype_with_x: fixed for positions 0-227 and CYS260 (index 259), X elsewhere in 228-401
    wt_list = list(PROTEIN_SEQUENCE)
    for i in range(228, 402):
        if i != 259:  # preserve CYS260 (0-indexed: 259)
            wt_list[i] = "X"
    wildtype_with_x = "".join(wt_list)

    n_variable = wildtype_with_x.count("X")
    mo.md(
        f"**Protein length**: {len(PROTEIN_SEQUENCE)}  \n"
        f"**Variable positions**: {n_variable} (residues 229-402 minus CYS260)  \n"
        f"**Fixed positions**: {402 - n_variable}  \n"
        f"**CYS260 check**: position 259 is '{wildtype_with_x[259]}'"
    )
    return PROTEIN_SEQUENCE, n_variable, wildtype_with_x


@app.cell
def _():
    boltz2 = Boltz2()
    esmc = load_esmc("esmc_300m")
    stability_loss = StabilityModel.from_pretrained(esmc)
    return boltz2, stability_loss


@app.cell
def _():
    pdb_path = Path(__file__).parent / "1dli_g1p.pdb"
    st = gemmi.read_structure(str(pdb_path))

    # Extract chain A only for the template
    template_st = gemmi.Structure()
    template_model = gemmi.Model("0")
    chain_a = st[0].find_chain("A")

    new_chain = gemmi.Chain("A")
    for res in chain_a:
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
    import yaml as _yaml

    # Build YAML as a dict to avoid formatting issues
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
    asym_id = np.array(features["asym_id"]).squeeze()
    protein_len = 402

    # Tokens beyond the protein are ligands
    target_asym = asym_id[protein_len:]
    unique_target_asyms = np.unique(target_asym)

    mo.md(
        f"**Total tokens**: {len(asym_id)}  \n"
        f"**Protein tokens**: {protein_len}  \n"
        f"**Target tokens**: {len(target_asym)}  \n"
        f"**Unique target asym_ids**: {unique_target_asyms}"
    )

    # NAD is the first ligand (chain B), G1P is the second (chain C)
    nad_idx = [int(i) for i, a in enumerate(target_asym) if a == unique_target_asyms[0]]
    g1p_idx = [int(i) for i, a in enumerate(target_asym) if a == unique_target_asyms[-1]]

    mo.md(
        f"**NAD token indices** (relative to target): {nad_idx}  \n"
        f"**G1P token indices** (relative to target): {g1p_idx}"
    )
    return g1p_idx, nad_idx


@app.cell
def _(PROTEIN_SEQUENCE, boltz2, features):
    wt_pssm = jax.nn.one_hot(
        jnp.array([TOKENS.index(aa) for aa in PROTEIN_SEQUENCE]), 20
    )

    wt_output = boltz2.model_output(
        PSSM=wt_pssm,
        features=features,
        recycling_steps=3,
        key=jax.random.key(42),
    )

    # Slice to protein-only (402x402) since DistogramCE compares against binder_len
    wt_distogram_probs = jax.nn.softmax(wt_output.distogram_logits)[:402, :402]
    wt_plddt = wt_output.plddt

    mo.md(
        f"**Wildtype mean pLDDT**: {float(wt_plddt[:402].mean()):.3f}  \n"
        f"**Distogram shape**: {wt_distogram_probs.shape}"
    )
    return (wt_distogram_probs,)


@app.cell
def _(PROTEIN_SEQUENCE, boltz2, features, writer):
    wt_pred = boltz2.predict(
        PSSM=jax.nn.one_hot(
            jnp.array([TOKENS.index(aa) for aa in PROTEIN_SEQUENCE]), 20
        ),
        features=features,
        writer=writer,
        recycling_steps=3,
        key=jax.random.key(42),
    )
    mo.md(f"**Wildtype iPTM**: {float(wt_pred.iptm):.3f}")
    pdb_viewer(wt_pred.st)
    return


@app.cell
def _(
    boltz2,
    features,
    g1p_idx,
    nad_idx,
    stability_loss,
    wildtype_with_x,
    wt_distogram_probs,
):
    # Convert indices to numpy arrays for JAX compatibility
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
        + 1.0
        * sp.BinderTargetContact(
            epitope_idx=nad_idx_arr,
            paratope_idx=nad_lobe_idx,
            contact_distance=12.0,
        )
        # G1P binding (redesigned lobe must bind G1P)
        + 2.0
        * sp.BinderTargetContact(
            epitope_idx=g1p_idx_arr,
            paratope_idx=g1p_lobe_idx,
            contact_distance=12.0,
        )
        # Interface PAE
        + 0.1 * sp.BinderTargetPAE()
        + 0.1 * sp.TargetBinderPAE()
        # Scaffold adherence via distogram cross-entropy
        + 2.0
        * SoftClip(
            sp.DistogramCE(wt_distogram_probs, name="scaffold_CE"),
            2.5,
            3.0,
        )
    )

    boltz_loss = boltz2.build_loss(
        loss=inner_loss,
        features=features,
        recycling_steps=1,
        sampling_steps=25
    )

    # Stability loss (sequence-level, no structure prediction needed)
    stability_term = SoftClip(stability_loss, 2.5, 3.0)

    # Combine and restrict to variable positions
    design_loss = SetPositions.from_sequence(
        wildtype_with_x, boltz_loss + 0.5 * stability_term
    )

    jax.clear_caches()
    return (design_loss,)


@app.cell
def _(design_loss, n_variable):
    _, PSSM_soft = simplex_APGM(
        loss_function=design_loss,
        x=jax.nn.softmax(
            0.5
            * jax.random.gumbel(
                key=jax.random.key(np.random.randint(100000)),
                shape=(n_variable, 20),
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
def _(PSSM_sharp, design_loss):
    full_pssm = design_loss.sequence(PSSM_sharp)
    designed_tokens = full_pssm.argmax(-1)
    designed_sequence = "".join([TOKENS[int(i)] for i in designed_tokens])

    mo.md(f"**Designed sequence** ({len(designed_sequence)} residues):\n\n`{designed_sequence}`")
    return designed_sequence, designed_tokens, full_pssm


@app.cell
def _(boltz2, features, full_pssm, writer):
    designed_pred = boltz2.predict(
        PSSM=full_pssm,
        features=features,
        writer=writer,
        recycling_steps=3,
        key=jax.random.key(99),
    )
    mo.md(
        f"**Designed iPTM**: {float(designed_pred.iptm):.3f}  \n"
        f"**Mean pLDDT**: {float(designed_pred.plddt.mean()):.3f}"
    )
    pdb_viewer(designed_pred.st)
    return (designed_pred,)


@app.cell
def _(designed_pred, full_pssm):
    fig_pae = plt.figure(dpi=125)
    plt.imshow(designed_pred.pae)
    plt.title("PAE heatmap")
    plt.colorbar()

    fig_plddt = plt.figure(dpi=125)
    plt.plot(np.array(designed_pred.plddt))
    plt.axvline(x=228, color="red", linestyle="--", label="Redesign boundary")
    plt.title("pLDDT per residue")
    plt.xlabel("Residue index")
    plt.ylabel("pLDDT")
    plt.legend()

    fig_pssm = plt.figure(dpi=125)
    plt.imshow(full_pssm)
    plt.xlabel("Amino acid")
    plt.ylabel("Sequence position")
    plt.title("PSSM")

    mo.ui.tabs({"PAE": fig_pae, "pLDDT": fig_plddt, "PSSM": fig_pssm})
    return


@app.cell
def _(design_loss, designed_tokens):
    seq_mcmc = gradient_MCMC(
        design_loss,
        jax.device_put(designed_tokens),
        temp=0.001,
        proposal_temp=0.00001,
        steps=50,
        fix_loss_key=False,
        serial_evaluation=False,
    )
    return (seq_mcmc,)


@app.cell
def _(boltz2, design_loss, features, seq_mcmc, writer):
    mcmc_full_pssm = design_loss.sequence(jax.nn.one_hot(seq_mcmc, 20))
    mcmc_pred = boltz2.predict(
        PSSM=mcmc_full_pssm,
        features=features,
        writer=writer,
        recycling_steps=3,
        key=jax.random.key(100),
    )
    mo.md(
        f"**MCMC iPTM**: {float(mcmc_pred.iptm):.3f}  \n"
        f"**MCMC mean pLDDT**: {float(mcmc_pred.plddt.mean()):.3f}"
    )
    pdb_viewer(mcmc_pred.st)
    return (mcmc_pred,)


@app.cell
def _(design_loss, designed_pred, designed_sequence, mcmc_pred, seq_mcmc):
    mcmc_sequence = "".join(
        [TOKENS[int(i)] for i in design_loss.sequence(jax.nn.one_hot(seq_mcmc, 20)).argmax(-1)]
    )

    mo.md(f"**Designed sequence**:\n\n`{designed_sequence}`\n\n**MCMC-refined sequence**:\n\n`{mcmc_sequence}`")

    mo.hstack([
        mo.download(
            designed_pred.st.make_pdb_string(),
            filename="designed.pdb",
            label="Download designed PDB",
        ),
        mo.download(
            mcmc_pred.st.make_pdb_string(),
            filename="mcmc_refined.pdb",
            label="Download MCMC-refined PDB",
        ),
    ])
    return (mcmc_sequence,)


@app.cell
def _(PROTEIN_SEQUENCE, designed_sequence, mcmc_sequence):
    def validate(seq, label):
        # NAD lobe unchanged
        nad_lobe_match = seq[:228] == PROTEIN_SEQUENCE[:228]
        # CYS260 preserved
        cys260_ok = seq[259] == "C"
        # Count mutations in redesigned region
        mutations = sum(
            1 for i in range(228, 402) if i != 259 and seq[i] != PROTEIN_SEQUENCE[i]
        )
        return mo.md(
            f"### {label}\n"
            f"- NAD lobe (1-228) unchanged: **{nad_lobe_match}**\n"
            f"- CYS260 preserved: **{cys260_ok}**\n"
            f"- Mutations in redesigned region: **{mutations}** / 173\n"
        )

    mo.vstack([
        validate(designed_sequence, "Continuous optimization"),
        validate(mcmc_sequence, "MCMC-refined"),
    ])
    return


@app.cell
def _():
    0
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
