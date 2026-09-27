# Receptor Cross-Regulation: Ligand Effects vs. SIG30 Knockouts

Compares how ligands change the expression of receptor and signaling genes with how the
SIG30 knockouts change the same genes. This shows, for example, whether a ligand induces
the receptor of another ligand, and whether knocking out one pathway component changes the
others.

## Pipeline

- **`Th_receptor_crossregulation.Rmd`**: LFC heatmaps with significance stars.
  - Receptor induction by single ligands at 6 h (SIG14) and 96 h (SIG18), for receptors
    with `padj < 0.1` in at least one condition.
  - The CRISPR target genes (plus `Il2rb` and `Tgfbr1`) under SIG29 ligands (`TGFb`, `TNF`,
    `TGFb-TNF`, the SIG29 `TGFb:TNF` interaction term and the TL1A/OX40L/DTA1 arms), SIG14
    ligands, and each SIG30 knockout. SIG29 was sequenced in the same batch as SIG30.
  - The same comparison for all `Ptpn*` and `Nr4a*` family genes.

## Inputs

- `imports_stable/SIG30/analysis_outs/res_targets_SIG30.csv`: SIG30 knockout DESeq2 results
  from `../01_deg` (stable copy).
- DEG tables from the SIG14, SIG18 and SIG29 projects, read from `imports_stable/`:
  `SIG14/analysis_outs/deg_updated/res_conditions_SIG14.csv`,
  `SIG18/analysis_outs/deg/res_conditions_SIG18.csv`,
  `SIG18/analysis_outs/deg/res_interaction_scored_SIG18.csv`,
  `SIG29/analysis_outs/res_conditions_SIG29.csv` and
  `SIG29/analysis_outs/res_interaction_scored_SIG29.csv`. The SIG18 and SIG29 tables are
  stable copies of the outputs of `../../05_in_vitro_differentiation_RNAseq_SIG18/01_deg` and
  `../../08_tnf_family_tgfb_interaction_RNAseq_SIG29/01_deg`; the SIG14 table comes from a
  project that is not in this repo.

## Outputs

Written to `analysis_outs/07_crispr_arrayed_validation_RNAseq_SIG30/program_crossreg/`, which is gitignored:
`crossreg_*.pdf` heatmaps.

## Running

Knit in the `R-signalseq` env. It reads the stable copy of
`../01_deg`'s output, so it does not need `01_deg` to be re-run first. It creates its output
folder if needed.
