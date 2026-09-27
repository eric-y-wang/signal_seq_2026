# Arrayed CRISPR Validation Bulk RNA-seq (SIG30)

SIG30 is an arrayed bulk RNA-seq experiment in murine CD4 T cells. Each sample carries a
single CRISPR knockout of one of 9 target genes, or a control guide (30 samples, with a
`replicate` factor).

| pathway | KO targets |
|---|---|
| IL2 / STAT5 | `Stat5a`, `Stat5b`, `Il2ra` |
| TGFb | `Tgfbr2` |
| TNF | `Tnfrsf1a`, `Tnfrsf1b` |
| other | `Gata3`, `Ptpn1`, `Nr4a3` |

The analysis finds the genes each knockout changes, checks on-target knockdown, and scores
the SIG18 sparse-PCA programs (dGEPs) in each sample to see which TGFb- and TNF-driven
programs each knockout affects. Every model compares each target with `control` and
includes replicate as a covariate:

```text
DESeq2 (genes):     ~ target + replicate
lm (dGEP scores):   score ~ target + replicate
```

SIG30 has only single-gene knockouts, so there is no interaction scoring.

## Folders

| folder | what it does |
|---|---|
| `01_deg` | DESeq2 of each knockout vs. control, sample QC, and on-target knockdown checks |
| `02_SIG18_dGEP_projection` | scores the SIG18 sPCA programs in each sample and tests each knockout's effect on them |
| `03_program_crossreg` | compares ligand effects on receptor and signaling genes (SIG14, SIG18, SIG29) with the SIG30 knockout effects |

## Run order

```text
01_deg (01) ──┬──> 01_deg (02)
              └──> 03_program_crossreg

02_SIG18_dGEP_projection (01 ──> 02)      needs ../05_in_vitro_differentiation_RNAseq_SIG18/02_spca
```

`01_deg_analysis_SIG30.Rmd` writes `res_targets_SIG30.csv`, which the `01_deg`
visualization and `03_program_crossreg` read. `02_SIG18_dGEP_projection` does not depend on
`01_deg`, but needs the SIG18 sPCA outputs. `03_program_crossreg` also reads DEG tables from
the SIG14, SIG18 and SIG29 projects. Later steps read the stable copies of earlier outputs in
`imports_stable/` (`SIG30/`, `SIG18/`, `SIG29/`, `SIG14/`), so each step can run on its own.

Each folder's README has its own steps, inputs and outputs.

## Environments

| folder | Python | R |
|---|---|---|
| 01 | | `R-signalseq` |
| 02 | `scanpy_standard2` | `R-signalseq` |
| 03 | | `R-signalseq` |

Environment files are in
`../../environments/`.

## Outputs

All outputs are written under `analysis_outs/07_crispr_arrayed_validation_RNAseq_SIG30/` (not tracked). Inputs are read from
`imports_stable/`.
Each script finds the repo root by searching upward for `imports_stable/`, so run the
notebook and Rmds from inside the repo.
