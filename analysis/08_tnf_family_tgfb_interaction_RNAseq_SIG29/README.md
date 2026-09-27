# TNF-family x TGF-beta Interactions (SIG29, bulk RNA-seq)

Bulk RNA-seq of CD4 T cells stimulated with TGFb, with one of four TNF-family ligands
(`TNF`, `TL1A`, `OX40L` or the agonist antibody `DTA1`), or with TGFb plus each
TNF-family ligand. The question is whether the TGFb:TNF interaction seen in SIG18
(`../05_in_vitro_differentiation_RNAseq_SIG18`) is shared by the other TNF-family
members. This is tested at the gene level and by projecting the SIG18 sPCA gene programs
(dGEPs) onto the SIG29 samples.

| ligand1 | ligand2 | conditions |
|---|---|---|
| `none` | `none` | `none_none` (control) |
| `TGFb` | `none` | `TGFb_none` |
| `none` | TNF-family | `none_TNF`, `none_TL1A`, `none_OX40L`, `none_DTA1` |
| `TGFb` | TNF-family | `TGFb_TNF`, `TGFb_TL1A`, `TGFb_OX40L`, `TGFb_DTA1` |

Each combination is scored against its two single conditions, as in SIG18:

```text
lfc_interaction   = lfc_combo - lfc_TGFb - lfc_member
interaction_score = lfc_interaction / lfc_combo     (0 if padj_interaction > 0.1)
```

Genes or programs with `padj_interaction <= 0.1` are called `synergy positive` (score > 0,
total effect > 0), `synergy negative` (score > 0, total effect < 0) or `buffering`
(score < 0). All others are `none`.

## Folders

| folder | what it does |
|---|---|
| `01_deg` | DESeq2 condition effects and gene-level interaction scoring, and concordance of each TNF-family member's interaction with TGFb:TNF |
| `02_SIG18_dGEP_projection` | scores the SIG18 sPCA programs in each sample and calls program-level interactions with a linear model |

## Run order

```text
01_deg (01 ──> 02)
02_SIG18_dGEP_projection (01 ──> 02)      needs ../05_in_vitro_differentiation_RNAseq_SIG18/02_spca
```

Within each folder, run the files in numbered order. The two folders are independent of
each other. `02_SIG18_dGEP_projection` needs the SIG18 sPCA outputs. Later steps read the
stable copies of earlier outputs in `imports_stable/` (`SIG29/`, `SIG18/`), so each step can
also run on its own.

Each folder's README has its own steps, inputs and outputs.

## Environments

| folder | Python | R |
|---|---|---|
| 01 | | `R-signalseq` |
| 02 | `scanpy_standard2` | `R-signalseq` |

Environment files are in
`../../environments/`.

## Outputs

All outputs are written under `analysis_outs/08_tnf_family_tgfb_interaction_RNAseq_SIG29/` (not tracked). Inputs are read from
`imports_stable/`.
Each script finds the repo root by searching upward for `imports_stable/`, so run the
notebook and Rmds from inside the repo.
