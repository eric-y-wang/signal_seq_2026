# Shared Inference Model Code

The ligand-activity inference model, shared by the analyses in this section. It
lives outside the numbered folders because several of them import it.

| file | what it is | used by |
|---|---|---|
| `model_core.py` | bulk (per-sample) model: net construction, component scoring, target matrix, ridge + permutation null | `02_inference_model_mixture_validation`, `03_inference_model_disease_bulk` |
| `model_core_sc.py` | single-cell GPU port of the same model, with a wider alpha grid | `04_inference_model_disease_sc` |
| `mouse_human_ortholog_map.csv` | pinned mouse->human ortholog map, read by both modules | both |

The model is `ridge_zscore_50_weighted`, selected in
`../01_inference_model_construction_validation`.

The two modules share no code, so the single-cell alpha grid
(`logspace(-1, 5, 200)`) can differ from the bulk one without changing bulk
results. Both read the same net, explanatory matrix and ortholog map.

## Importing

Scripts add this folder to `sys.path` relative to their own location. Notebooks add
`<repo>/analysis/03_activity_inference_model/model_core` using the repo root they find:

```python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model_core"))
import model_core as mc
```

Both modules find the repo root from their own location, by searching upward for
`imports_stable/`. They read the net inputs from `imports_stable/SIG13/analysis_outs/`
(`SPCA_DIR` = `spca/`, `MODEL_DIR` = `inference_model_final/`). `OUT_DIR` is
`analysis_outs/03_activity_inference_model/inference_model_disease_bulk/` (bulk) or
`.../inference_model_disease_sc/` (single cell). The ortholog map stays in this folder.

The ortholog map pins the gene set the model scores on. To refresh it from Ensembl
BioMart, delete the file. The next time either module maps mouse genes to human,
it will query BioMart and write a new map.
