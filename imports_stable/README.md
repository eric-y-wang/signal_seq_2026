# imports_stable

Frozen copies of every input file the analysis scripts read. Scripts read from here and write to `analysis_outs/<module>/`.

This includes both inputs generated in this study (Cell Ranger outputs, bulk RNA-seq count matrices, ATAC-seq peak data) and files produced by earlier steps of this repo's pipelines. The earlier-step outputs are included so most scripts can be run on their own against the exact files used in the manuscript; the files that are not included are listed under [Not included](#not-included). Re-running an upstream step writes a fresh copy to `analysis_outs/` and does not change `imports_stable/`.

The data is distributed as a single Zenodo deposit (`https://doi.org/10.5281/zenodo.22997145`). Certain files are not available in Zenodo but can be downloaded from GEO as described in the Zenodo repository.

The full imports_stable folder can be created as described in the Zenodo repository. 




