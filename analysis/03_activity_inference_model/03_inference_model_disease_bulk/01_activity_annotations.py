#!/usr/bin/env python
"""Step 01: build the ligand-activity annotation table used by the viz Rmds.

The model's activities are named after one representative ligand condition per
consensus cluster (e.g. `IL6_TNFSF18_c` stands for all ten IL6/IL21 x TNF-family
combinations that clustered together). Plots need a readable label for each, plus
the single/combinatorial split.

Display names come from `inference_model_activity_lookup.csv` (tracked next to
this script), which maps each in-code annotation to its manuscript label.

One fix is applied to `01_inference_model_construction_validation/01`'s cluster export on the way:
single-ligand rows keep the screen's linker scaffold in their names
(`IL4_linker_c`, `linker_IL2_c`), while the explanatory matrix has it stripped
(`IL4_c`, `IL2_c`) -- so the two files do not join as-is. The linker text is
removed here, and from the lookup's keys as well.

Output: `analysis_outs/03_activity_inference_model/inference_model_disease_bulk/activity_annotations.csv`
  annotation      activity name as it appears in the activity-score tables (join key)
  name            display label for plots
  activity_type   single | combinatorial
  members         the ligand conditions averaged into this activity, ";"-separated
"""
import os
import sys

import pandas as pd

# shared model module, in ../model_core
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model_core"))
import model_core as mc  # noqa: E402

# Hand-curated display names, keyed on the in-code annotation. Tracked in git
# alongside this script (not in analysis_outs): it is reference data rather than
# an output. Resolved relative to this file so the step runs from any cwd.
ACTIVITY_LOOKUP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "inference_model_activity_lookup.csv")
OUT_PATH = f"{mc.OUT_DIR}/activity_annotations.csv"


def load_name_map():
    """annotation (linker-stripped) -> manuscript display name."""
    lookup = pd.read_csv(ACTIVITY_LOOKUP)
    lookup["annotation_in_code"] = strip_linker(lookup["annotation_in_code"])
    name_map = (lookup
                .drop_duplicates(["annotation_in_code", "annotation_manuscript"])
                .set_index("annotation_in_code")["annotation_manuscript"])
    ambiguous = name_map.index[name_map.index.duplicated()].unique().tolist()
    if ambiguous:
        raise ValueError(f"lookup maps these annotations to >1 display name: {ambiguous}")
    return name_map


def strip_linker(s):
    """Remove the linker scaffold from ligand condition names: single-ligand
    conditions are `linker_X` / `X_linker`, combinations are `X_Y`."""
    return s.str.replace("linker_|_linker", "", regex=True)


def main():
    clusters = pd.read_csv(mc.ACTIVITY_CLUSTERS)
    clusters["activity"] = strip_linker(clusters["activity"])
    clusters["annotation"] = strip_linker(clusters["annotation"])

    # one row per activity, with its member ligand conditions
    annotations = (clusters
                   .drop_duplicates(["annotation", "activity"])
                   .groupby(["annotation", "activity_type"], as_index=False)
                   .agg(members=("activity", lambda x: ";".join(sorted(x)))))

    # display names from the curated lookup
    annotations["name"] = annotations["annotation"].map(load_name_map())
    missing = annotations.loc[annotations["name"].isna(), "annotation"].tolist()
    if missing:
        raise ValueError(f"activities missing from {ACTIVITY_LOOKUP}: {sorted(missing)}")

    # every activity in the explanatory matrix must get a label
    activities = set(mc.load_explanatory_matrix().columns)
    unlabelled = activities - set(annotations["annotation"])
    if unlabelled:
        raise ValueError(f"activities with no annotation row: {sorted(unlabelled)}")
    annotations = annotations[annotations["annotation"].isin(activities)]

    annotations = annotations[["annotation", "name", "activity_type", "members"]]
    os.makedirs(mc.OUT_DIR, exist_ok=True)
    annotations.to_csv(OUT_PATH, index=False)

    print(f"{len(annotations)} activities "
          f"({(annotations.activity_type == 'single').sum()} single, "
          f"{(annotations.activity_type == 'combinatorial').sum()} combinatorial)")
    print(f"wrote {OUT_PATH}")
    print(annotations.to_string(index=False))


if __name__ == "__main__":
    main()
