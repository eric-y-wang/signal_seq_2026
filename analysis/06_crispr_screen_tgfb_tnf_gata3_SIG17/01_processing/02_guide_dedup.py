#!/usr/bin/env python
"""
Per-sample UMI deduplication for SIG17 CRISPR screen sgRNA-seq.

Pipeline (see 01_umi_extract.sh for the preceding step):
  1. Input is the R1 fastq already UMI-tagged by `umi_tools extract`
     (read name suffix "_<20bp UMI>").
  2. cutadapt isolates the 20bp guide sequence using the constant vector
     flanks observed in R1 (ACACCG <guide> GTTTAAGAGC). Reads where the
     linked adapter isn't found are NOT discarded (--discard-untrimmed is
     intentionally omitted) so record order/count stays 1:1 with the input
     fastq, and are instead identified downstream by trimmed length > 25bp.
  3. The trimmed 20bp sequence is matched against the library (exact, with
     an unambiguous single-mismatch fallback).
  4. Reads are grouped by guide, and umi_tools' own directional-adjacency
     UMIClusterer collapses UMIs that likely differ only by sequencing
     error, merging PCR/sequencing duplicates of the same original molecule.
  5. One representative full-length read is written per surviving
     (guide, UMI-cluster) pair, to feed into `mageck count` unchanged.
"""
import argparse
import csv
import gzip
import json
import subprocess
import sys
from collections import Counter, defaultdict

from umi_tools.network import UMIClusterer

BASES = "ACGT"
ADAPTER = "ACACCG...GTTTAAGAGC"
MAX_GUIDE_TRIM_LEN = 25  # successful trims cluster at 19-21bp; failed passthroughs stay ~151bp
UMI_SEPARATOR = "_"


def load_library(path):
    """Return (exact_dict, mismatch_dict, gene_dict).

    exact_dict / mismatch_dict map 20bp guide seq -> sgRNA_id. mismatch_dict
    additionally covers every single-substitution neighbor of each guide,
    dropping any neighbor sequence that collides between two distinct real
    guides (kept ambiguous -> excluded) to avoid mis-assignment. gene_dict
    maps sgRNA_id -> gene.
    """
    exact = {}
    gene_of = {}
    with open(path) as f:
        for row in csv.reader(f):
            sgrna_id, seq, gene = row
            seq = seq.strip().upper()
            exact[seq] = sgrna_id
            gene_of[sgrna_id] = gene

    neighbor_owner = {}
    ambiguous = set()
    for seq, sgrna_id in exact.items():
        for i in range(len(seq)):
            for b in BASES:
                if b == seq[i]:
                    continue
                variant = seq[:i] + b + seq[i + 1:]
                if variant in exact:
                    continue  # already a real guide sequence, not a fallback case
                owner = neighbor_owner.get(variant)
                if owner is None:
                    neighbor_owner[variant] = sgrna_id
                elif owner != sgrna_id:
                    ambiguous.add(variant)

    mismatch = {v: sid for v, sid in neighbor_owner.items() if v not in ambiguous}
    return exact, mismatch, gene_of


def run_cutadapt(in_fastq, out_fastq, log_path, threads):
    cmd = [
        "cutadapt",
        "-g", ADAPTER,
        "-e", "0.15",
        "-j", str(threads),
        "-o", out_fastq,
        in_fastq,
    ]
    with open(log_path, "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def fastq_records(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        while True:
            header = f.readline()
            if not header:
                return
            seq = f.readline().rstrip("\n")
            plus = f.readline().rstrip("\n")
            qual = f.readline().rstrip("\n")
            yield header.rstrip("\n"), seq, plus, qual


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--umi-fastq", required=True, help="UMI-tagged R1 fastq(.gz) from umi_tools extract")
    ap.add_argument("--library", required=True, help="mageck_library.csv")
    ap.add_argument("--outdir", required=True, help="directory for dedup fastq output")
    ap.add_argument("--statsdir", required=True, help="directory for per-sample QC stats")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    exact_lib, mismatch_lib, gene_of = load_library(args.library)
    print(f"[{args.sample}] library: {len(exact_lib)} guides, "
          f"{len(mismatch_lib)} unambiguous 1-mismatch neighbors", file=sys.stderr)

    trimmed_fastq = f"{args.outdir}/{args.sample}_R1.trimmed_guide.fastq.gz"
    cutadapt_log = f"{args.statsdir}/{args.sample}.cutadapt.log"
    run_cutadapt(args.umi_fastq, trimmed_fastq, cutadapt_log, args.threads)

    # guide_id -> umi(str) -> [count, representative_record]
    # representative_record kept as (header, seq, plus, qual) of the first read seen with that exact umi
    guides = defaultdict(lambda: defaultdict(lambda: [0, None]))

    stats = Counter()
    for (orig_h, orig_seq, orig_plus, orig_qual), (trim_h, trim_seq, _, _) in zip(
        fastq_records(args.umi_fastq), fastq_records(trimmed_fastq)
    ):
        stats["total_reads"] += 1
        if len(trim_seq) > MAX_GUIDE_TRIM_LEN:
            continue  # linked adapter not found in this read
        stats["adapter_trimmed"] += 1

        guide_seq = trim_seq.upper()
        sgrna_id = exact_lib.get(guide_seq) or mismatch_lib.get(guide_seq)
        if sgrna_id is None:
            continue
        stats["guide_matched"] += 1

        read_id = orig_h.split(" ", 1)[0][1:]  # drop leading '@', trailing read2 metadata
        try:
            umi = read_id.rsplit(UMI_SEPARATOR, 1)[1]
        except IndexError:
            continue

        entry = guides[sgrna_id][umi]
        entry[0] += 1
        if entry[1] is None:
            entry[1] = (orig_h, orig_seq, orig_plus, orig_qual)

    clusterer = UMIClusterer(cluster_method="directional")
    dedup_out = f"{args.outdir}/{args.sample}_R1.dedup.fastq.gz"
    per_guide_rows = []
    clone_size_hist = Counter()
    n_guides_with_reads = 0
    total_unique_umis_raw = 0
    with gzip.open(dedup_out, "wt") as out:
        for sgrna_id, umi_counts in guides.items():
            n_guides_with_reads += 1
            raw_umis = len(umi_counts)
            total_unique_umis_raw += raw_umis

            bundle = {umi.encode(): count for umi, [count, _rec] in umi_counts.items()}
            clusters = clusterer(bundle, threshold=1)

            clone_sizes = []
            for cluster in clusters:
                rep_umi = cluster[0].decode()
                header, seq, plus, qual = umi_counts[rep_umi][1]
                out.write(f"{header}\n{seq}\n{plus}\n{qual}\n")
                stats["dedup_reads"] += 1

                clone_size = sum(umi_counts[u.decode()][0] for u in cluster)
                clone_sizes.append(clone_size)
                clone_size_hist[clone_size] += 1

            per_guide_rows.append({
                "sgRNA_id": sgrna_id,
                "gene": gene_of.get(sgrna_id, ""),
                "reads_matched": sum(c for c, _r in umi_counts.values()),
                "unique_umis_raw": raw_umis,
                "umi_clusters": len(clusters),
                "mean_clone_size": sum(clone_sizes) / len(clone_sizes) if clone_sizes else 0.0,
                "max_clone_size": max(clone_sizes) if clone_sizes else 0,
            })

    per_guide_path = f"{args.statsdir}/{args.sample}.umi_per_guide.tsv"
    with open(per_guide_path, "w") as f:
        cols = ["sgRNA_id", "gene", "reads_matched", "unique_umis_raw",
                "umi_clusters", "mean_clone_size", "max_clone_size"]
        f.write("\t".join(cols) + "\n")
        for row in per_guide_rows:
            f.write("\t".join(str(row[c]) for c in cols) + "\n")

    all_clone_sizes = sorted(
        size for size, n in clone_size_hist.items() for _ in range(n)
    )
    n_clusters = len(all_clone_sizes)
    stats["unique_guides_detected"] = n_guides_with_reads
    stats["total_unique_umis_raw"] = total_unique_umis_raw
    stats["total_umi_clusters"] = n_clusters  # == dedup_reads, i.e. distinct original molecules
    stats["duplication_rate"] = (
        1 - stats["dedup_reads"] / stats["guide_matched"] if stats["guide_matched"] else 0.0
    )
    stats["clone_size_mean"] = sum(all_clone_sizes) / n_clusters if n_clusters else 0.0
    stats["clone_size_median"] = (
        all_clone_sizes[n_clusters // 2] if n_clusters else 0
    )
    stats["clone_size_singletons"] = clone_size_hist.get(1, 0)
    stats["clone_size_histogram"] = {str(k): v for k, v in sorted(clone_size_hist.items())}

    stats_path = f"{args.statsdir}/{args.sample}.dedup_stats.json"
    with open(stats_path, "w") as f:
        json.dump(dict(stats), f, indent=2)

    print(f"[{args.sample}] {dict(stats)}", file=sys.stderr)


if __name__ == "__main__":
    main()
