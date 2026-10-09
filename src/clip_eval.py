#!/usr/bin/env python3
"""Image-text retrieval and modality-gap analysis for CLIP ViT-B/32 and ViT-L/14.

Core data: 7 images x 12 texts (+ 8 hard distractors for a separate expanded run).

Usage:
    python src/clip_eval.py                       # core pipeline, both models
    python src/clip_eval.py --dry-run             # random embeddings; code-path check only
    python src/clip_eval.py --offline             # locally cached models only
    python src/clip_eval.py --models clip-vit-base-patch32 --outdir results_b32

--dry-run uses random embeddings and must never be used for reported numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SEED = 42
PERPLEXITY = 5  # core set has 19 points (7 images + 12 texts); must stay below that
ROOT = Path(__file__).resolve().parents[1]  # repository root
log = logging.getLogger("clip_eval")
MODELS = {
    "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
    "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
}


# ----------------------------------------------------------------------------- data
def load_labels(path: Path | None = None) -> dict:
    path = path or ROOT / "data" / "labels.json"
    return json.loads(path.read_text(encoding="utf-8"))


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def build_valid_matrix(images, n_texts, valid_map) -> np.ndarray:
    valid = np.zeros((len(images), n_texts), dtype=bool)
    for i, name in enumerate(images):
        for j in valid_map[name]:
            valid[i, j] = True
    return valid


# ------------------------------------------------------------------------- backends
class Backend:
    """Wraps one model. dry=True returns deterministic random embeddings instead."""

    def __init__(self, tag: str, dry: bool = False, offline: bool = False):
        self.tag, self.dry = tag, dry
        self.hf_id = MODELS[tag]
        self.dim = 64
        if not dry:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self.torch = torch
            kw = {"local_files_only": offline}
            try:
                self.processor = CLIPProcessor.from_pretrained(self.hf_id, **kw)
                self.model = CLIPModel.from_pretrained(self.hf_id, **kw).eval()
            except OSError as e:  # hub unreachable, rate-limited, or not cached
                hint = ("not in the local cache (drop --offline to download it)" if offline
                        else "could not be downloaded (check the network, or pre-fetch it and rerun with --offline)")
                raise RuntimeError(f"Model {self.hf_id} {hint}: {e}") from e

    def _rng(self, salt: int) -> np.random.Generator:
        return np.random.default_rng((zlib.crc32(self.tag.encode()) + salt) % (2**32))

    def images(self, pil_images, batch_size: int = 32) -> np.ndarray:
        if self.dry:
            return self._rng(len(pil_images)).normal(size=(len(pil_images), self.dim))
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(pil_images), batch_size):
                inputs = self.processor(images=pil_images[i : i + batch_size], return_tensors="pt")
                out.append(self.model.get_image_features(pixel_values=inputs["pixel_values"]).cpu().numpy())
        return np.concatenate(out)

    def texts(self, texts, batch_size: int = 64) -> np.ndarray:
        if self.dry:
            return self._rng(1000 + len(texts)).normal(size=(len(texts), self.dim))
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                inputs = self.processor(text=texts[i : i + batch_size], return_tensors="pt", padding=True, truncation=True)
                feats = self.model.get_text_features(**{k: v for k, v in inputs.items() if k != "pixel_values"})
                out.append(feats.cpu().numpy())
        return np.concatenate(out)


# ------------------------------------------------------------------------ evaluation
def compute_metrics(similarity: np.ndarray, valid: np.ndarray) -> dict:
    """Top-1 and Mean Rank in both directions; queries without a valid match are excluded."""
    result = {}
    for direction, matrix, labels in (
        ("image_to_text", similarity, valid),
        ("text_to_image", similarity.T, valid.T),
    ):
        keep = labels.any(axis=1)
        ranks, hits = [], []
        for q in np.where(keep)[0]:
            order = np.argsort(-matrix[q], kind="stable")  # stable: ties favour lower index
            pos = np.flatnonzero(labels[q, order])
            ranks.append(int(pos[0]) + 1)
            hits.append(bool(pos[0] == 0))
        result[direction] = {
            "top1_accuracy": float(np.mean(hits)) if hits else None,
            "mean_rank": float(np.mean(ranks)) if ranks else None,
            "n_evaluated": int(keep.sum()),
            "n_excluded": int((~keep).sum()),
        }
    return result


def find_failures(sim, valid, images, texts) -> list[dict]:
    """Every real top-1 error, both directions (no cherry-picking)."""
    out = []
    for i, name in enumerate(images):
        j = int(np.argsort(-sim[i], kind="stable")[0])
        if not valid[i, j]:
            v = int(np.flatnonzero(valid[i])[np.argmax(sim[i][valid[i]])])
            out.append(dict(direction="image_to_text", image=name, predicted_text_id=j,
                            predicted_text=texts[j], predicted_score=float(sim[i, j]),
                            valid_text_id=v, valid_text=texts[v], valid_score=float(sim[i, v]),
                            margin=float(sim[i, j] - sim[i, v])))
    for j in range(sim.shape[1]):
        if not valid[:, j].any():
            continue
        i = int(np.argsort(-sim[:, j], kind="stable")[0])
        if not valid[i, j]:
            v = int(np.flatnonzero(valid[:, j])[np.argmax(sim[:, j][valid[:, j]])])
            out.append(dict(direction="text_to_image", text_id=j, text=texts[j],
                            predicted_image=images[i], predicted_score=float(sim[i, j]),
                            valid_image=images[v], valid_score=float(sim[v, j]),
                            margin=float(sim[i, j] - sim[v, j])))
    return out


def near_misses(sim, valid, images, texts) -> list[dict]:
    """Per image: best invalid text vs best valid text, sorted by smallest margin."""
    rows = []
    for i, name in enumerate(images):
        best_valid = sim[i][valid[i]].max()
        inv = np.flatnonzero(~valid[i])
        j = int(inv[np.argmax(sim[i][inv])])
        rows.append(dict(image=name, invalid_text_id=j, invalid_text=texts[j],
                         invalid_score=float(sim[i, j]), best_valid_score=float(best_valid),
                         margin=float(best_valid - sim[i, j])))
    return sorted(rows, key=lambda r: r["margin"])


# ------------------------------------------------------------------ gap / projections
def project_embeddings(emb: np.ndarray, perplexity: float = PERPLEXITY) -> dict:
    if len(emb) <= perplexity:
        raise ValueError("t-SNE perplexity must be below the number of points.")
    return {
        "PCA": PCA(n_components=2, random_state=SEED).fit_transform(emb),
        "t-SNE": TSNE(n_components=2, perplexity=perplexity, random_state=SEED, init="pca",
                      metric="cosine", learning_rate="auto").fit_transform(emb),
    }


def nearest_text_valid_rate(points: np.ndarray, n_images: int, valid: np.ndarray) -> float:
    """Fraction of images whose nearest text in 2D is a valid match."""
    img, txt = points[:n_images], points[n_images:]
    d = np.linalg.norm(img[:, None, :] - txt[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    return float(valid[np.arange(n_images), nearest].mean())


def gap_stats(img: np.ndarray, txt: np.ndarray, valid: np.ndarray) -> dict:
    """Modality-gap numbers from L2-normalised embeddings."""
    sim = img @ txt.T
    return {
        "centroid_distance": float(np.linalg.norm(img.mean(0) - txt.mean(0))),
        "mean_cosine_matched": float(sim[valid].mean()),
        "mean_cosine_unmatched": float(sim[~valid].mean()),
        "std_cosine_matched": float(sim[valid].std()),
        "std_cosine_unmatched": float(sim[~valid].std()),
    }


def plot_query_in_gap(query, query_kind, ref_img, ref_txt, path=None):
    """Place one query embedding in the PCA plane fitted on reference images + texts."""
    pca = PCA(n_components=2, random_state=SEED).fit(np.vstack([ref_img, ref_txt]))
    pi, pt, pq = pca.transform(ref_img), pca.transform(ref_txt), pca.transform(query.reshape(1, -1))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(*pi.T, c="tab:blue", marker="o", label="reference images", alpha=0.7)
    ax.scatter(*pt.T, c="tab:orange", marker="^", label="reference texts", alpha=0.7)
    ax.scatter(*pq.T, c="red", marker="*", s=250, label=f"your {query_kind} query")
    gap = np.linalg.norm(ref_img.mean(0) - ref_txt.mean(0))
    ax.set_title(f"PCA view (centroid gap = {gap:.3f}); 2D distances can mislead")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


# --------------------------------------------------------------------------- output
def write_similarity_csv(path, sim, image_names, n_texts):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image"] + [f"T{i}" for i in range(n_texts)])
        for i, n in enumerate(image_names):
            w.writerow([n] + [f"{v:.6f}" for v in sim[i]])


def plot_heatmap(sim, image_names, title, path):
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(sim, aspect="auto")
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xticks(range(sim.shape[1]))
    ax.set_xticklabels([f"T{i}" for i in range(sim.shape[1])], rotation=45, ha="right")
    ax.set_yticks(range(len(image_names)))
    ax.set_yticklabels(image_names)
    ax.set_xlabel("Text descriptions")
    ax.set_ylabel("Images")
    ax.set_title(title)
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_projection(points, n_images, labels, title, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(points[:n_images, 0], points[:n_images, 1], marker="o", s=70, label="Images")
    ax.scatter(points[n_images:, 0], points[n_images:, 1], marker="^", s=70, label="Texts")
    for k, (x, y) in enumerate(points):
        ax.annotate(labels[k], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_failure(f, tag, image_dir, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(Image.open(Path(image_dir) / f"{f['image']}.jpg").convert("RGB"))
    ax.axis("off")
    ax.set_title(f"{tag}: top-1 failure on image '{f['image']}'")
    fig.text(0.02, 0.02,
             f"Predicted T{f['predicted_text_id']}: {f['predicted_text']} ({f['predicted_score']:.3f})\n"
             f"Valid T{f['valid_text_id']}: {f['valid_text']} ({f['valid_score']:.3f}) | margin {f['margin']:.3f}",
             fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rankings(results, images, texts, out):
    lines = []
    for tag, r in results.items():
        s, v = r["similarity"], r["valid"]
        lines += [f"## {tag}", "### Image -> Text"]
        for i, n in enumerate(images):
            o = np.argsort(-s[i], kind="stable")
            lines.append(f"{n}: " + " > ".join(f"T{j}{'*' if v[i, j] else ''}({s[i, j]:.3f})" for j in o))
        lines.append("### Text -> Image")
        for j in range(len(texts)):
            o = np.argsort(-s[:, j], kind="stable")
            lines.append(f"T{j} ({texts[j]}): " + " > ".join(f"{images[i]}{'*' if v[i, j] else ''}({s[i, j]:.3f})" for i in o))
        lines.append("")
    lines.append("* = valid image-text match")
    (out / "rankings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------- driver
def run_model(tag, backend, images, pil, valid, texts, hard, out, image_dir):
    img = l2_normalize(backend.images(pil))
    txt = l2_normalize(backend.texts(texts))
    sim = img @ txt.T
    pts = project_embeddings(np.vstack([img, txt]))
    labels = images + [f"T{i}" for i in range(len(texts))]
    for method, p in pts.items():
        extra = f", perplexity={PERPLEXITY}" if method == "t-SNE" else ""
        plot_projection(p, len(images), labels, f"{tag}: {method} (seed={SEED}{extra})", out / f"{tag}_{method}.png")
    plot_heatmap(sim, images, f"{tag}: Image x Text cosine similarity", out / f"{tag}_matrix.png")
    write_similarity_csv(out / f"{tag}_similarity.csv", sim, images, len(texts))

    ii, tt = img @ img.T, txt @ txt.T
    iu, tu = np.triu_indices(len(images), 1), np.triu_indices(len(texts), 1)
    stats = {
        **gap_stats(img, txt, valid),
        "n_projected_points": int(len(img) + len(txt)),
        "tsne_perplexity": PERPLEXITY,
        "seed": SEED,
        "pca_nearest_text_valid_rate": nearest_text_valid_rate(pts["PCA"], len(images), valid),
        "tsne_nearest_text_valid_rate": nearest_text_valid_rate(pts["t-SNE"], len(images), valid),
        "mean_image_image_cosine": float(ii[iu].mean()),
        "mean_text_text_cosine": float(tt[tu].mean()),
    }
    base = {
        "metrics": compute_metrics(sim, valid),
        "stats": stats,
        "failures": find_failures(sim, valid, images, texts),
        "near_misses": near_misses(sim, valid, images, texts)[:3],
    }
    # expanded run (hard distractors with no valid image), always reported separately
    all_texts = texts + hard
    ev = np.zeros((len(images), len(all_texts)), dtype=bool)
    ev[:, : len(texts)] = valid
    esim = img @ l2_normalize(backend.texts(all_texts)).T
    expanded = {
        "metrics": compute_metrics(esim, ev),
        "failures": find_failures(esim, ev, images, all_texts),
        "near_misses": near_misses(esim, ev, images, all_texts)[:3],
    }
    fl = base["failures"] or expanded["failures"]
    first = next((f for f in fl if f["direction"] == "image_to_text"), None)
    if first:
        plot_failure(first, tag, image_dir, out / f"{tag}_failure_case.png")
    return dict(similarity=sim, valid=valid, base=base, expanded=expanded, img=img, txt=txt, pts=pts)


def write_summary(results, out):
    def pct(x):
        return f"{100 * x:.1f}%"

    L = ["# Summary", "", f"Seed {SEED}; t-SNE perplexity {PERPLEXITY}.", "",
         "## Core retrieval (7 images x 12 texts)", "",
         "| Model | I2T Top-1 | I2T MeanRank | T2I Top-1 | T2I MeanRank | Excl. I2T/T2I |", "|---|---:|---:|---:|---:|---:|"]
    for t, r in results.items():
        a, b = r["base"]["metrics"]["image_to_text"], r["base"]["metrics"]["text_to_image"]
        L.append(f"| {t} | {pct(a['top1_accuracy'])} | {a['mean_rank']:.2f} | {pct(b['top1_accuracy'])} | {b['mean_rank']:.2f} | {a['n_excluded']}/{b['n_excluded']} |")
    L += ["", "## Expanded set (+hard distractors), reported separately", "",
          "| Model | I2T Top-1 | I2T MeanRank | T2I Top-1 | T2I MeanRank | # real top-1 errors |", "|---|---:|---:|---:|---:|---:|"]
    for t, r in results.items():
        a, b = r["expanded"]["metrics"]["image_to_text"], r["expanded"]["metrics"]["text_to_image"]
        L.append(f"| {t} | {pct(a['top1_accuracy'])} | {a['mean_rank']:.2f} | {pct(b['top1_accuracy'])} | {b['mean_rank']:.2f} | {len(r['expanded']['failures'])} |")
    L += ["", "## Modality gap, core set", "",
          "| Model | Centroid dist. | Cos matched | Cos unmatched | 2D nearest-valid PCA | 2D nearest-valid t-SNE | Retrieval Top-1 (I2T) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for t, r in results.items():
        s = r["base"]["stats"]
        L.append(f"| {t} | {s['centroid_distance']:.3f} | {s['mean_cosine_matched']:.3f} | {s['mean_cosine_unmatched']:.3f} | "
                 f"{pct(s['pca_nearest_text_valid_rate'])} | {pct(s['tsne_nearest_text_valid_rate'])} | {pct(r['base']['metrics']['image_to_text']['top1_accuracy'])} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--dry-run", action="store_true", help="random embeddings; code-path check only")
    ap.add_argument("--offline", action="store_true", help="use only locally cached models (HF_HOME)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plt.switch_backend("Agg")  # scripts render to files; importing this module leaves the backend alone
    seed_everything()
    out = Path(args.outdir or ("results_dry" if args.dry_run else "results"))
    out.mkdir(parents=True, exist_ok=True)
    lab = load_labels()
    images, texts, hard = lab["images"], lab["texts"], lab["hard_distractors"]
    image_dir = ROOT / "data" / "images"
    pil = [Image.open(image_dir / f"{n}.jpg").convert("RGB") for n in images]
    valid = build_valid_matrix(images, len(texts), lab["valid"])

    results = {}
    for tag in args.models:
        log.info("[%s] core evaluation", tag)
        be = Backend(tag, dry=args.dry_run, offline=args.offline)
        results[tag] = run_model(tag, be, images, pil, valid, texts, hard, out, image_dir)
        del be

    save_rankings(results, images, texts, out)
    ser = {t: {"base": r["base"], "expanded": r["expanded"]} for t, r in results.items()}
    (out / "evaluation.json").write_text(json.dumps(ser, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "expanded_evaluation.json").write_text(
        json.dumps({t: r["expanded"] for t, r in results.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(results, out)
    log.info("Done. Results in %s%s", out.resolve(), "  (DRY RUN: not valid results)" if args.dry_run else "")


if __name__ == "__main__":
    main()
