"""
stage0.py

Shared utility module for file output and logging.

Features:
- Prepare result directories (txt / csv / visual)
- Log messages to both the console and a text file
- Save matplotlib figures
- Save DataFrames as CSV files
- Save Python objects as pickle files
- Save dictionaries as JSON files
"""

import os
import sys
import json
import pickle

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Allow figures to be saved in headless environments
import matplotlib.pyplot as plt


def prepare_result_dirs(base_dir: str) -> dict:
    """Create txt, csv, and visual subdirectories under base_dir and return their paths."""

    paths = {
        "root": base_dir,
        "txt": os.path.join(base_dir, "txt"),
        "csv": os.path.join(base_dir, "csv"),
        "visual": os.path.join(base_dir, "visual"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


class ReportLogger:
    """Write messages to both the console and a text file."""


    def __init__(self, txt_path: str):
        self.txt_path = txt_path
        self._buffer = []

    def log(self, *args, sep=" ", end="\n"):
        line = sep.join(str(a) for a in args) + end
        sys.stdout.write(line)
        self._buffer.append(line)

    def section(self, title: str):
        bar = "=" * 70
        self.log("\n" + bar)
        self.log(title)
        self.log(bar)

    def save(self):
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.writelines(self._buffer)


def save_fig(fig_or_plt, visual_dir: str, filename: str, dpi: int = 120):
    """Save the current figure or the provided figure object in the visual directory."""

    path = os.path.join(visual_dir, filename)
    if hasattr(fig_or_plt, "savefig"):
        fig_or_plt.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    return path


def save_dataframe(df, csv_dir: str, filename: str):
    path = os.path.join(csv_dir, filename)
    df.to_csv(path, index=False)
    return path


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path
