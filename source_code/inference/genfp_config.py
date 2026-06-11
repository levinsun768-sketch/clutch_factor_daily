from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GenFingerprintConfig:
    model_run_dir: str = ""
    checkpoint_name: str = "best.pt"
    checkpoint_path: str = ""
    tensor_dir: str = ""
    start_date: str = ""
    end_date: str = ""
    batch_size: int = 4096
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_subdir: str = "fp_dataset"


default_genfp_config = GenFingerprintConfig()
