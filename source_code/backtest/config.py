from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BacktestConfig:
    signal_path: str
    output_root: str = "./artifacts/backtests"
    signal_col: str = "score"
    start_date: str = ""
    end_date: str = ""
    horizon: int = 5
    cost_bps: float = 12.0
    groups: int = 10
    long_group: int = 0
    short_group: int = 0
    min_cross_section: int = 100
    only_tradable: bool = True
    benchmark_flag: str = ""
    use_adjusted_price: bool = True

    def to_dict(self) -> dict:
        data = asdict(self)
        data["output_root"] = str(Path(self.output_root).expanduser().resolve())
        data["signal_path"] = str(Path(self.signal_path).expanduser().resolve())
        return data
