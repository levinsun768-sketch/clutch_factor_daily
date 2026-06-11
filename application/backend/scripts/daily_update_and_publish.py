from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DATE_FMT = "%Y%m%d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily research update, inference/backtest tasks, then publish product artifacts.")
    parser.add_argument("--research-workspace", default="../../source_code")
    parser.add_argument("--start-date", default="", help="Default: --end-date")
    parser.add_argument("--end-date", default=datetime.now().strftime(DATE_FMT))
    parser.add_argument("--skip-data-update", action="store_true")
    parser.add_argument("--refresh-reference", action="store_true")
    parser.add_argument("--build-features", action="store_true", default=True)
    parser.add_argument("--build-tensors", action="store_true")
    parser.add_argument("--window-size", type=int, default=40)

    parser.add_argument("--model-run-dir", default="artifacts/models/run_20260531_190247")
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--tensor-dir", default="", help="Optional tensor dataset dir for fingerprint generation.")
    parser.add_argument("--skip-fingerprint", action="store_true")
    parser.add_argument("--fingerprint-output-subdir", default="fp_dataset")
    parser.add_argument("--merge-fingerprint-dataset", action="store_true", help="Reserved hook: use an existing full fp_dataset unless you have merged incrementals yourself.")

    parser.add_argument("--downstream-run-dir", default="downstream/run/DailyFingerprintGRU_5d_20260531_204908")
    parser.add_argument("--skip-gru-inference", action="store_true")
    parser.add_argument("--prediction-output-name", default="", help="Default: prediction_scores_inference_<end_date>.parquet")

    parser.add_argument("--skip-barra", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--backtest-start-date", default="20250102")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--skip-neutralize", action="store_true")
    parser.add_argument("--neutral-start-date", default="20250102")
    parser.add_argument("--skip-portfolio", action="store_true")
    parser.add_argument("--portfolio-start-date", default="20250102")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--sell-rank", type=int, default=20)
    parser.add_argument("--max-industry-count", type=int, default=3)

    parser.add_argument("--publish-only", action="store_true", help="Only run publish_artifacts.py from existing research artifacts.")
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument("--skip-factor-layers", action="store_true", help="Skip publishing all-64 factor layered backtests.")
    parser.add_argument("--publish-universes", default="all,hs300,csi500,csi1000")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_research(path: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = (Path(__file__).resolve().parents[3] / raw).resolve()
    if repo_candidate.exists():
        return repo_candidate
    script_candidate = (Path(__file__).resolve().parents[1] / raw).resolve()
    return script_candidate


def run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("[daily]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def latest_file(root: Path, pattern: str) -> Path | None:
    files = list(root.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: (p.stat().st_mtime, p.name))


def main() -> None:
    args = parse_args()
    research = resolve_research(args.source_code)
    start_date = args.start_date or args.end_date
    prediction_name = args.prediction_output_name or f"prediction_scores_inference_{args.end_date}.parquet"

    if not args.publish_only:
        if not args.skip_data_update:
            cmd = [sys.executable, "-m", "data.update_pipeline", "--start-date", start_date, "--end-date", args.end_date, "--skip-existing"]
            if args.refresh_reference:
                cmd.append("--refresh-reference")
            if args.build_features:
                cmd.append("--build-features")
            if args.build_tensors:
                cmd += ["--build-tensors", "--window-size", str(args.window_size)]
            run(cmd, research, args.dry_run)

        if not args.skip_fingerprint:
            cmd = [
                sys.executable, "-m", "inference.generate_fingerprints",
                "--model-run-dir", args.model_run_dir,
                "--checkpoint-name", args.checkpoint_name,
                "--start-date", start_date,
                "--end-date", args.end_date,
                "--output-subdir", args.fingerprint_output_subdir,
            ]
            if args.tensor_dir:
                cmd += ["--tensor-dir", args.tensor_dir]
            run(cmd, research, args.dry_run)

        model_run = research / args.model_run_dir
        fp_file = latest_file(model_run / args.fingerprint_output_subdir, "fingerprints_daily_*.parquet")
        fp_file_name = fp_file.name if fp_file else ""

        if not args.skip_gru_inference:
            cmd = [
                sys.executable, "-m", "downstream.prediction.run_inference",
                "--downstream-run-dir", args.downstream_run_dir,
                "--start-date", start_date,
                "--end-date", args.end_date,
                "--output-name", prediction_name,
            ]
            if fp_file_name:
                cmd += ["--fingerprint-file-name", fp_file_name]
            run(cmd, research, args.dry_run)

        prediction_path = research / args.downstream_run_dir / prediction_name
        backtest_run = None
        if not args.skip_backtest:
            cmd = [
                sys.executable, "-m", "backtest.run_backtest",
                "--signal-path", str(prediction_path),
                "--start-date", args.backtest_start_date,
                "--end-date", args.end_date,
                "--horizon", str(args.horizon),
                "--groups", "10",
                "--min-cross-section", "100",
            ]
            run(cmd, research, args.dry_run)
            backtest_run = latest_file(research / "artifacts" / "backtests", "backtest_*/summary.json")
            backtest_run = backtest_run.parent if backtest_run else None

        if not args.skip_barra:
            run([sys.executable, "-m", "barra.build_exposures", "--start-date", start_date, "--end-date", args.end_date, "--skip-existing"], research, args.dry_run)

        neutral_signal = None
        if not args.skip_neutralize:
            labeled = backtest_run / "labeled_signals.parquet" if backtest_run else latest_file(research / "artifacts" / "backtests", "backtest_*/labeled_signals.parquet")
            if not labeled:
                raise FileNotFoundError("No labeled_signals.parquet found for neutralization.")
            cmd = [
                sys.executable, "-m", "barra.run_signal_neutral_ic",
                "--signal-path", str(prediction_path),
                "--start-date", args.neutral_start_date,
                "--end-date", args.end_date,
                "--labeled-signal-path", str(labeled),
                "--save-neutral-signal",
            ]
            run(cmd, research, args.dry_run)
            neutral_signal = research / "artifacts" / "barra" / "signal_neutral_ic" / f"{args.neutral_start_date}_{args.end_date}" / "neutral_signal.parquet"

        if not args.skip_portfolio:
            signal_for_portfolio = neutral_signal or latest_file(research / "artifacts" / "barra" / "signal_neutral_ic", "*/ewma/neutral_signal_ewma20.parquet")
            if not signal_for_portfolio:
                signal_for_portfolio = latest_file(research / "artifacts" / "barra" / "signal_neutral_ic", "*/neutral_signal.parquet")
            if not signal_for_portfolio:
                raise FileNotFoundError("No neutral signal found for portfolio construction.")
            cmd = [
                sys.executable, "-m", "portfolio.run_topn_portfolio",
                "--signal-path", str(signal_for_portfolio),
                "--start-date", args.portfolio_start_date,
                "--end-date", args.end_date,
                "--top-n", str(args.top_n),
                "--sell-rank", str(args.sell_rank),
                "--max-industry-count", str(args.max_industry_count),
            ]
            run(cmd, research, args.dry_run)

    if not args.skip_publish:
        script_dir = Path(__file__).resolve().parent
        publish_script = script_dir / "publish_artifacts.py"
        cmd = [
            sys.executable, str(publish_script),
            "--research-workspace", str(research),
            "--date", args.end_date,
            "--universes", args.publish_universes,
        ]
        run(cmd, research, args.dry_run)

        if not args.skip_factor_layers:
            layer_script = script_dir / "publish_factor_layers.py"
            cmd = [
                sys.executable, str(layer_script),
                "--research-workspace", str(research),
                "--start-date", args.backtest_start_date,
                "--end-date", args.end_date,
                "--horizon", str(args.horizon),
                "--universes", args.publish_universes,
            ]
            run(cmd, research, args.dry_run)


if __name__ == "__main__":
    main()
