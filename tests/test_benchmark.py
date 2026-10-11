"""Tests for the benchmarking suite, reporting, plots, and resume mechanism."""

import json
import pytest

from exam_scheduler.benchmark import BENCHMARK_COLUMNS, BenchmarkConfig, run_benchmark
from exam_scheduler.data import ring_instance
from exam_scheduler.plots import generate_all_plots
from exam_scheduler.validator import schedule_objective, validate_schedule


def test_run_benchmark_fast(tmp_path):
    """Test full benchmark execution and output validation on a fast small instance."""
    inst = ring_instance(3, seed=42)
    cfg = BenchmarkConfig(
        n_slots=3,
        random_seeds=3,
        p1_seeds=1,
        include_p2=False,
        penalty_seeds=1,
        scaling_seeds=1,
        shots=256,
        restarts=1,
        maxiter=15,
        penalties=(1.0, 1.5),
        scaling_sizes=(3,),
    )

    df = run_benchmark(cfg, out_dir=tmp_path, headline_instance=inst)

    # 1. Check all required output files exist
    runs_csv = tmp_path / "benchmark_runs.csv"
    summary_csv = tmp_path / "benchmark_summary.csv"
    summary_md = tmp_path / "benchmark_summary.md"
    meta_json = tmp_path / "benchmark_meta.json"

    assert runs_csv.exists()
    assert summary_csv.exists()
    assert summary_md.exists()
    assert meta_json.exists()

    # 2. Check all required columns are present
    for col in BENCHMARK_COLUMNS:
        assert col in df.columns

    # 3. Every row with is_valid True re-validates and objective matches
    for _, row in df[df["is_valid"] == True].iterrows():
        assign_str = str(row["assignment"]).strip()
        assignment = [int(tok) for tok in assign_str.split()]
        target_inst = (
            inst
            if row["instance"].startswith("inst_")
            else ring_instance(int(row["instance"].split("_")[1]), seed=cfg.base_seed)
        )
        val = validate_schedule(target_inst, assignment, int(row["n_slots"]))
        assert val.is_valid is True
        calc_obj = schedule_objective(target_inst, assignment)
        assert row["objective"] == pytest.approx(calc_obj, abs=1e-6)

    # 4. Exact optimum identical across E1 rows
    e1_rows = df[df["experiment"] == "E1_headline"]
    e1_opts = e1_rows["optimal_objective"].dropna().unique()
    assert len(e1_opts) == 1

    # 5. No valid row has optimality_gap < -1e-9
    valid_rows = df[df["is_valid"] == True]
    assert (valid_rows["optimality_gap"] >= -1e-9).all()

    # 6. Running again with resume=True adds no duplicate rows
    initial_len = len(df)
    df_resumed = run_benchmark(cfg, out_dir=tmp_path, headline_instance=inst, resume=True)
    assert len(df_resumed) == initial_len

    # 7. Plot functions create non-empty PNG files
    generate_all_plots(df, tmp_path)
    for plot_name in [
        "fig_feasibility.png",
        "fig_objective.png",
        "fig_penalty.png",
        "fig_scaling.png",
    ]:
        plot_file = tmp_path / plot_name
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

    # 8. benchmark_meta.json contains qiskit version and no-advantage sentence
    meta_content = meta_json.read_text(encoding="utf-8")
    meta_data = json.loads(meta_content)
    assert "qiskit" in meta_data
    assert "No quantum advantage is claimed; all quantum runs are noiseless classical simulations." in meta_content
