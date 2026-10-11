"""Smoke tests for the Streamlit dashboard using AppTest."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "app.py")


def test_app_smoke_default_load():
    """Verify default application startup raises no exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception


def test_app_smoke_run_algorithms():
    """Verify configuring algorithms and clicking run generates valid results dataframe."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    # Select Greedy and Exact search algorithms
    at.multiselect(key="algorithms").set_value(["Greedy (largest first)", "Exact search"])
    at.button(key="run").click().run()

    assert not at.exception

    # Assert that a dataframe containing both methods and 'valid' exists
    found_results_dataframe = False
    for df_elem in at.dataframe:
        val = df_elem.value
        if hasattr(val, "to_string"):
            table_text = val.to_string()
            has_greedy = "Greedy (largest first)" in table_text or "greedy-largest_first" in table_text
            has_exact = "Exact search" in table_text or "exact" in table_text
            has_valid = "valid" in table_text
            if has_greedy and has_exact and has_valid:
                found_results_dataframe = True
                break

    assert found_results_dataframe, (
        "Could not find results dataframe containing both 'Greedy (largest first)', "
        "'Exact search', and 'valid' status."
    )
