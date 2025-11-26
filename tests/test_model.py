"""Tests for core model functionality."""

import pytest
import numpy as np
from erucakra import ClimateModel, SCENARIOS, get_scenario


class TestClimateModel:
    def test_initialization(self):
        model = ClimateModel()
        assert model.params["c"] == 0.2
        assert model.params["epsilon"] == 0.02
    
    def test_run_with_scenario(self):
        model = ClimateModel()
        results = model.run(scenario="ssp245", n_points=100, show_progress=False)
        assert len(results.t) == 100


class TestScenarios:
    def test_all_scenarios_exist(self):
        expected = ["ssp126", "ssp245", "ssp370", "ssp585", "overshoot"]
        for key in expected:
            assert key in SCENARIOS
