"""
Tests for the BaseMixin class
in the ensemble module.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from aqua.diagnostics.ensemble.base import(
    BaseMixin)

pytestmark = [pytest.mark.diagnostics, pytest.mark.ensemble]


class TestBaseMixin:
    """
    Test suite for the BaseMixin class.
    """

    def test_initialization_single(self):
        """
        Test BaseMixin initialization
        with single-item lists.
        """
        base = BaseMixin(
            diagnostic_name="test_diag",
            diagnostic_product="test_prod",
            catalog_list=["catalogA"],
            model_list=["modelA"],
            exp_list=["expA"],
            source_list=["sourceA"],
        )
        assert base.catalog == "catalogA"
        assert base.model == "modelA"
        assert base.exp == "expA"
        assert base.source == "sourceA"
        assert base.diagnostic_name == "test_diag"
        assert base.diagnostic_product == "test_prod"

    def test_initialization_none(self):
        """
        Test BaseMixin initialization
        when lists are None.
        """
        base = BaseMixin()
        assert base.catalog == "ensemble_catalog"
        assert base.model == "ensemble_model"
        assert base.exp == "ensemble_exp"
        assert base.source == "ensemble_source"

    def test_initialization_multi(self):
        """
        Test BaseMixin initialization
        with multi-item lists.
        """
        base = BaseMixin(
            catalog_list=["cat1", "cat2"],
            model_list=["mod1", "mod2"],
            exp_list=["exp1", "exp2"],
            source_list=["src1", "src2"],
        )
        assert base.catalog == "multi_catalog"
        assert base.model == "multi_model"
        assert base.exp == "multi_exp"
        assert base.source == "multi_source"

    def test_save_netcdf(self, tmp_path):
        """
        Test that BaseMixin successfully
        hands off to OutputSaver to save a NetCDF.
        """
        # Create a dummy DataArray
        data = xr.DataArray(
            np.random.rand(10, 10),
            dims=["lat", "lon"],
            coords={"lat": np.linspace(-90, 90, 10), "lon": np.linspace(0, 360, 10)},
            name="dummy_var",
        )

        base = BaseMixin(
            diagnostic_name="ensemble",
            diagnostic_product="test_prod",
            catalog_list=["cat1"],
            model_list=["mod1"],
            exp_list=["exp1"],
            outputdir=str(tmp_path),
        )

        # Execute save
        base.save_netcdf(
            var="dummy_var",
            data_name="mean",
            data=data,
            description="Test save netcdf",
            startdate="2000-01-01",
            enddate="2000-12-31",
        )

        # OutputSaver creates a "netcdf" subfolder in the outputdir
        nc_dir = tmp_path / "netcdf"
        assert nc_dir.exists()

        # Check if any .nc file was created
        nc_files = list(nc_dir.glob("*.nc"))
        assert len(nc_files) == 1
        assert "dummy_var" in nc_files[0].name
        assert "mean" in nc_files[0].name

    def test_save_figure(self, tmp_path):
        """
        Test that BaseMixin successfully
        hands off to OutputSaver to save a figure.
        """
        # Create a dummy figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        base = BaseMixin(
            diagnostic_name="ensemble",
            diagnostic_product="test_prod",
            catalog_list=["cat1"],
            model_list=["mod1"],
            exp_list=["exp1"],
            outputdir=str(tmp_path),
        )

        # Execute save for the main figure
        base.save_figure(var="dummy_var", fig=fig, description="Test save figure", format="png", dpi=50)

        # OutputSaver creates a "png" subfolder in the outputdir
        png_dir = tmp_path / "png"
        assert png_dir.exists()

        # Check if the .png files were created
        png_files = list(png_dir.glob("*.png"))
        assert len(png_files) >= 1

        plt.close("all")
