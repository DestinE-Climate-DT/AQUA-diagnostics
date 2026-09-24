import numpy as np
import pytest
import xarray as xr

from aqua.core.util import select_season
from aqua.diagnostics.lat_lon_profiles import LatLonProfiles
from tests.shared_constants import LOGLEVEL

loglevel = LOGLEVEL


@pytest.mark.diagnostics
class TestLatLonProfilesZonal:
    """Basic tests for the LatLonProfiles class with zonal mean"""

    def setup_method(self):
        """Setup method to initialize LatLonProfiles instance"""
        self.diagnostic = LatLonProfiles(
            model="IFS",
            exp="test-tco79",
            source="teleconnections",
            startdate="1991-01-01",
            enddate="1992-12-31",
            mean_type="zonal",
            loglevel=loglevel,
        )

    def _assert_files_created(self, tmp_path, min_files=1):
        """Helper to check that files were created"""
        files = list(tmp_path.rglob("*.nc"))
        assert len(files) >= min_files, f"Expected at least {min_files} .nc files in {tmp_path}"
        return files

    def _assert_seasons_are_whole_period_climatologies(self, seasonal, dims):
        """Check that seasonal[i] is the whole-period climatology of the i-th season of [DJF, MAM, JJA, SON].

        The expected profile of each season is rebuilt independently, pooling every month of that season over
        every year via select_season, i.e. a different code path from the groupby('time.season') used to compute
        it. This pins down both the labelling (slot i really is that season) and the averaging window (all years,
        not just the first one). The season order is spelled out here on purpose, so that reordering the SEASONS
        constant in the diagnostic cannot keep this test green.
        """
        ref = self.diagnostic.reader.fldmean(
            self.diagnostic.data,
            lon_limits=self.diagnostic.lon_limits,
            lat_limits=self.diagnostic.lat_limits,
            dims=dims,
        )
        monthly_ref = self.diagnostic.reader.timmean(ref, freq="monthly", exclude_incomplete=True, center_time=True)
        for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
            expected = select_season(monthly_ref, season).mean("time")
            np.testing.assert_allclose(
                seasonal[i].values,
                expected.values,
                rtol=1e-4,
                err_msg=f"Seasonal slot {i} does not match the '{season}' climatology",
            )

    def test_retrieve_simple_var(self):
        """Test retrieve method with a simple variable"""
        self.diagnostic.retrieve(var="skt")

        assert self.diagnostic.data is not None
        assert isinstance(self.diagnostic.data, xr.DataArray)
        assert "skt" in str(self.diagnostic.data.name) or "skt" in str(self.diagnostic.data.attrs.get("standard_name", ""))

    def test_retrieve_with_formula(self):
        """Test retrieve method with a formula"""
        self.diagnostic.retrieve(
            var="skt+2", formula=True, long_name="Temperature plus 2", units="K", standard_name="skt_plus_2"
        )

        assert self.diagnostic.data is not None
        assert self.diagnostic.data.attrs["standard_name"] == "skt_plus_2"
        assert self.diagnostic.data.attrs["units"] == "K"

    @pytest.mark.parametrize("freq,attr_name", [("seasonal", "seasonal"), ("longterm", "longterm")])
    def test_compute_dim_mean(self, freq, attr_name):
        """Test computation of dimensional mean for different frequencies"""
        self.diagnostic.retrieve(var="skt")
        self.diagnostic.compute_dim_mean(freq=freq)

        data = getattr(self.diagnostic, attr_name)
        assert data is not None

        if freq == "seasonal":
            assert len(data) == 4  # DJF, MAM, JJA, SON
            for season_data in data:
                assert isinstance(season_data, xr.DataArray)
                assert "AQUA_mean_type" in season_data.attrs
                assert season_data.attrs["AQUA_mean_type"] == "zonal"

            self._assert_seasons_are_whole_period_climatologies(data, dims=["lon"])
        else:
            assert isinstance(data, xr.DataArray)
            assert "AQUA_mean_type" in data.attrs
            assert data.attrs["AQUA_mean_type"] == "zonal"

    @pytest.mark.parametrize("freq,std_attr", [("seasonal", "std_seasonal"), ("longterm", "std_annual")])
    def test_compute_std(self, freq, std_attr):
        """Test computation of standard deviation for different frequencies"""
        self.diagnostic.retrieve(var="skt")
        self.diagnostic.compute_std(freq=freq)

        std_data = getattr(self.diagnostic, std_attr)
        assert std_data is not None
        # 4 element of DataArray list for seasonal, single DataArray for longterm
        if freq == "seasonal":
            assert len(std_data) == 4
            for season_std in std_data:
                assert isinstance(season_std, xr.DataArray)
        elif freq == "longterm":
            assert isinstance(std_data, xr.DataArray)

    @pytest.mark.parametrize(
        "freq,with_std", [("seasonal", False), ("seasonal", True), ("longterm", False), ("longterm", True)]
    )
    def test_save_netcdf(self, tmp_path, freq, with_std):
        """Test saving data to netcdf with different frequencies and std options"""
        self.diagnostic.retrieve(var="skt")
        self.diagnostic.compute_dim_mean(freq=freq)

        if with_std:
            self.diagnostic.compute_std(freq=freq)

        self.diagnostic.save_netcdf(freq=freq, outputdir=str(tmp_path), rebuild=True)

        # Verify files were created
        self._assert_files_created(tmp_path)

        # Verify std data exists if requested
        if with_std:
            std_attr = "std_seasonal" if freq == "seasonal" else "std_annual"
            assert getattr(self.diagnostic, std_attr) is not None

    @pytest.mark.parametrize("freq", ["seasonal", "longterm", ["seasonal", "longterm"]])
    def test_run(self, tmp_path, freq):
        """Test full run method with different frequency options"""
        self.diagnostic.run(
            var="skt", freq=freq if isinstance(freq, list) else [freq], std=True, outputdir=str(tmp_path), rebuild=True
        )

        # Check that appropriate data was computed
        freq_list = freq if isinstance(freq, list) else [freq]

        for f in freq_list:
            if f == "seasonal":
                assert self.diagnostic.seasonal is not None
                assert self.diagnostic.std_seasonal is not None
            elif f == "longterm":
                assert self.diagnostic.longterm is not None
                assert self.diagnostic.std_annual is not None

        # Verify files were created
        self._assert_files_created(tmp_path)


@pytest.mark.diagnostics
class TestLatLonProfilesMeridional:
    """Basic tests for the LatLonProfiles class with meridional mean"""

    def setup_method(self):
        """Setup method to initialize LatLonProfiles instance with meridional mean"""
        self.diagnostic = LatLonProfiles(
            model="IFS",
            exp="test-tco79",
            source="teleconnections",
            startdate="1991-01-01",
            enddate="1992-12-31",
            mean_type="meridional",
            loglevel=loglevel,
        )

    def _assert_files_created(self, tmp_path):
        """Helper to check that files were created"""
        files = list(tmp_path.rglob("*.nc"))
        assert len(files) > 0, f"No .nc files found in {tmp_path}"
        return files

    @pytest.mark.parametrize("freq", ["seasonal", "longterm"])
    def test_compute_meridional_mean(self, freq):
        """Test computation of meridional mean for different frequencies"""
        self.diagnostic.retrieve(var="skt")
        self.diagnostic.compute_dim_mean(freq=freq)

        if freq == "seasonal":
            assert self.diagnostic.seasonal is not None
            assert len(self.diagnostic.seasonal) == 4
            for season_data in self.diagnostic.seasonal:
                assert season_data.attrs["AQUA_mean_type"] == "meridional"
        else:
            assert self.diagnostic.longterm is not None
            assert self.diagnostic.longterm.attrs["AQUA_mean_type"] == "meridional"

    def test_run_meridional(self, tmp_path):
        """Test full run method with meridional mean"""
        self.diagnostic.run(var="skt", freq=["seasonal", "longterm"], std=False, outputdir=str(tmp_path), rebuild=True)

        assert self.diagnostic.seasonal is not None
        assert self.diagnostic.longterm is not None
        self._assert_files_created(tmp_path)


@pytest.mark.diagnostics
class TestLatLonProfilesWithRegion:
    """Tests for LatLonProfiles class with region specification"""

    def _assert_files_created(self, tmp_path):
        """Helper to check that files were created"""
        files = list(tmp_path.rglob("*.nc"))
        assert len(files) > 0, f"No .nc files found in {tmp_path}"
        return files

    def test_compute_with_region_limits(self):
        """Test computation with specified region limits"""
        diagnostic = LatLonProfiles(
            model="IFS",
            exp="test-tco79",
            source="teleconnections",
            startdate="1991-01-01",
            enddate="1992-12-31",
            lon_limits=[-180, 180],
            lat_limits=[-60, 60],
            mean_type="zonal",
            loglevel=loglevel,
        )

        diagnostic.retrieve(var="skt")
        diagnostic.compute_dim_mean(freq="seasonal")

        assert diagnostic.seasonal is not None
        assert diagnostic.lon_limits == [-180, 180]
        assert diagnostic.lat_limits == [-60, 60]

    @pytest.mark.parametrize("freq", ["seasonal", "longterm"])
    def test_compute_with_region_name(self, freq):
        """Test computation with named region sets AQUA_region attribute"""
        diagnostic = LatLonProfiles(
            model="IFS",
            exp="test-tco79",
            source="teleconnections",
            startdate="1991-01-01",
            enddate="1992-12-31",
            region="tropics",
            mean_type="zonal",
            loglevel=loglevel,
        )

        diagnostic.retrieve(var="skt")
        diagnostic.compute_dim_mean(freq=freq)

        if freq == "seasonal":
            assert diagnostic.seasonal is not None
            for season_data in diagnostic.seasonal:
                assert "AQUA_region" in season_data.attrs
                assert season_data.attrs["AQUA_region"] == "Tropics"
        else:
            assert diagnostic.longterm is not None
            assert "AQUA_region" in diagnostic.longterm.attrs
            assert diagnostic.longterm.attrs["AQUA_region"] == "Tropics"

    @pytest.mark.parametrize("freq,with_std", [("seasonal", False), ("seasonal", True), ("longterm", True)])
    def test_save_with_region(self, tmp_path, freq, with_std):
        """Test saving data with region information"""
        diagnostic = LatLonProfiles(
            model="IFS",
            exp="test-tco79",
            source="teleconnections",
            startdate="1991-01-01",
            enddate="1992-12-31",
            region="tropics",
            mean_type="zonal",
            loglevel=loglevel,
        )

        diagnostic.retrieve(var="skt")
        diagnostic.compute_dim_mean(freq=freq)

        if with_std:
            diagnostic.compute_std(freq=freq)

        diagnostic.save_netcdf(freq=freq, outputdir=str(tmp_path), rebuild=True)
        self._assert_files_created(tmp_path)


@pytest.mark.diagnostics
class TestLatLonProfilesErrors:
    """Test error handling in LatLonProfiles class"""

    def test_invalid_mean_type_in_compute(self):
        """Test that invalid mean_type raises error in compute_dim_mean"""
        diagnostic = LatLonProfiles(
            model="IFS",
            exp="test-tco79",
            source="teleconnections",
            startdate="1991-01-01",
            enddate="1992-12-31",
            mean_type="invalid",
            loglevel=loglevel,
        )

        diagnostic.retrieve(var="skt")

        with pytest.raises(ValueError):
            diagnostic.compute_dim_mean(freq="seasonal")

    def test_invalid_mean_type_in_compute_std(self):
        """Test that invalid mean_type raises error in compute_std"""
        diagnostic = LatLonProfiles(
            model="IFS",
            exp="test-tco79",
            source="teleconnections",
            startdate="1991-01-01",
            enddate="1992-12-31",
            mean_type="invalid",
            loglevel=loglevel,
        )

        diagnostic.retrieve(var="skt")

        with pytest.raises(ValueError, match="Mean type invalid not recognized for std computation"):
            diagnostic.compute_std(freq="seasonal")

    def test_save_without_data(self, tmp_path):
        """Test save_netcdf without computing data first"""
        diagnostic = LatLonProfiles(
            model="IFS",
            exp="test-tco79",
            source="teleconnections",
            startdate="1991-01-01",
            enddate="1992-12-31",
            loglevel=loglevel,
        )

        # Should log error and return without raising exception
        diagnostic.save_netcdf(freq="seasonal", outputdir=str(tmp_path))

        # No files should be created when data is missing
        files = list(tmp_path.rglob("*.nc"))
        assert len(files) == 0, "No files should be created when data is missing"


@pytest.mark.diagnostics
class TestLatLonProfilesRealization:
    """Test realization extraction from data attributes"""

    def test_realization_in_filenames(self, tmp_path):
        """Test that realization appears in saved filenames"""
        diagnostic = LatLonProfiles(
            model="IFS",
            exp="test-tco79",
            source="teleconnections",
            startdate="1991-01-01",
            enddate="1992-12-31",
            mean_type="zonal",
            loglevel=loglevel,
        )

        diagnostic.retrieve(var="skt")

        # Manually set realization to test
        diagnostic.realization = "r5"
        if hasattr(diagnostic.data, "attrs"):
            diagnostic.data.attrs["AQUA_realization"] = "r5"

        assert diagnostic.realization == "r5"

        diagnostic.compute_dim_mean(freq="longterm")
        diagnostic.save_netcdf(freq="longterm", outputdir=str(tmp_path), rebuild=True)

        files = list(tmp_path.rglob("*.nc"))
        assert len(files) > 0
        assert any("r5" in f.name for f in files), "Realization 'r5' not found in any filename"


@pytest.mark.diagnostics
class TestLatLonProfilesLoad:
    """Tests for reading back the results written by a previous run"""

    def _profile(self, value: float):
        """A profile like the ones compute_dim_mean produces, without retrieving anything"""
        data = xr.DataArray(np.full(5, value), dims=["lat"], coords={"lat": np.arange(5.0)}, name="skt")
        data.attrs.update(
            {
                "standard_name": "skt",
                "units": "K",
                "AQUA_catalog": "ci",
                "AQUA_model": "IFS",
                "AQUA_exp": "test-tco79",
                "AQUA_mean_type": "zonal",
            }
        )
        return data

    def _producer(self, region=None):
        """A LatLonProfiles holding results, as it would be right before saving them"""
        diagnostic = LatLonProfiles(
            model="IFS", exp="test-tco79", source="teleconnections", catalog="ci", region=region, loglevel=loglevel
        )
        diagnostic.seasonal = [self._profile(value) for value in range(4)]
        diagnostic.longterm = self._profile(10.0)
        diagnostic.std_seasonal = [self._profile(value + 0.5) for value in range(4)]
        diagnostic.std_annual = self._profile(10.5)
        return diagnostic

    def _consumer(self, region=None):
        """A LatLonProfiles that never retrieved and does not even know its catalog"""
        return LatLonProfiles(model="IFS", exp="test-tco79", source="teleconnections", region=region, loglevel=loglevel)

    @pytest.mark.parametrize("region", [None, "tropics"])
    def test_load_roundtrip(self, tmp_path, region):
        """A run that only plots finds on disk exactly the results a previous run computed"""
        producer = self._producer(region=region)
        producer.save_netcdf(freq="seasonal", outputdir=str(tmp_path))
        producer.save_netcdf(freq="longterm", outputdir=str(tmp_path))

        consumer = self._consumer(region=region)
        consumer.load(var="skt", std=True, outputdir=str(tmp_path))

        # Each profile was built with its own constant, so the values pin down the season order too
        assert consumer.longterm.values[0] == 10.0
        assert consumer.std_annual.values[0] == 10.5
        np.testing.assert_allclose([data.values[0] for data in consumer.seasonal], [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose([data.values[0] for data in consumer.std_seasonal], [0.5, 1.5, 2.5, 3.5])

        # The attributes the plot classes read survived the round trip
        assert consumer.longterm.attrs["AQUA_mean_type"] == "zonal"
        assert consumer.longterm.attrs["units"] == "K"

    def test_load_nothing_on_disk(self, tmp_path):
        """With no files to read, the results are left as they are instead of being wiped"""
        consumer = self._consumer()
        consumer.load(var="skt", std=True, outputdir=str(tmp_path))

        assert consumer.seasonal is None
        assert consumer.longterm is None

        # A load that finds nothing must not destroy what a run has just computed
        consumer.longterm = self._profile(10.0)
        consumer.load(var="skt", outputdir=str(tmp_path))
        assert consumer.longterm is not None

    def test_load_incomplete_seasons(self, tmp_path):
        """A missing season gives no seasonal data at all, rather than a shorter, mislabelled list"""
        self._producer().save_netcdf(freq="seasonal", outputdir=str(tmp_path))

        # the season is lower cased in the filename, match it regardless of the case
        files = sorted(f for f in (tmp_path / "netcdf").glob("*.nc") if "jja" in f.name.lower())
        assert len(files) == 2  # the mean and the std of that season
        files[0].unlink()

        consumer = self._consumer()
        consumer.load(var="skt", freq="seasonal", outputdir=str(tmp_path))

        assert consumer.seasonal is None

    def test_load_realization(self, tmp_path):
        """The realization takes part in the filename, so a load has to be told the one of the run"""
        producer = self._producer()
        producer.realization = 2  # what retrieve would set from reader_kwargs
        producer.save_netcdf(freq="longterm", outputdir=str(tmp_path))

        # Without the reader_kwargs of the run, the default realization addresses another file
        consumer = self._consumer()
        consumer.load(var="skt", freq="longterm", outputdir=str(tmp_path))
        assert consumer.longterm is None

        consumer.load(var="skt", freq="longterm", outputdir=str(tmp_path), reader_kwargs={"realization": 2})
        assert consumer.longterm is not None
