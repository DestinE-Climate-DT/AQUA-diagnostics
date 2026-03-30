import pytest
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from aqua.diagnostics.histogram import PlotHistogram
from conftest import DPI, LOGLEVEL

loglevel = LOGLEVEL


def _make_hist_data(model='IFS', exp='test-tco79', catalog='test_catalog',
                    region='global', startdate='2000-01-01', enddate='2005-12-31',
                    realization=None):
    """Helper to create histogram DataArray with metadata."""
    bins = np.linspace(250, 320, 50)
    values = np.random.exponential(scale=10, size=50)
    attrs = {
        'AQUA_catalog': catalog,
        'AQUA_model': model,
        'AQUA_exp': exp,
        'AQUA_region': region,
        'AQUA_startdate': startdate,
        'AQUA_enddate': enddate,
        'short_name': 'skt',
        'standard_name': 'skin_temperature',
        'long_name': 'Skin Temperature',
        'units': 'K'
    }
    if realization is not None:
        attrs['AQUA_realization'] = realization

    da = xr.DataArray(values, dims=['center_of_bin'],
                      coords={'center_of_bin': bins}, attrs=attrs)
    da.center_of_bin.attrs['units'] = 'K'
    return da


@pytest.mark.diagnostics
class TestPlotHistogram:
    """Basic tests for the PlotHistogram class"""

    def setup_method(self):
        self.hist_data = _make_hist_data()
        self.ref_data = _make_hist_data(model='ERA5', exp='era5',
                                        startdate='1980-01-01', enddate='2020-12-31')

    def test_plot_histogram_initialization(self):
        """Test basic initialization and metadata extraction"""
        plotter = PlotHistogram(data=self.hist_data, loglevel=loglevel)

        assert plotter.len_data == 1
        assert plotter.len_ref == 0
        assert plotter.models[0] == 'IFS'
        assert plotter.exps[0] == 'test-tco79'
        assert plotter.region == 'global'

    def test_plot_histogram_with_ref(self):
        """Test initialization with reference data"""
        plotter = PlotHistogram(data=self.hist_data, ref_data=self.ref_data,
                                loglevel=loglevel)

        assert plotter.len_data == 1
        assert plotter.len_ref == 1

    def test_set_labels_and_title(self):
        """Test label and title generation"""
        plotter = PlotHistogram(data=self.hist_data, ref_data=self.ref_data,
                                loglevel=loglevel)

        data_labels = plotter.set_data_labels()
        ref_label = plotter.set_ref_label()
        title = plotter.set_title()

        assert 'IFS test-tco79' in data_labels[0]
        assert 'ERA5 era5' in ref_label
        assert 'Skin Temperature' in title or 'skin_temperature' in title
        assert 'global' in title

    def test_description_different_dates(self):
        """Test description with different date ranges for data and ref"""
        plotter = PlotHistogram(data=self.hist_data, ref_data=self.ref_data,
                                loglevel=loglevel)
        desc = plotter.set_description()

        # PDF prefix
        assert desc.startswith('Probability density function (PDF)')
        # Both date ranges present (different periods)
        assert '2000-01-01' in desc and '2005-12-31' in desc
        assert '1980-01-01' in desc and '2020-12-31' in desc
        # Model names
        assert 'IFS' in desc and 'ERA5' in desc

    def test_description_same_dates(self):
        """Test description when data and ref share the same period"""
        ref_same = _make_hist_data(model='ERA5', exp='era5',
                                   startdate='2000-01-01', enddate='2005-12-31')
        plotter = PlotHistogram(data=self.hist_data, ref_data=ref_same,
                                loglevel=loglevel)
        desc = plotter.set_description()

        # Dates should appear only once
        assert desc.count('2000-01-01') == 1
        assert 'vs' in desc

    def test_description_counts_mode(self):
        """Test description prefix when density=False"""
        plotter = PlotHistogram(data=self.hist_data, density=False,
                                loglevel=loglevel)
        desc = plotter.set_description()
        assert desc.startswith('Histogram of')

    def test_description_multiple_datasets(self):
        """Test description with multiple datasets"""
        data2 = _make_hist_data(exp='test-tco159')
        plotter = PlotHistogram(data=[self.hist_data, data2],
                                ref_data=self.ref_data, loglevel=loglevel)
        desc = plotter.set_description()

        assert 'comparing 2 datasets' in desc
        assert 'IFS/test-tco79' in desc
        assert 'IFS/test-tco159' in desc

    def test_realization_extraction(self):
        """Test that realization metadata is correctly extracted"""
        data_r = _make_hist_data(realization='r1i1p1f1')
        plotter = PlotHistogram(data=data_r, loglevel=loglevel)

        assert plotter.realizations[0] == 'r1i1p1f1'

    def test_description_non_global_region(self):
        """Test that non-global region appears in description"""
        data_tropics = _make_hist_data(region='Tropics')
        plotter = PlotHistogram(data=data_tropics, loglevel=loglevel)
        desc = plotter.set_description()

        assert 'over Tropics' in desc

    def test_plot_basic(self):
        """Test basic plotting"""
        plotter = PlotHistogram(data=self.hist_data, loglevel=loglevel)
        fig, ax = plotter.plot()

        assert fig is not None
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_plot_with_ref_and_smooth(self):
        """Test plotting with reference and smoothing"""
        plotter = PlotHistogram(data=self.hist_data, ref_data=self.ref_data,
                                loglevel=loglevel)
        fig, ax = plotter.plot(smooth=True, smooth_window=5,
                               xlogscale=True, ylogscale=True)

        assert len(ax.lines) == 2
        plt.close(fig)

    def test_run_complete(self, tmp_path):
        """Test complete run method"""
        plotter = PlotHistogram(data=self.hist_data, ref_data=self.ref_data,
                                loglevel=loglevel)
        plotter.run(outputdir=str(tmp_path), rebuild=True, dpi=DPI,
                    format='png', smooth=True)

        assert plotter.data is not None
        assert plotter.ref_data is not None
