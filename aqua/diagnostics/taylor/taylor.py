"""Module for the statistics of the Taylor diagram of a dataset against a reference."""

import pandas as pd
import xarray as xr

from aqua.core.exceptions import NoDataError
from aqua.core.fixer import EvaluateFormula
from aqua.core.logger import log_configure
from aqua.core.util import DEFAULT_REALIZATION, select_season
from aqua.diagnostics.base import Diagnostic, OutputSaver
from aqua.diagnostics.biases.util import handle_pressure_level

from .stat_taylor import compute_taylor_statistics

xr.set_options(keep_attrs=True)

SEASONS = ["DJF", "MAM", "JJA", "SON"]
ANNUAL = "annual"


class Taylor(Diagnostic):
    """
    Diagnostic class for the statistics of the Taylor diagram of a single dataset.

    One instance handles one dataset. The climatology of a variable is computed by ``run``, as the
    annual mean and optionally the four seasonal means. The reference dataset is another instance of
    this class, so that its climatology is computed once and shared by all the datasets to evaluate.
    ``compute_statistics`` then compares the spatial pattern of the two climatologies and produces the
    standard deviation, correlation and centred RMS difference needed by the Taylor diagram, which are
    saved as a NetCDF file and can be read back by ``load`` to plot without recomputing.

    Inherits from `Diagnostic`.

    Args:
        model (str): The model to be used.
        exp (str): The experiment to be used.
        source (str): The source to be used.
        catalog (str): The catalog to be used. If None, the catalog will be determined by the Reader.
        regrid (str): Target grid for regridding. If None, no regridding. Data and reference must
            end up on the same grid.
        startdate (str): Start date of the climatology. If None, all available data is used.
        enddate (str): End date of the climatology. If None, all available data is used.
        region (str): Region of the centralized regions file to select. If None, the whole domain is used.
        regions_file_path (str): Path to a custom regions file. If None, the centralized one is used.
        diagnostic_name (str): Name of the diagnostic, used in the filenames. Default is 'taylor'.
        loglevel (str): Log level. Default is 'WARNING'.
    """

    MINIMUM_MONTHS_REQUIRED = 12

    def __init__(
        self,
        model: str,
        exp: str,
        source: str,
        catalog: str = None,
        regrid: str = None,
        startdate: str = None,
        enddate: str = None,
        region: str = None,
        regions_file_path: str = None,
        diagnostic_name: str = "taylor",
        loglevel: str = "WARNING",
    ):
        super().__init__(
            catalog=catalog,
            model=model,
            exp=exp,
            source=source,
            regrid=regrid,
            startdate=startdate,
            enddate=enddate,
            loglevel=loglevel,
        )
        self.logger = log_configure(log_level=loglevel, log_name="Taylor")
        self.diagnostic_name = diagnostic_name
        self.region_key = region
        self.regions_file_path = regions_file_path
        # Long name of the region, used for titles and filenames, resolved without data access
        self.region = self._set_region(region=region, regions_file_path=regions_file_path)[0] if region is not None else None

        self.var = None
        self.plev = None
        # Results
        self.climatology = None
        self.area = None
        self.statistics = None
        # Reference used for the statistics, part of the filenames
        self.catalog_ref = None
        self.model_ref = None
        self.exp_ref = None

    def run(
        self,
        var: str,
        formula: bool = False,
        long_name: str = None,
        short_name: str = None,
        units: str = None,
        plev: float = None,
        seasonal: bool = True,
        reader_kwargs: dict = {},
    ):
        """
        Retrieve the data and compute its climatology.

        Args:
            var (str): The variable to retrieve, or the formula to evaluate if formula is True.
            formula (bool): If True, var is a formula of the dataset variables. Default is False.
            long_name (str): The long name of the variable, if different from the original one.
            short_name (str): The short name of the variable, if different from var. Recommended with formulae.
            units (str): The units to convert the variable to, if different from the original ones.
            plev (float): The pressure level to select, for 3D variables.
            seasonal (bool): If True, compute the seasonal climatologies as well. Default is True.
            reader_kwargs (dict): Additional keyword arguments for the Reader.
        """
        self.retrieve(
            var=var,
            formula=formula,
            long_name=long_name,
            short_name=short_name,
            units=units,
            plev=plev,
            reader_kwargs=reader_kwargs,
        )
        self.compute_climatology(seasonal=seasonal)

    def retrieve(
        self,
        var: str,
        formula: bool = False,
        long_name: str = None,
        short_name: str = None,
        units: str = None,
        plev: float = None,
        reader_kwargs: dict = {},
    ):
        """
        Retrieve the variable, evaluating the formula, converting the units and selecting the pressure level and region.

        Args:
            var (str): The variable to retrieve, or the formula to evaluate if formula is True.
            formula (bool): If True, var is a formula of the dataset variables. Default is False.
            long_name (str): The long name of the variable, if different from the original one.
            short_name (str): The short name of the variable, if different from var.
            units (str): The units to convert the variable to, if different from the original ones.
            plev (float): The pressure level to select, for 3D variables.
            reader_kwargs (dict): Additional keyword arguments for the Reader.

        Raises:
            NoDataError: If no data is retrieved.
            KeyError: If the variable is not in the dataset.
        """
        if formula:
            super().retrieve(reader_kwargs=reader_kwargs, months_required=self.MINIMUM_MONTHS_REQUIRED)
            self.logger.info("Evaluating formula: %s", var)
            values = EvaluateFormula(
                data=self.data, formula=var, long_name=long_name, short_name=short_name, units=units, loglevel=self.loglevel
            ).evaluate()
            if values is None:
                raise ValueError(f"Error evaluating formula {var}. Check the variable names and the formula syntax.")
            self.data[var] = values
        else:
            super().retrieve(var=var, reader_kwargs=reader_kwargs, months_required=self.MINIMUM_MONTHS_REQUIRED)

        if self.data is None:
            raise NoDataError(f"No data retrieved for {self.model} {self.exp} {self.source}")
        if var not in self.data.data_vars:
            raise KeyError(f"Variable '{var}' not found in dataset. Available variables: {list(self.data.data_vars)}")

        data = self.data[var]
        if units is not None:
            data = self._check_data(data=data, var=var, units=units)
        if long_name is not None:
            data.attrs["long_name"] = long_name

        self.var = short_name or var
        data.name = self.var
        data.attrs["short_name"] = self.var

        if plev is not None:
            self.plev = plev
            data = handle_pressure_level(data.to_dataset(), self.var, plev, loglevel=self.loglevel)[self.var]
        elif "plev" in data.dims:
            raise ValueError(f"Variable '{var}' has multiple pressure levels, select one with plev")

        if self.region_key is not None:
            data = self.select_region(data=data, region=self.region_key)["data"]

        # Keep the resolved dates as strings, they end up in the metadata of the output
        self.startdate = pd.Timestamp(self.startdate).strftime("%Y-%m-%d")
        self.enddate = pd.Timestamp(self.enddate).strftime("%Y-%m-%d")
        self.data = data
        self.area = self._area_weights()

    def _area_weights(self) -> xr.DataArray | None:
        """
        Get the cell areas of the grid of the data, restricted to the selected region.

        Returns:
            xr.DataArray or None: The cell areas, or None if the Reader does not provide them,
                in which case the statistics are not weighted.
        """
        grid_area = self.reader.tgt_grid_area if self.regrid else self.reader.src_grid_area
        if grid_area is None or "cell_area" not in grid_area:
            self.logger.warning("No cell area available for %s %s, statistics are not area weighted", self.model, self.exp)
            return None

        area = grid_area["cell_area"]
        if self.region_key is not None:
            area = self.select_region(data=area, region=self.region_key)["data"]
        return area.load()

    def compute_climatology(self, seasonal: bool = True):
        """
        Compute the annual and optionally the seasonal climatologies of the retrieved variable.

        The climatologies are stacked along a 'period' dimension ('annual', 'DJF', 'MAM', 'JJA', 'SON'),
        so that the statistics of all of them are computed together.

        Args:
            seasonal (bool): If True, compute the seasonal climatologies as well. Default is True.

        Raises:
            ValueError: If no data has been retrieved.
        """
        if self.data is None:
            raise ValueError("No data retrieved, run retrieve first")

        self.logger.info("Computing the climatology of %s", self.var)
        climatologies = [self.data.mean(dim="time").expand_dims(period=[ANNUAL])]
        if seasonal:
            for season in SEASONS:
                climatologies.append(select_season(self.data, season).mean(dim="time").expand_dims(period=[season]))

        climatology = xr.concat(climatologies, dim="period")
        climatology.name = self.var
        climatology.attrs.update(self._identity_attrs())

        self.logger.debug("Loading the climatology in memory")
        self.climatology = climatology.load()

    def compute_statistics(self, reference: "Taylor", save_netcdf: bool = True, outputdir: str = "./", rebuild: bool = True):
        """
        Compute the Taylor diagram statistics of the climatology against the one of the reference.

        The statistics are computed over the spatial dimensions, weighted by the cell area,
        separately for each period of the climatology.

        Args:
            reference (Taylor): The reference, whose climatology has already been computed.
            save_netcdf (bool): If True, save the statistics as a NetCDF file. Default is True.
            outputdir (str): The output directory. Default is './'.
            rebuild (bool): If True, overwrite an existing NetCDF file. Default is True.

        Raises:
            ValueError: If the climatology of the dataset or of the reference is missing,
                or if the two are not on the same grid.
        """
        if self.climatology is None or reference.climatology is None:
            raise ValueError("The climatologies of both dataset and reference are needed, call run on both first")

        self._set_reference(reference)
        periods = [p for p in self.climatology.period.values if p in reference.climatology.period.values]
        dims = [dim for dim in self.climatology.dims if dim != "period"]

        self.logger.info("Computing the Taylor statistics of %s against %s %s", self.var, self.model_ref, self.exp_ref)
        statistics = compute_taylor_statistics(
            data=self.climatology.sel(period=periods),
            data_ref=reference.climatology.sel(period=periods),
            weights=self.area,
            dims=dims,
            loglevel=self.loglevel,
        )

        # The selected level is kept in the attributes
        statistics = statistics.drop_vars("plev", errors="ignore")
        statistics.attrs.update(self._identity_attrs())
        statistics.attrs.update(
            {
                "AQUA_catalog_ref": reference.catalog,
                "AQUA_model_ref": reference.model,
                "AQUA_exp_ref": reference.exp,
                "AQUA_startdate_ref": str(reference.startdate),
                "AQUA_enddate_ref": str(reference.enddate),
            }
        )
        self.statistics = statistics

        for period in periods:
            stats = statistics.sel(period=period)
            self.logger.info(
                "%s %s: normalised std %.3f, correlation %.3f, normalised CRMSD %.3f",
                self.var,
                period,
                stats["std_norm"],
                stats["corr"],
                stats["crmse_norm"],
            )

        if save_netcdf:
            self.save_netcdf(
                data=self.statistics,
                diagnostic=self.diagnostic_name,
                diagnostic_product="taylor_statistics",
                outputdir=outputdir,
                rebuild=rebuild,
                extra_keys=self._extra_keys(),
            )

    def load(
        self,
        var: str,
        reference: "Taylor",
        plev: float = None,
        outputdir: str = "./",
        reader_kwargs: dict = {},
        expect_netcdf: bool = False,
    ):
        """
        Populate the statistics from the NetCDF file written by a previous run.

        A missing file leaves the statistics untouched, so that load can be called both after
        compute_statistics and alone, to plot without recomputing anything. If startdate or enddate
        is set, a file computed over a different period is not loaded.

        Args:
            var (str): The variable name appearing in the filename, i.e. the short name if one was given.
            reference (Taylor): The reference used for the statistics, it does not need to be run.
            plev (float): The pressure level given to run, if any.
            outputdir (str): The directory where the data was saved. Default is './'.
            reader_kwargs (dict): The Reader keyword arguments of the run, to match its realization.
            expect_netcdf (bool): If True, a missing file is logged as an error. Default is False.
        """
        self.var = var
        self.plev = plev
        self.realization = reader_kwargs.get("realization", DEFAULT_REALIZATION)
        self._set_reference(reference)

        data = self.load_netcdf(
            diagnostic=self.diagnostic_name,
            diagnostic_product="taylor_statistics",
            outputdir=outputdir,
            extra_keys=self._extra_keys(),
            startdate=self.startdate,
            enddate=self.enddate,
            expect_netcdf=expect_netcdf,
        )
        if data is not None:
            self.statistics = data

    def _set_reference(self, reference: "Taylor"):
        """
        Store the identity of the reference, which takes part in the filenames.

        Args:
            reference (Taylor): The reference instance. Its catalog is resolved if it was not retrieved.
        """
        if reference.catalog is None:
            reference._resolve_catalog()
        self.catalog_ref = reference.catalog
        self.model_ref = reference.model
        self.exp_ref = reference.exp

    def _identity_attrs(self) -> dict:
        """
        Attributes describing the dataset, its period and the variable.

        Returns:
            dict: The attributes.
        """
        attrs = {
            "AQUA_catalog": self.catalog,
            "AQUA_model": self.model,
            "AQUA_exp": self.exp,
            "AQUA_realization": self.realization,
            "AQUA_startdate": str(self.startdate),
            "AQUA_enddate": str(self.enddate),
            "short_name": self.var,
            "long_name": self.data.attrs.get("long_name", self.var) if self.data is not None else self.var,
        }
        if self.data is not None and "units" in self.data.attrs:
            attrs["units"] = self.data.attrs["units"]
        if self.plev is not None:
            attrs["AQUA_plev"] = self.plev
        if self.region is not None:
            attrs["AQUA_region"] = self.region
        return attrs

    def _extra_keys(self) -> dict:
        """
        Keys added to the filenames, the same for save and load.

        Returns:
            dict: The extra keys.
        """
        extra_keys = {"var": self.var}
        if self.plev is not None:
            extra_keys["plev"] = self.plev
        if self.region is not None:
            extra_keys["region"] = self.region
        return extra_keys

    def _outputsaver(self, diagnostic: str, outputdir: str = "."):
        """
        Build the OutputSaver, with the reference in the filenames since the statistics depend on it.

        Args:
            diagnostic (str): The diagnostic name.
            outputdir (str): The path where the data is saved. Default is '.'.

        Returns:
            OutputSaver: The output saver for this dataset and reference.
        """
        return OutputSaver(
            diagnostic=diagnostic,
            catalog=self.catalog,
            model=self.model,
            exp=self.exp,
            realization=self.realization,
            catalog_ref=self.catalog_ref,
            model_ref=self.model_ref,
            exp_ref=self.exp_ref,
            outputdir=outputdir,
            loglevel=self.loglevel,
        )
