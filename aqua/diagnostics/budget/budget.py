"""Budget tool: compare two global mean quantities, optionally the time derivative of the second one."""

import re

import numpy as np
import xarray as xr

from aqua.core.fixer import EvaluateFormula
from aqua.core.logger import log_configure
from aqua.core.util import DEFAULT_REALIZATION
from aqua.diagnostics.base import Diagnostic


class Budget(Diagnostic):
    """
    Compare, as global means, two quantities that should balance each other.

    Each quantity is a formula (or a single variable) of the same dataset. The second one can be replaced
    by its time derivative, to compare a flux with the change of the quantity it feeds: e.g. the net surface
    heat flux with the tendency of the ocean heat content, or the net TOA radiation with the net surface flux.
    The difference of the two is the budget imbalance, to be compared with its expected value in the plot.
    """

    # One full year is needed to resolve the annual cycle of the budget
    MINIMUM_MONTHS_REQUIRED = 12
    # Name of the NetCDF product; save_netcdf() and load() must agree on it
    DIAGNOSTIC_PRODUCT = "budget"

    def __init__(
        self,
        model: str,
        exp: str,
        source: str,
        catalog: str = None,
        regrid: str = None,
        startdate: str = None,
        enddate: str = None,
        diagnostic_name: str = "budget",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the Budget class.

        Args:
            model (str): The model to be used.
            exp (str): The experiment to be used.
            source (str): The source to be used.
            catalog (str, optional): The catalog to be used. If None, it is determined by the Reader.
            regrid (str, optional): The target grid for regridding. If None, no regridding is done.
            startdate (str, optional): The start date of the analysis. If None, all data are used.
            enddate (str, optional): The end date of the analysis. If None, all data are used.
            diagnostic_name (str, optional): The name used for the output files,
                usually the collection name. Default is 'budget'.
            loglevel (str, optional): The log level. Default is 'WARNING'.
        """
        super().__init__(
            model=model,
            exp=exp,
            source=source,
            catalog=catalog,
            regrid=regrid,
            startdate=startdate,
            enddate=enddate,
            loglevel=loglevel,
        )
        self.logger = log_configure(log_level=loglevel, log_name="Budget")
        self.diagnostic_name = diagnostic_name
        self.result = None

    def run(
        self,
        name: str,
        first: str,
        second: str,
        tendency: bool = False,
        mask: str = None,
        units: str = None,
        long_names: list = None,
        reader_kwargs: dict = None,
        outputdir: str = ".",
        rebuild: bool = True,
        save_netcdf: bool = True,
    ):
        """
        Retrieve the data, compute the two quantities and optionally save them.

        Args:
            name (str): The name of the budget, used in the filenames (e.g. 'ocean_heat').
            first (str): Formula (or variable) of the first quantity.
            second (str): Formula (or variable) of the second quantity.
            tendency (bool, optional): If True, compare with the time derivative of the second quantity.
            mask (str, optional): Variable whose missing values (at the first time step) are excluded
                from both means, e.g. 'tos' to average over the ocean only.
            units (str, optional): Units both quantities are converted to. If None, no conversion is done.
            long_names (list, optional): Long names of the two quantities. If None, the formulae are used.
            reader_kwargs (dict, optional): Additional keyword arguments for the Reader.
            outputdir (str, optional): The output directory. Default is '.'.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
            save_netcdf (bool, optional): If True, save the result to NetCDF. Default is True.

        Raises:
            NotEnoughDataError: If less than MINIMUM_MONTHS_REQUIRED months are available.
        """
        # Variable names of the formulae: identifiers not preceded by a digit or a dot (so 3.34e5 is a number)
        names = re.findall(r"(?<![\w.])[A-Za-z_]\w*", " ".join(filter(None, (first, second, mask))))
        self.retrieve(
            var=list(dict.fromkeys(names)), reader_kwargs=reader_kwargs or {}, months_required=self.MINIMUM_MONTHS_REQUIRED
        )
        self.compute_result(first=first, second=second, tendency=tendency, mask=mask, units=units, long_names=long_names)
        if save_netcdf:
            self.save_netcdf(name=name, outputdir=outputdir, rebuild=rebuild)

    def _global_mean(self, formula: str, long_name: str, ocean) -> xr.DataArray:
        """Evaluate a formula on self.data and return its (masked) global mean, keeping its attributes."""
        field = EvaluateFormula(data=self.data, formula=formula, long_name=long_name, loglevel=self.loglevel).evaluate()
        attrs = field.attrs
        if ocean is not None:
            field = field.where(ocean)
        return self.reader.fldmean(field).assign_attrs(attrs)

    def compute_result(
        self,
        first: str,
        second: str,
        tendency: bool = False,
        mask: str = None,
        units: str = None,
        long_names: list = None,
    ):
        """
        Compute the global means of the two quantities and store them in self.result ('first', 'second').

        Args:
            first (str): Formula (or variable) of the first quantity.
            second (str): Formula (or variable) of the second quantity.
            tendency (bool, optional): If True, replace the second quantity with its time derivative (per second).
            mask (str, optional): Variable whose missing values (at the first time step) are excluded from the means.
            units (str, optional): Units both quantities are converted to. If None, no conversion is done.
            long_names (list, optional): Long names of the two quantities. If None, the formulae are used.
        """
        first_name, second_name = long_names or (first, f"Time derivative of {second}" if tendency else second)
        # Missing values are excluded, not set to zero: the means of both quantities are over the same points
        ocean = self.data[mask].isel(time=0).notnull() if mask else None

        first_mean = self._global_mean(first, first_name, ocean)
        if tendency:
            second_mean = self._global_mean(second, second_name, ocean)
            # The derivative adds s**-1 to the units (J m**-2 becomes J m**-2 s**-1, i.e. W m**-2)
            attrs = {**second_mean.attrs, "units": f"{second_mean.attrs.get('units', '')} s**-1"}
            # Forward difference between consecutive steps, labelled at the first one; the last step has none
            seconds = second_mean.time.diff("time", label="lower") / np.timedelta64(1, "s")
            second_mean = (second_mean.diff("time", label="lower") / seconds).assign_attrs(attrs)
            first_mean = first_mean.sel(time=second_mean.time)
        else:
            second_mean = self._global_mean(second, second_name, ocean)

        if units:
            # _check_data keeps the original string when no conversion is needed: write the target units anyway
            first_mean, second_mean = (
                self._check_data(da, da.attrs["long_name"], units).assign_attrs(units=units)
                for da in (first_mean, second_mean)
            )

        result = xr.Dataset({"first": first_mean, "second": second_mean})
        # The provenance travels with the result: the period is on the dataset, the Reader attributes on the variables
        provenance = {**self.data[list(self.data.data_vars)[0]].attrs, **self.data.attrs}
        result.attrs.update({k: v for k, v in provenance.items() if k.startswith("AQUA_")})
        self.result = result.load()

    def _extra_keys(self, name: str) -> dict:
        """
        Build the filename keys of a result.

        Shared by save_netcdf and load, so that both address the same file.

        Args:
            name (str): The name of the budget appearing in the filename.

        Returns:
            dict: The extra keys, in the order in which they appear in the filename.
        """
        return {"budget": name}

    def save_netcdf(self, name: str, outputdir: str = ".", rebuild: bool = True):
        """
        Save self.result to a NetCDF file.

        Args:
            name (str): The name of the budget.
            outputdir (str, optional): The output directory. Default is '.'.
            rebuild (bool, optional): If True, overwrite an existing file. Default is True.
        """
        # load_netcdf() matches startdate/enddate against the dataset-level attributes: self.result is a Dataset
        # with the AQUA_* attributes there, so a plot-only run with dates in the configuration finds the file.
        super().save_netcdf(
            data=self.result,
            diagnostic=self.diagnostic_name,
            diagnostic_product=self.DIAGNOSTIC_PRODUCT,
            outputdir=outputdir,
            rebuild=rebuild,
            extra_keys=self._extra_keys(name),
        )

    def load(self, name: str, outputdir: str = ".", reader_kwargs: dict = None):
        """
        Populate self.result from the NetCDF file written by a previous run.

        A missing file, or one computed over a different period, leaves self.result untouched.

        Args:
            name (str): The name of the budget.
            outputdir (str, optional): The directory where the data was saved. Default is '.'.
            reader_kwargs (dict, optional): The Reader keyword arguments of the run, to match its realization.
        """
        self.realization = (reader_kwargs or {}).get("realization", DEFAULT_REALIZATION)
        data = self.load_netcdf(
            diagnostic=self.diagnostic_name,
            diagnostic_product=self.DIAGNOSTIC_PRODUCT,
            outputdir=outputdir,
            extra_keys=self._extra_keys(name),
            as_dataarray=False,
            startdate=self.startdate,
            enddate=self.enddate,
        )
        if data is not None:
            self.result = data
