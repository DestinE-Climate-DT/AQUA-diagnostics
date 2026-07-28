"""Timeseries class for retrieve and netcdf saving of a single experiment"""

import xarray as xr

from aqua.core.util import frequency_string_to_pandas, pandas_freq_to_string, to_list

from .base import BaseMixin

xr.set_options(keep_attrs=True)


class Timeseries(BaseMixin):
    """Timeseries class for retrieve and netcdf saving of a single experiment"""

    MINIMUM_MONTHS_REQUIRED = 2

    def __init__(
        self,
        diagnostic_name: str = "timeseries",
        catalog: str = None,
        model: str = None,
        exp: str = None,
        source: str = None,
        regrid: str = None,
        startdate: str = None,
        enddate: str = None,
        std_startdate: str = None,
        std_enddate: str = None,
        region: str = None,
        lon_limits: list = None,
        lat_limits: list = None,
        loglevel: str = "WARNING",
    ):
        """
        Initialize the Timeseries class.

        Args:
            diagnostic_name (str): The name of the diagnostic. Used for logger and filenames. Default is 'timeseries'.
            catalog (str): The catalog to be used. If None, the catalog will be determined by the Reader.
            model (str): The model to be used.
            exp (str): The experiment to be used.
            source (str): The source to be used.
            regrid (str): The target grid to be used for regridding. If None, no regridding will be done.
            startdate (str): The start date of the data to be retrieved.
                             If None, all available data will be retrieved.
            enddate (str): The end date of the data to be retrieved.
                           If None, all available data will be retrieved.
            std_startdate (str): The start date of the standard period.
            std_enddate (str): The end date of the standard period.
            region (str): The region to select. This will define the lon and lat limits.
            lon_limits (list): The longitude limits to be used. Overriden by region.
            lat_limits (list): The latitude limits to be used. Overriden by region.
            loglevel (str): The log level to be used. Default is 'WARNING'.
        """
        super().__init__(
            diagnostic_name=diagnostic_name,
            catalog=catalog,
            model=model,
            exp=exp,
            source=source,
            regrid=regrid,
            startdate=startdate,
            enddate=enddate,
            std_startdate=std_startdate,
            std_enddate=std_enddate,
            region=region,
            lon_limits=lon_limits,
            lat_limits=lat_limits,
            loglevel=loglevel,
        )

    def run(
        self,
        var: str,
        formula: bool = False,
        long_name: str = None,
        units: str = None,
        short_name: str = None,
        std: bool = False,
        freq: list = ["monthly", "annual"],
        exclude_incomplete: bool = True,
        center_time: bool = True,
        box_brd: bool = True,
        outputdir: str = "./",
        rebuild: bool = True,
        reader_kwargs: dict = {},
        create_catalog_entry: bool = False,
    ):
        """
        Run all the steps necessary for the computation of the Timeseries.
        Save the results to netcdf files.
        Can evaluate different frequencies.

        Args:
            var (str): The variable to be retrieved.
            formula (bool): If True, the variable is a formula.
            long_name (str): The long name of the variable, if different from the variable name.
            units (str): The units of the variable, if different from the original units.
            short_name (str): The short name of the variable, if different from the variable name.
            std (bool): If True, compute the standard deviation. Default is False.
            freq (list): The frequencies to be used for the computation. Available options are 'hourly', 'daily',
                         'monthly' and 'annual'. Default is ['monthly', 'annual'].
            exclude_incomplete (bool): If True, exclude incomplete periods.
            center_time (bool): If True, the time will be centered.
            box_brd (bool): choose if coordinates are comprised or not in area selection.
            outputdir (str): The directory to save the data.
            rebuild (bool): If True, rebuild the data from the original files.
            reader_kwargs (dict): Additional keyword arguments for the Reader. Default is an empty dictionary.
            create_catalog_entry (bool): If True, create a catalog entry for the data. Default is False.
        """
        self.logger.info("Running Timeseries for %s", var)
        self.retrieve(
            var=var, formula=formula, long_name=long_name, units=units, short_name=short_name, reader_kwargs=reader_kwargs
        )
        freq = to_list(freq)

        for f in freq:
            self.compute(freq=f, exclude_incomplete=exclude_incomplete, center_time=center_time, box_brd=box_brd)
            if std:
                if self.std_startdate is None or self.std_enddate is None:
                    self.logger.error(
                        "Skipping std evaluation. Std start and end dates must be provided to compute the standard deviation."
                    )
                else:
                    self.compute_std(freq=f, exclude_incomplete=exclude_incomplete, center_time=center_time, box_brd=box_brd)
            self.save_netcdf(
                diagnostic_product="timeseries",
                freq=f,
                outputdir=outputdir,
                rebuild=rebuild,
                create_catalog_entry=create_catalog_entry,
            )

    def compute(self, freq: str, exclude_incomplete: bool = True, center_time: bool = True, box_brd: bool = True):
        """
        Compute the mean of the data. Support for hourly, daily, monthly and annual means.

        Args:
            freq (str): The frequency to be used for the resampling.
            exclude_incomplete (bool): If True, exclude incomplete periods.
            center_time (bool): If True, the time will be centered.
            box_brd (bool,opt): choose if coordinates are comprised or not in area selection.
                                Default is True
        """
        if freq is None:
            self.logger.error("Frequency not provided, cannot compute mean")
            return

        freq = frequency_string_to_pandas(freq)
        str_freq = pandas_freq_to_string(freq)

        self.logger.info("Computing %s mean", str_freq)
        data = self.data

        # Field and time average
        data = self.reader.fldmean(data, box_brd=box_brd, lon_limits=self.lon_limits, lat_limits=self.lat_limits)
        data = self.reader.timmean(data, freq=freq, exclude_incomplete=exclude_incomplete, center_time=center_time)

        # If no data is available after the time mean, return
        if data.time.size == 0:
            self.logger.warning(f"Not enough data available to compute {str_freq} mean")
            data = None
        else:  # Enough data is available, we can proceed with the computation and saving
            if self.region is not None:
                data.attrs["AQUA_region"] = self.region

            # Load data in memory for faster plot
            self.logger.debug(f"Loading data for frequency {str_freq} in memory")
            data.load()
            self.logger.debug(f"Loaded data for frequency {str_freq} in memory")

        if str_freq == "hourly":
            self.hourly = data
        elif str_freq == "daily":
            self.daily = data
        elif str_freq == "monthly":
            self.monthly = data
        elif str_freq == "annual":
            self.annual = data
