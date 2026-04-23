from aqua.core.fixer import EvaluateFormula
from aqua.core.logger import log_configure
from aqua.core.util import pandas_freq_to_string, time_to_string, to_list, xarray_to_pandas_freq
from aqua.diagnostics.base import Diagnostic


class LatLonProfiles(Diagnostic):
    """
    Class to compute lat-lon profiles of a variable over a specified region.
    It retrieves the data from the catalog, computes the mean and standard deviation
    over the specified period and saves the results to netcdf files.

    Supported Frequencies:
        - 'seasonal': Computes seasonal means (DJF, MAM, JJA, SON)
        - 'longterm': Computes the temporal mean over the entire analysis period

    Supported Mean Types:
        - 'zonal': Average over longitude, producing latitude profiles
        - 'meridional': Average over latitude, producing longitude profiles

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
        std_startdate: str = None,
        std_enddate: str = None,
        region: str = None,
        lon_limits: list = None,
        lat_limits: list = None,
        regions_file_path: str = None,
        mean_type: str = "zonal",
        diagnostic_name: str = "latlonprofile",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the LatLonProfiles class.

        Args:
            model (str): The model to be used for the retrieval of the data.
            exp (str): The experiment to be used for the retrieval of the data.
            source (str): The source to be used for the retrieval of the data.
            catalog (str, optional): The catalog to be used for the retrieval of the data.
            regrid (str, optional): The regridding method to be used for the retrieval of the data.
            startdate (str, optional): The start date of the plot/analysis period.
            enddate (str, optional): The end date of the plot/analysis period.
            std_startdate (str, optional): The start date of the standard deviation period.
            std_enddate (str, optional): The end date of the standard deviation period.
            region (str, optional): The region to be used for the retrieval of the data.
            lon_limits (list, optional): The longitude limits of the region.
            lat_limits (list, optional): The latitude limits of the region.
            regions_file_path (str, optional): The path to the regions file. Default is the AQUA config path.
            mean_type (str, optional): The type of mean to compute ('zonal' or 'meridional').
            diagnostic_name (str, optional): The name of the diagnostic.
            loglevel (str, optional): The log level to be used for the logging.
        """

        super().__init__(
            catalog=catalog,
            model=model,
            exp=exp,
            source=source,
            regrid=regrid,
            startdate=startdate,
            enddate=enddate,
            std_startdate=std_startdate,
            std_enddate=std_enddate,
            loglevel=loglevel,
        )
        self.diagnostic_name = diagnostic_name

        self.logger = log_configure(log_level=loglevel, log_name="LatLonProfiles")

        # Set the region based on the region name or the lon and lat limits
        self.region, self.lon_limits, self.lat_limits = self._set_region(
            region=region,
            diagnostic="lat_lon_profiles",
            regions_file_path=regions_file_path,
            lon_limits=lon_limits,
            lat_limits=lat_limits,
        )

        # Initialize the possible results
        self.seasonal = None  # Seasonal means [DJF, MAM, JJA, SON]
        self.longterm = None  # Temporal mean over the entire time period
        self.std_seasonal = None  # Seasonal std deviations [DJF, MAM, JJA, SON]
        self.std_annual = None  # Annual std deviation, used by the longterm mean

        self.mean_type = mean_type

    def retrieve(
        self,
        var: str,
        formula: bool = False,
        long_name: str = None,
        units: str = None,
        standard_name: str = None,
        reader_kwargs: dict = {},
    ):
        """
        Retrieve the data for the specified variable and apply any formula if required.

        Args:
            var (str): The variable to be retrieved.
            formula (bool): Whether to use a formula for the variable.
            long_name (str): The long name of the variable.
            units (str): The units of the variable.
            standard_name (str): The standard name of the variable.
            reader_kwargs (dict): Additional keyword arguments for the Reader. Default is an empty dictionary.
        """
        self.logger.info("Retrieving data for variable %s", var)

        if formula:
            super().retrieve(reader_kwargs=reader_kwargs, months_required=self.MINIMUM_MONTHS_REQUIRED)
            self.logger.debug("Evaluating formula %s", var)
            self.data = EvaluateFormula(
                data=self.data, formula=var, long_name=long_name, short_name=standard_name, units=units, loglevel=self.loglevel
            ).evaluate()
            if self.data is None:
                raise ValueError(f"Error evaluating formula {var}. Check the variable names and the formula syntax.")
            if self.std_data is not None:
                self.std_data = EvaluateFormula(
                    data=self.std_data,
                    formula=var,
                    long_name=long_name,
                    short_name=standard_name,
                    units=units,
                    loglevel=self.loglevel,
                ).evaluate()
        else:
            super().retrieve(var=var, reader_kwargs=reader_kwargs, months_required=self.MINIMUM_MONTHS_REQUIRED)
            if var not in self.data:
                raise ValueError(f"Variable {var} not found in the data. Check the variable name and the data source.")
            # Get the xr.DataArray to be aligned with the formula code
            self.data = self.data[var]
            if self.std_data is not None:
                self.std_data = self.std_data[var]

        # Customization of the data, especially needed for formula
        self.data = self._apply_variable_metadata(
            self.data, var=var, units=units, long_name=long_name, standard_name=standard_name
        )
        if self.std_data is not None:
            self.std_data = self._apply_variable_metadata(
                self.std_data, var=var, units=units, long_name=long_name, standard_name=standard_name
            )
        else:
            # Fallback: no std dates provided, use plot window for std too
            self.std_data = self.data
            self.std_startdate = self.startdate
            self.std_enddate = self.enddate

    def _apply_variable_metadata(self, data, var, units=None, long_name=None, standard_name=None):
        """Apply unit conversion and set variable metadata attributes in-place."""
        if units is not None:
            data = self._check_data(data=data, var=var, units=units)
        if long_name is not None:
            data.attrs["long_name"] = long_name
        if standard_name is not None:
            data.attrs["standard_name"] = standard_name
            data.name = standard_name
        else:
            data.attrs["standard_name"] = var
        return data

    def compute_std(self, freq: str, exclude_incomplete: bool = True, center_time: bool = True, box_brd: bool = True):
        """
        Compute the standard deviation of the data over the std period.
        Supports seasonal and longterm frequencies.

        Args:
            freq (str): The frequency to be used ('seasonal' or 'longterm').
            exclude_incomplete (bool): If True, exclude incomplete periods.
            center_time (bool): If True, the time will be centered.
            box_brd (bool,opt): choose if coordinates are comprised or not in area selection. Default is True
        """
        self.logger.info("Computing %s standard deviation", freq)

        if self.std_data is None:
            self.logger.warning("No std_data available; cannot compute %s std", freq)
            return

        if self.mean_type == "zonal":
            dims = ["lon"]
        elif self.mean_type == "meridional":
            dims = ["lat"]
        else:
            raise ValueError(f"Mean type {self.mean_type} not recognized for std computation.")

        data = self.reader.fldmean(
            self.std_data, box_brd=box_brd, lon_limits=self.lon_limits, lat_limits=self.lat_limits, dims=dims
        )
        monthly_data = self.reader.timmean(
            data, freq="monthly", exclude_incomplete=exclude_incomplete, center_time=center_time
        )

        self.logger.debug("Loading monthly data in memory for std computation")
        monthly_data.load()

        data_freq = pandas_freq_to_string(xarray_to_pandas_freq(self.data))

        if freq == "seasonal":
            # Group by season and compute std
            seasonal_std = monthly_data.groupby("time.season").std("time")

            # Convert to list [DJF, MAM, JJA, SON]
            seasons = ["DJF", "MAM", "JJA", "SON"]
            seasonal_std_list = []
            for season in seasons:
                season_data = seasonal_std.sel(season=season)
                season_data.attrs["AQUA_std_startdate"] = time_to_string(self.std_startdate)
                season_data.attrs["AQUA_std_enddate"] = time_to_string(self.std_enddate)
                season_data.attrs["AQUA_data_freq"] = data_freq
                seasonal_std_list.append(season_data)

            self.std_seasonal = seasonal_std_list

            for season_data in self.std_seasonal:
                season_data.load()

        elif freq == "longterm":
            annual_data = monthly_data.groupby("time.year").mean("time")
            annual_std = annual_data.std("year")
            annual_std.attrs["AQUA_std_startdate"] = time_to_string(self.std_startdate)
            annual_std.attrs["AQUA_std_enddate"] = time_to_string(self.std_enddate)
            annual_std.attrs["AQUA_data_freq"] = data_freq
            self.std_annual = annual_std

            self.std_annual.load()

    def save_netcdf(self, freq: str, outputdir: str = "./", rebuild: bool = True):
        """
        Save the data to a netcdf file.

        Args:
            freq (str): The frequency of the data ('seasonal' or 'longterm').
            outputdir (str): The directory to save the data.
            rebuild (bool): If True, rebuild the data from the original files.
        """
        if freq == "seasonal":
            data = self.seasonal if self.seasonal is not None else None
            data_std = self.std_seasonal if self.std_seasonal is not None else None
            if data is None:
                self.logger.error("No seasonal data available")
                return
        elif freq == "longterm":
            data = self.longterm if self.longterm is not None else None
            data_std = self.std_annual if self.std_annual is not None else None
            if data is None:
                self.logger.error("No longterm data available")
                return

        diagnostic_product = f"{self.mean_type}_profile"

        if freq == "seasonal":
            seasons = ["DJF", "MAM", "JJA", "SON"]
            for i, season_data in enumerate(data):
                var = getattr(season_data, "standard_name", "unknown")

                extra_keys = {"freq": freq, "season": seasons[i], "var": var}
                if self.region is not None:
                    extra_keys["AQUA_region"] = self.region

                self.logger.info("Saving %s data for %s to netcdf in %s", seasons[i], diagnostic_product, outputdir)
                super().save_netcdf(
                    data=season_data,
                    diagnostic=self.diagnostic_name,
                    diagnostic_product=diagnostic_product,
                    outputdir=outputdir,
                    rebuild=rebuild,
                    extra_keys=extra_keys,
                )
        elif freq == "longterm":
            var = getattr(data, "standard_name", "unknown")

            extra_keys = {"freq": freq, "var": var}
            if self.region is not None:
                extra_keys["AQUA_region"] = self.region

            self.logger.info("Saving %s data for %s to netcdf in %s", freq, diagnostic_product, outputdir)
            super().save_netcdf(
                data=data,
                diagnostic=self.diagnostic_name,
                diagnostic_product=diagnostic_product,
                outputdir=outputdir,
                rebuild=rebuild,
                extra_keys=extra_keys,
            )

        if data_std is not None:
            if freq == "seasonal":
                # Seasonal std data: always has 4 seasons (DJF, MAM, JJA, SON)
                seasons = ["DJF", "MAM", "JJA", "SON"]
                for i, std_data in enumerate(data_std):
                    var = getattr(std_data, "standard_name", "unknown")
                    extra_keys = {"freq": freq, "season": seasons[i], "std": "std", "var": var}
                    if self.region is not None:
                        extra_keys["AQUA_region"] = self.region

                    super().save_netcdf(
                        data=std_data,
                        diagnostic=self.diagnostic_name,
                        diagnostic_product=diagnostic_product,
                        outputdir=outputdir,
                        rebuild=rebuild,
                        extra_keys=extra_keys,
                    )

            elif freq == "longterm":
                var = getattr(data_std, "standard_name", "unknown")

                extra_keys = {"freq": "longterm", "std": "std", "var": var}
                if self.region is not None:
                    extra_keys["AQUA_region"] = self.region

                super().save_netcdf(
                    data=data_std,
                    diagnostic=self.diagnostic_name,
                    diagnostic_product=diagnostic_product,
                    outputdir=outputdir,
                    rebuild=rebuild,
                    extra_keys=extra_keys,
                )

    def compute_dim_mean(self, freq: str, exclude_incomplete: bool = True, center_time: bool = True, box_brd: bool = True):
        """
        Compute the mean of the data. Support for seasonal and longterm means.

        Args:
            freq (str): The frequency to be used ('seasonal' or 'longterm').
            exclude_incomplete (bool): If True, exclude incomplete periods.
            center_time (bool): If True, the time will be centered.
            box_brd (bool,opt): choose if coordinates are comprised or not in area selection.
                                    Default is True
        """
        if self.mean_type == "zonal":
            dims = ["lon"]
        elif self.mean_type == "meridional":
            dims = ["lat"]
        else:
            raise ValueError("Mean type %s not recognized", self.mean_type)

        self.logger.info("Computing %s mean", freq)
        data = self.data
        data_freq = pandas_freq_to_string(xarray_to_pandas_freq(self.data))

        if freq == "seasonal":
            data = self.reader.fldmean(
                data, box_brd=box_brd, lon_limits=self.lon_limits, lat_limits=self.lat_limits, dims=dims
            )
            seasonal_dataset = self.reader.timmean(
                data, freq=freq, exclude_incomplete=exclude_incomplete, center_time=center_time
            )
            seasonal_data = [seasonal_dataset.isel(time=i, drop=True) for i in range(4)]
            for season_data in seasonal_data:
                if self.region is not None:
                    season_data.attrs["AQUA_region"] = self.region
                season_data.attrs["AQUA_mean_type"] = self.mean_type
                season_data.attrs["AQUA_startdate"] = time_to_string(self.startdate)
                season_data.attrs["AQUA_enddate"] = time_to_string(self.enddate)
                season_data.attrs["AQUA_data_freq"] = data_freq
                self.logger.debug("Loading data in memory")
                season_data.load()
                self.logger.debug("Loaded data in memory")

            self.seasonal = seasonal_data

        elif freq == "longterm":
            data = self.reader.timmean(data, freq=None, exclude_incomplete=exclude_incomplete, center_time=center_time)
            data = self.reader.fldmean(
                data, box_brd=box_brd, lon_limits=self.lon_limits, lat_limits=self.lat_limits, dims=dims
            )

            data.attrs["AQUA_startdate"] = time_to_string(self.startdate)
            data.attrs["AQUA_enddate"] = time_to_string(self.enddate)
            data.attrs["AQUA_data_freq"] = data_freq

            if self.region is not None:
                data.attrs["AQUA_region"] = self.region
            data.attrs["AQUA_mean_type"] = self.mean_type

            self.logger.debug("Loading data in memory")
            data.load()
            self.logger.debug("Loaded data in memory")
            self.longterm = data

    def run(
        self,
        var: str,
        formula: bool = False,
        long_name: str = None,
        units: str = None,
        standard_name: str = None,
        std: bool = False,
        freq: list = ["seasonal", "longterm"],
        exclude_incomplete: bool = True,
        center_time: bool = True,
        box_brd: bool = True,
        outputdir: str = "./",
        rebuild: bool = True,
        reader_kwargs: dict = {},
    ):
        """
        Run all the steps necessary for the computation of the LatLonProfiles.

        Args:
            var (str): The variable to be retrieved and computed.
            formula (bool): Whether to use a formula for the variable.
            long_name (str): The long name of the variable.
            units (str): The units of the variable.
            standard_name (str): The standard name of the variable.
            std (bool): Whether to compute the standard deviation.
            freq (list): The frequencies to compute. Options:
                - 'seasonal': Seasonal means (DJF, MAM, JJA, SON)
                - 'longterm': Long-term mean over the entire analysis period
            exclude_incomplete (bool): Whether to exclude incomplete time periods.
            center_time (bool): Whether to center the time coordinate.
            box_brd (bool): Whether to include the box boundaries.
            outputdir (str): The output directory to save the results.
            rebuild (bool): Whether to rebuild existing files.
            reader_kwargs (dict): Additional keyword arguments for the Reader. Default is an empty dictionary.
        """
        self.logger.info("Running LatLonProfiles for %s", var)

        self.retrieve(
            var=var,
            formula=formula,
            long_name=long_name,
            units=units,
            standard_name=standard_name,
            reader_kwargs=reader_kwargs,
        )

        self.logger.info("Mean type set to %s", self.mean_type)

        self.logger.info("Computing temporal means")
        freq = to_list(freq)
        for f in freq:
            self.logger.info(f"Computing {f} mean")
            self.compute_dim_mean(freq=f, exclude_incomplete=exclude_incomplete, center_time=center_time, box_brd=box_brd)

            if std:
                self.logger.info(f"Computing {f} standard deviation")
                self.compute_std(freq=f, exclude_incomplete=exclude_incomplete, center_time=center_time, box_brd=box_brd)

            self.logger.info(f"Saving {f} netcdf file")
            self.save_netcdf(freq=f, outputdir=outputdir, rebuild=rebuild)

        self.logger.info("LatLonProfiles computation completed")
