import pandas as pd
import xarray as xr

from aqua.core.graphics import plot_timeseries
from aqua.diagnostics.base import SAVE_FORMAT, TitleBuilder

# from aqua.logger import log_configure
# from aqua.exceptions import NoDataError
from .base import BaseMixin

xr.set_options(keep_attrs=True)


class PlotEnsembleTimeseries(BaseMixin):
    """
    Class to plot ensemble timeseries data.

    This class inherits from `BaseMixin` and provides functionality to visualize
    ensemble timeseries. It supports plotting the ensemble mean, shading for
    standard deviation (+/- 2x STD), individual ensemble members, and reference
    datasets.
    """

    # TODO: support hourly and daily data

    def __init__(
        self,
        diagnostic_product: str = "EnsembleTimeseries",
        catalog_list: list[str] = None,
        model_list: list[str] = None,
        exp_list: list[str] = None,
        source_list: list[str] = None,
        ref_catalog: str = None,
        ref_model: str = None,
        ref_exp: str = None,
        outputdir="./",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the PlotEnsembleTimeseries class.

        Args:
            diagnostic_product (str, optional): Name of the diagnostic product.
                Defaults to "EnsembleTimeseries". This is used for output file naming.
            catalog_list (list[str], optional): List of catalog names. If None, it is assigned
                to 'None_catalog'. For multiple catalogs, assigned to 'multi-catalog'. Defaults to None.
            model_list (list[str], optional): List of model names. If None, it is assigned
                to 'None_model'. For multiple models, assigned to 'multi-model'. Defaults to None.
            exp_list (list[str], optional): List of experiment names. If None, it is assigned
                to 'None_exp'. For multiple experiments, assigned to 'multi-exp'. Defaults to None.
            source_list (list[str], optional): List of data source names. If None, it is assigned
                to 'None_source'. For multiple sources, assigned to 'multi-source'. Defaults to None.
            ref_catalog (str, optional): Reference catalog name for comparison. Defaults to None.
            ref_model (str, optional): Reference model name for comparison. Defaults to None.
            ref_exp (str, optional): Reference experiment name for comparison. Defaults to None.
            outputdir (str, optional): Directory path for saving plots. Defaults to "./".
            loglevel (str, optional): Logging level. Defaults to "WARNING".
        """
        self.diagnostic_product = diagnostic_product

        self.catalog_list = catalog_list
        self.model_list = model_list
        self.exp_list = exp_list
        self.source_list = source_list
        self.ref_catalog = ref_catalog
        self.ref_model = ref_model
        self.ref_exp = ref_exp
        # TODO: Include region information
        # self.region = region

        self.outputdir = outputdir
        self.loglevel = loglevel

        super().__init__(
            loglevel=self.loglevel,
            diagnostic_product=self.diagnostic_product,
            catalog_list=self.catalog_list,
            model_list=self.model_list,
            exp_list=self.exp_list,
            source_list=self.source_list,
            ref_catalog=self.ref_catalog,
            ref_model=self.ref_model,
            ref_exp=self.ref_exp,
            outputdir=self.outputdir,
        )

    def plot(
        self,
        var=None,
        title=None,
        startdate=None,
        enddate=None,
        hourly_data=None,
        hourly_data_mean=None,
        hourly_data_std=None,
        daily_data=None,
        daily_data_mean=None,
        daily_data_std=None,
        monthly_data=None,
        monthly_data_mean=None,
        monthly_data_std=None,
        annual_data=None,
        annual_data_mean=None,
        annual_data_std=None,
        ref_hourly_data=None,
        ref_daily_data=None,
        ref_monthly_data=None,
        ref_annual_data=None,
        description=None,
        save_format=SAVE_FORMAT,
        dpi=300,
        figure_size=[10, 5],
        plot_ensemble_members=True,
    ):
        """
        Plot the ensemble timeseries including mean, standard deviation, and members.

        This method plots the ensemble mean and a shaded region representing +/- 2
        standard deviations around the ensemble mean. It can optionally plot individual
        ensemble members and reference data for comparison.

        Note:
            - Currently, hourly and daily data plotting are not supported and will be ignored.
            - The standard deviation is computed and plotted point-wise along the mean.
            - Standard deviation is NOT plotted for the reference data.

        Args:
            var (str, optional): Variable name to plot. Defaults to None.
            title (str, optional): Title for the plot. Auto-generated if None. Defaults to None.
            startdate (str, optional): Start date for the timeseries plot. Auto-derived if None.
            Defaults to None.
            enddate (str, optional): End date for the timeseries plot. Auto-derived if None.
            Defaults to None.
            hourly_data (xr.Dataset, optional): Hourly ensemble data. Defaults to None (Ignored).
            hourly_data_mean (xr.Dataset, optional): Hourly ensemble mean. Defaults to None (Ignored).
            hourly_data_std (xr.Dataset, optional): Hourly ensemble std. Defaults to None (Ignored).
            daily_data (xr.Dataset, optional): Daily ensemble data. Defaults to None (Ignored).
            daily_data_mean (xr.Dataset, optional): Daily ensemble mean. Defaults to None (Ignored).
            daily_data_std (xr.Dataset, optional): Daily ensemble std. Defaults to None (Ignored).
            monthly_data (xr.Dataset, optional): Monthly ensemble members concatenated along the ensemble dimension.
            Defaults to None.
            monthly_data_mean (xr.Dataset, optional): Monthly ensemble mean timeseries. Defaults to None.
            monthly_data_std (xr.Dataset, optional): Monthly ensemble standard deviation timeseries.
            Defaults to None.
            annual_data (xr.Dataset, optional): Annual ensemble members concatenated along the ensemble dimension.
            Defaults to None.
            annual_data_mean (xr.Dataset, optional): Annual ensemble mean timeseries. Defaults to None.
            annual_data_std (xr.Dataset, optional): Annual ensemble standard deviation timeseries. Defaults to None.
            ref_hourly_data (xr.Dataset, optional): Reference hourly timeseries. Defaults to None (Ignored).
            ref_daily_data (xr.Dataset, optional): Reference daily timeseries. Defaults to None (Ignored).
            ref_monthly_data (xr.Dataset, optional): Reference monthly timeseries. Defaults to None.
            ref_annual_data (xr.Dataset, optional): Reference annual timeseries. Defaults to None.
            description (str, optional): Description string for saving the plot. Defaults to None.
            save_format (str or list, optional): Format(s) to save the figure in (e.g. 'png', 'pdf', 'svg').
            Defaults to SAVE_FORMAT.
            dpi (int, optional): Resolution for saved figures. Defaults to 300.
            figure_size (list, optional): Figure dimensions [width, height]. Defaults to [10, 5].
            plot_ensemble_members (bool, optional): If True, plots the individual ensemble members.
            Defaults to True.

        Returns:
            tuple or None: A tuple containing the `(matplotlib.figure.Figure, matplotlib.axes.Axes)`
            objects if plotting succeeds, or `None` if no valid data is provided.
        """
        if hourly_data is not None or daily_data is not None:
            self.logger.warning("Hourly and daily data are not yet supported, they will be ignored")

        self.logger.info("Plotting the ensemble timeseries")
        self.logger.info("Assigning label to the given model name")

        if isinstance(self.model, list):
            model_str = " ".join(str(x) for x in self.model)
        else:
            model_str = str(self.model)

        # This dictionary is to check if ensemble mean and std data or reference data mean is provided
        # in order to aviod error in the plotting function below
        plot_data_dict = {
            "ref_monthly_data": ref_monthly_data,
            "ref_annual_data": ref_annual_data,
            "ens_monthly_data": monthly_data_mean,
            "ens_annual_data": annual_data_mean,
            "std_ens_monthly_data": monthly_data_std,
            "std_ens_annual_data": annual_data_std,
        }

        plot_data_dict = {k: v for k, v in plot_data_dict.items() if v is not None and v.sizes.get("time", 0) > 1}

        # Check if the data is provided otherwise return None
        if not plot_data_dict:
            self.logger.warning(f"No data given to plot Ensemble timeseries for {var}")
            return
        elif (plot_data_dict["ens_monthly_data"] is None) and (plot_data_dict["ens_annual_data"] is None):
            self.logger(f"No data given to plot ensemble timeseries for {var}")
            return

        # In case time bounds are not given
        # Derive time bounds; prefer monthly, fall back to annual
        _time_src = monthly_data_mean if monthly_data_mean is not None else annual_data_mean
        if startdate is None:
            startdate = _time_src.time.isel(time=0).values
        if enddate is None:
            enddate = _time_src.time.isel(time=-1).values

        # Select the time interval
        plot_data_dict = {k: v.sel(time=slice(startdate, enddate)) for k, v in plot_data_dict.items()}

        # Converting the dates into string format for plot titles
        startdate = pd.Timestamp(startdate).strftime("%Y-%m-%d")
        enddate = pd.Timestamp(enddate).strftime("%Y-%m-%d")

        if title is None:
            title = TitleBuilder(
                diagnostic="Ensemble analysis", model=self.model, startyear=startdate, endyear=enddate
            ).generate()

        fig, ax = plot_timeseries(
            **plot_data_dict,
            ref_label=self.ref_model,
            ens_label=model_str,
            figsize=figure_size,
            title=title,
            loglevel=self.loglevel,
        )
        # Loop over if need to plot the ensemble members
        if plot_ensemble_members and monthly_data is not None:
            for i in range(0, len(monthly_data[var][:, 0])):
                fig1, ax1 = plot_timeseries(
                    fig=fig,
                    ax=ax,
                    ens_monthly_data=monthly_data_mean,
                    ens_annual_data=annual_data_mean,
                    monthly_data=monthly_data[var][i, :] if monthly_data is not None else None,
                    annual_data=annual_data[var][i, :] if annual_data is not None else None,
                    figsize=figure_size,
                    title=title,
                    loglevel=self.loglevel,
                )
        # Saving plots
        self.logger.debug(f"Saving plots for Ensemble Timeseries for {var}")
        self.save_figure(
            var=var, fig=fig, startdate=startdate, enddate=enddate, description=description, format=save_format, dpi=dpi
        )
        return fig, ax
