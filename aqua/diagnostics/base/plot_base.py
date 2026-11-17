"""Base class for the plot classes of the AQUA diagnostics."""

from typing import Optional, Union

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure

from aqua.core.logger import log_configure
from aqua.core.util import DEFAULT_REALIZATION, to_list

from .defaults import SAVE_FORMAT
from .output_saver import OutputSaver
from .title import TitleBuilder


class PlotBase:
    """
    Base class for the plot classes of the diagnostics.
    It holds what every plot class does around its figure: read from the ``AQUA_*`` attributes which
    datasets produced the results, fill a :class:`TitleBuilder` and save through one :class:`OutputSaver`.
    The results, the figure and its description are left to the subclass.
    """

    def __init__(
        self,
        diagnostic_name: str,
        outputdir: str = ".",
        rebuild: bool = True,
        save_format: Union[str, list] = SAVE_FORMAT,
        dpi: int = 300,
        loglevel: str = "WARNING",
    ):
        """
        Initialize the PlotBase class.

        Args:
            diagnostic_name (str): The name of the diagnostic, used in the names of the output files.
            outputdir (str, optional): The output directory. Default is '.'.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
            save_format (str or list, optional): The format(s) of the figures. Default is SAVE_FORMAT.
            dpi (int, optional): The resolution of raster formats. Default is 300.
            loglevel (str, optional): The log level. Default is 'WARNING'.
        """
        self.loglevel = loglevel
        self.logger = log_configure(log_level=loglevel, log_name=type(self).__name__)
        self.diagnostic_name = diagnostic_name
        self.outputdir = outputdir
        self.rebuild = rebuild
        self.save_format = save_format
        self.dpi = dpi

        # Filled by get_data_info(), one entry per dataset and per reference
        self.catalogs, self.models, self.exps, self.realizations = [], [], [], []
        self.startdates, self.enddates = [], []
        self.ref_catalogs, self.ref_models, self.ref_exps = [], [], []
        self.ref_startdates, self.ref_enddates = [], []
        self.region = None
        self.outputsaver = None

    @staticmethod
    def _get_attrs(data: Union[xr.DataArray, xr.Dataset]) -> dict:
        """Return the attributes of a result. Those missing from a Dataset are taken from its first variable."""
        if isinstance(data, xr.Dataset) and data.data_vars:
            return {**data[list(data.data_vars)[0]].attrs, **data.attrs}
        return dict(data.attrs)

    def get_data_info(self, data, ref_data=None):
        """
        Read datasets, region and period from the ``AQUA_*`` attributes and build the OutputSaver.

        One list entry per dataset; the region is the one of the first result.
        Subclasses needing more metadata extend this method through ``super()``.

        Args:
            data (xr.DataArray, xr.Dataset or list): The results to plot, one per dataset.
            ref_data (xr.DataArray, xr.Dataset or list, optional): The results of the references.
        """
        attrs = [self._get_attrs(d) for d in to_list(data)]
        self.catalogs = [a.get("AQUA_catalog") for a in attrs]
        self.models = [a.get("AQUA_model") for a in attrs]
        self.exps = [a.get("AQUA_exp") for a in attrs]
        self.realizations = [a.get("AQUA_realization", DEFAULT_REALIZATION) for a in attrs]
        self.startdates = [a.get("AQUA_startdate") for a in attrs]
        self.enddates = [a.get("AQUA_enddate") for a in attrs]
        self.region = attrs[0].get("AQUA_region") if attrs else None

        ref_attrs = [self._get_attrs(d) for d in to_list(ref_data)]
        self.ref_catalogs = [a.get("AQUA_catalog") for a in ref_attrs]
        self.ref_models = [a.get("AQUA_model") for a in ref_attrs]
        self.ref_exps = [a.get("AQUA_exp") for a in ref_attrs]
        self.ref_startdates = [a.get("AQUA_startdate") for a in ref_attrs]
        self.ref_enddates = [a.get("AQUA_enddate") for a in ref_attrs]

        self.outputsaver = OutputSaver(
            diagnostic=self.diagnostic_name,
            catalog=self.catalogs,
            model=self.models,
            exp=self.exps,
            # A single realization is passed as a value: a one-element list would give ['r1'] in the file name
            realization=OutputSaver.unpack_list(self.realizations),
            catalog_ref=self.ref_catalogs,
            model_ref=self.ref_models,
            exp_ref=self.ref_exps,
            outputdir=self.outputdir,
            loglevel=self.loglevel,
        )

    def set_title(self, max_chars: Optional[int] = None, split_on: Optional[list] = None, **kwargs) -> str:
        """
        Build the title of the figure with :class:`TitleBuilder`.

        The region, models, experiments and references read by :meth:`get_data_info` are filled in.
        The keyword arguments add what is specific to the figure (``diagnostic``, ``variable``, ``catalog``,
        ``startyear``, ...) or replace one of these defaults (``ref_model=None`` leaves the reference out).

        Args:
            max_chars (int, optional): The maximum length of a line, see :meth:`TitleBuilder.generate`.
            split_on (list, optional): The markers where a long title is split, see :meth:`TitleBuilder.generate`.
            **kwargs: Arguments of :class:`TitleBuilder`.

        Returns:
            str: The title of the figure.
        """
        builder_kwargs = {
            "regions": self.region,
            "model": self.models,
            "exp": self.exps,
            "ref_model": self.ref_models,
            "ref_exp": self.ref_exps,
            "loglevel": self.loglevel,
            **kwargs,
        }
        # A dataset without AQUA_* attributes is left out, instead of being written as "None"
        for key, value in builder_kwargs.items():
            if isinstance(value, list) and all(v is None for v in value):
                builder_kwargs[key] = None

        title = TitleBuilder(**builder_kwargs).generate(max_chars=max_chars, split_on=split_on)
        self.logger.debug("Title: %s", title)
        return title

    def save_plot(
        self,
        fig: Figure,
        diagnostic_product: str,
        extra_keys: Optional[dict] = None,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
        rebuild: Optional[bool] = None,
        save_format: Optional[Union[str, list]] = None,
        dpi: Optional[int] = None,
        close: bool = True,
    ):
        """
        Save a figure through the OutputSaver and close it.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            diagnostic_product (str): The product of the diagnostic, used in the file name.
            extra_keys (dict, optional): Additional keys for the file name (e.g. var, region).
            description (str, optional): The caption of the figure, stored in the metadata as ``description``.
            metadata (dict, optional): Additional metadata to store in the file.
            rebuild (bool, optional): If True, overwrite existing files. Default is the value given at initialization.
            save_format (str or list, optional): The format(s) of the figure. Default is the value given at initialization.
            dpi (int, optional): The resolution of raster formats. Default is the value given at initialization.
            close (bool, optional): If True, close the figure, also when the saving fails. Default is True.
        """
        if description is not None:
            metadata = {**(metadata or {}), "description": description}
        try:
            self.outputsaver.save_figure(
                fig,
                diagnostic_product=diagnostic_product,
                extra_keys=extra_keys,
                metadata=metadata,
                extension=self.save_format if save_format is None else save_format,
                rebuild=self.rebuild if rebuild is None else rebuild,
                dpi=self.dpi if dpi is None else dpi,
            )
        finally:
            if close:
                plt.close(fig)
