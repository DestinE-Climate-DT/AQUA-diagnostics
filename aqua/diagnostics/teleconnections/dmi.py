from aqua.core.exceptions import NotEnoughDataError
from aqua.core.logger import log_configure
from aqua.core.util import time_to_string

from .base import BaseMixin


class DMI(BaseMixin):
    """
    Dipole Mode Index (DMI) calculation class.
    This class is used to calculate the DMI index from a given dataset.
    It inherits from the BaseMixin class and implements the necessary methods
    to calculate the DMI index.
    """

    MINIMUM_MONTHS_REQUIRED = 24

    def __init__(
        self,
        catalog: str = None,
        model: str = None,
        exp: str = None,
        source: str = None,
        regrid: str = None,
        startdate: str = None,
        enddate: str = None,
        configdir: str = None,
        definition: str = "teleconnections-destine",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the DMI class.

        Args:
            catalog (str): Catalog name.
            model (str): Model name.
            exp (str): Experiment name.
            source (str): Source name.
            regrid (str): Regrid method.
            startdate (str): Start date for data retrieval.
            enddate (str): End date for data retrieval.
            configdir (str): Configuration directory. Default is the installation directory.
            definition (str): definition filename. Default is 'teleconnections-destine'.
            loglevel (str): Logging level. Default is 'WARNING'.
        """
        super().__init__(
            telecname="DMI",
            catalog=catalog,
            model=model,
            exp=exp,
            source=source,
            regrid=regrid,
            startdate=startdate,
            enddate=enddate,
            configdir=configdir,
            definition=definition,
            loglevel=loglevel,
        )
        self.logger = log_configure(log_name="DMI", log_level=loglevel)

        self.var = self.definition.get("field")

    def retrieve(self, reader_kwargs: dict = {}) -> None:
        """
        Retrieve the data for DMI index calculation.

        Args:
            reader_kwargs (dict): Additional keyword arguments for the data reader.
        """
        self.logger.info("Retrieving data for DMI index calculation.")
        super().retrieve(var=self.var, reader_kwargs=reader_kwargs, months_required=self.MINIMUM_MONTHS_REQUIRED)

        self.data = self.reader.timmean(self.data, freq="MS")

    def compute_index(self, rebuild: bool = False) -> None:
        """
        Compute the DMI index.

        Args:
            rebuild (bool): Whether to rebuild the index if it already exists. Default is False.
        """
        if self.index is not None and not rebuild:
            self.logger.info("DMI index already calculated, skipping.")
            return
        if self.data is None:
            raise NotEnoughDataError("Data not retrieved")

        # Extract coordinates from definitions and convert longitudes to 0-360 if necessary
        weio_coords = self.definition.get("weio")
        eeio_coords = self.definition.get("eeio")

        # if self.data[self.var].lon.min() >= 0: # we need to convert to 0-360 if the data is in -180,180
        # weio_lon = [lon_to_360(weio_coords["lon_limits"][i]) for i in range(2)]
        # eeio_lon = [lon_to_360(eeio_coords["lon_limits"][i]) for i in range(2)]

        self.logger.debug(f"WEIO coordinates: {weio_coords}")
        self.logger.debug(f"EEIO coordinates: {eeio_coords}")

        # Evaluation of the individual monthly climatology (globally) and then on the two regions separately
        clim = self.data[self.var].groupby("time.month").mean(dim="time")

        # For the two regions, evaluate the anomalies with respect to the global climatology month by month
        grouped_data = self.data[self.var].groupby("time.month")

        anom_weio = self.reader.fldmean(
            grouped_data - clim, lat_limits=weio_coords["lat_limits"], lon_limits=weio_coords["lon_limits"]
        )
        anom_eeio = self.reader.fldmean(
            grouped_data - clim, lat_limits=eeio_coords["lat_limits"], lon_limits=eeio_coords["lon_limits"]
        )

        # DMI index is the difference between the two anomalies
        dmi = anom_weio - anom_eeio
        dmi = dmi.rename("DMI index")

        dmi.attrs["long_name"] = f"{self.telecname} index"

        dmi.attrs["AQUA_startdate"] = time_to_string(self.startdate)
        dmi.attrs["AQUA_enddate"] = time_to_string(self.enddate)
        dmi.load()

        self.logger.debug("DMI index calculated")
        self.index = dmi
