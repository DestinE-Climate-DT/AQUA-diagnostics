"""Detect and track tropical cyclones using lazily retrieved atmospheric fields.

Example:
    tropical = TCs(tdict=config, streaming=True, stream_step="1D")
    while tropical.data_retrieve():
        tropical.detect_nodes_zoomin()
"""

import copy
import os

import numpy as np
import pandas as pd
import xarray as xr

from aqua import Reader
from aqua.core.logger import log_configure, log_history

from .detect_nodes import DetectNodes
from .stitch_nodes import StitchNodes
from .tools.tcs_utils import lonlatbox

xr.set_options(keep_attrs=True)


class TCs(DetectNodes, StitchNodes):
    """
    This class contains all methods related to the TCs (Tropical Cyclones)
    diagnostic based on tempest-estremes tracking. It provides two main functions -
    "detect_nodes_zoomin" and "stitch_nodes_zoomin" - for detecting the nodes of TCs and
    producing tracks of selected variables stored in netcdf files, respectively.
    """

    def __init__(
        self,
        tdict=None,
        paths=None,
        model=None,
        exp="None",
        source2d="None",
        source3d="None",
        boxdim=10,
        lowgrid="r100",
        highgrid="r010",
        var2store=None,
        streaming=False,
        frequency="6h",
        startdate=None,
        enddate=None,
        stream_step=1,
        stream_unit="days",
        stream_startdate=None,
        loglevel="INFO",
        orography=False,
        engine="fdb",
        nproc=1,
        write_fullres=False,
    ):
        """
        Constructor method that initializes the class attributes based on the
        input arguments or tdict dictionary.

        Args:
            tdict (dict): A dictionary containing various configurations for the TCs diagnostic.
                          If tdict is provided, the configurations will be loaded from it,
                          otherwise the configurations will be set based on the input arguments.
            paths (dict): A dictionary containing file paths for input and output files.
            model (str): The name of the weather model to be used for the TCs diagnostic. Default is "IFS".
            exp (str): The name of the weather model experiment to be used for the TCs diagnostic. Default is "tco2559-ng5".
            boxdim (int): The size of the box centred over the TCs centres in the detect_nodes_zoomin method. Default is 10.
            lowgrid (str): The low-resolution grid used for detecting the nodes of TCs. Default is 'r100'.
            highgrid (str): The high-resolution grid used for detecting the nodes of TCs. Default is 'r010'.
            var2store (list): The list of variables to be stored in netcdf files. Default is None.
            streaming (bool): A flag indicating whether the TCs diagnostic is performed in streaming mode. Default is False.
            frequency (str): The time frequency for processing the TCs diagnostic. Default is '6h'.
            startdate (str): The start date for processing the TCs diagnostic.
            enddate (str): The end date for processing the TCs diagnostic.
            engine (str): The engine to use for reading the data. Default is 'fdb'.
            stream_step (int or str): Processing window in whole days, e.g. 1 or "1D". Default is 1.
            stream_unit (str): The unit of stream_step in streaming mode. Default is 'days'.
            stream_startdate (str): The start date for processing the TCs diagnostic in streaming mode.
            loglevel (str): The logging level for the TCs diagnostic. Default is 'INFO'.
            write_fullres (bool): A flag indicating whether to write full-resolution output files. Default is False.

        Returns:
            A TCs object
        """

        self.logger = log_configure(loglevel, "TCs")
        self.loglevel = loglevel

        self.nproc = nproc

        if tdict is not None:
            tdict = copy.deepcopy(tdict)
            self.paths = tdict["paths"]
            self.dataset = tdict.get("datasets", [{}])[0]
            self.catalog = self.dataset.get("catalog", None)
            self.engine = self.dataset.get("engine", "fdb")
            self.reader_kwargs = self.dataset.get("reader_kwargs", {})
            if not self.reader_kwargs:  # make sure it is an empty dictionary
                self.reader_kwargs = {}
            self.model = self.dataset["model"]
            self.exp = self.dataset["exp"]
            self.source2d = self.dataset["source2d"]
            self.source3d = self.dataset["source3d"]
            self.boxdim = tdict["detect"]["boxdim"]
            self.lowgrid = tdict["grids"]["lowgrid"]
            self.highgrid = tdict["grids"]["highgrid"]
            self.var2store = tdict["varlist"]
            self.frequency = tdict["time"]["frequency"]
            self.startdate = tdict["time"]["startdate"]
            self.enddate = tdict["time"]["enddate"]
            self.orography = orography
            self.write_fullres = tdict["detect"].get("write_fullres", False)
            if self.orography:
                self.orography_file = tdict["orography"]["file_path"]
                self.source_oro = tdict["orography"].get("source_oro", None)
                self.var_oro = tdict["orography"].get("var_oro", None)
                if not (self.source_oro and self.var_oro) and not self.orography_file:
                    raise ValueError("To use orography you need to define source_oro and var_oro or orography_file")
        else:
            if paths is None:
                raise ValueError("Without paths defined you cannot go anywhere!")
            else:
                self.paths = paths
            if startdate is None or enddate is None:
                raise ValueError("Define startdate and/or enddate")
            self.model = model
            self.exp = exp
            self.source2d = source2d
            self.source3d = source3d
            self.boxdim = boxdim
            self.lowgrid = lowgrid
            self.highgrid = highgrid
            self.var2store = var2store
            self.frequency = frequency
            self.startdate = startdate
            self.enddate = enddate
            self.write_fullres = write_fullres

        self.orog = None

        self.streaming = streaming
        self.stream_step = stream_step
        self.stream_units = stream_unit
        self.stream_startdate = stream_startdate or self.startdate
        self._initial_stream_startdate = self.stream_startdate
        if self.streaming:
            self._stream_step = (
                pd.to_timedelta(stream_step)
                if isinstance(stream_step, str)
                else pd.to_timedelta(stream_step, unit=stream_unit)
            )
            if pd.isna(self._stream_step) or self._stream_step <= pd.Timedelta(0) or self._stream_step % pd.Timedelta(days=1):
                raise ValueError("Stream step must be a positive whole number of days (e.g. '1D')")

        # create directory structure
        self.paths["tmpdir"] = os.path.join(self.paths["tmpdir"], self.model, self.exp)
        self.paths["fulldir"] = os.path.join(self.paths["fulldir"], self.model, self.exp)
        self.paths["trackdir"] = os.path.join(self.paths["trackdir"], self.model, self.exp)

        for path in self.paths:
            os.makedirs(self.paths[path], exist_ok=True)

        self.catalog_init()

    def loop_streaming(self, tdict):
        """
        Run DetectNodes and StitchNodes over lazy processing windows.

        Atmospheric datasets are retrieved once and sliced for each window.

        Args:
            self: The current object instance.
            tdict: A dictionary containing various parameters for streaming and stitching.

        Returns:
            None
        """

        # parameters for stitch nodes (to save tracks of selected variables in netcdf)
        n_days_stitch = tdict["stitch"]["n_days_freq"] + 2 * tdict["stitch"]["n_days_ext"]
        last_run_stitch = None

        while self.data_retrieve():
            if last_run_stitch is None:
                last_run_stitch = self.stream_startdate
            self.logger.warning(
                "Streaming from %s to %s", pd.to_datetime(self.stream_startdate), pd.to_datetime(self.stream_enddate)
            )
            self.detect_nodes_zoomin()

            # add one hour since time ends at 23
            dayspassed = (
                np.datetime64(self.stream_enddate) + np.timedelta64(1, "h") - np.datetime64(last_run_stitch)
            ) / np.timedelta64(1, "D")

            # call Tempest StitchNodes every n_days_freq days time period and save TCs tracks in a netcdf file

            if dayspassed >= n_days_stitch:
                end_run_stitch = np.datetime64(last_run_stitch) + np.timedelta64(tdict["stitch"]["n_days_freq"], "D")
                self.logger.warning(
                    "Running stitch nodes from %s to %s", pd.to_datetime(last_run_stitch), pd.to_datetime(end_run_stitch)
                )
                self.stitch_nodes_zoomin(
                    startdate=pd.to_datetime(last_run_stitch),
                    enddate=pd.to_datetime(end_run_stitch),
                    n_days_freq=tdict["stitch"]["n_days_freq"],
                    n_days_ext=tdict["stitch"]["n_days_ext"],
                )
                last_run_stitch = copy.deepcopy(end_run_stitch)

    def catalog_init(self):
        """
        Initialize the catalog for data retrieval based on the specified model.

        Args:
        - self: Reference to the current instance of the class.

        Returns:
            None

        Raises:
        - Exception: If the specified model is not supported.
        """
        self.logger.warning("Model %s - Exp: %s", self.model, self.exp)

        if self.streaming:
            self.logger.warning(
                "Initialised streaming for %s %s starting on %s",
                self.stream_step,
                self.stream_units,
                pd.to_datetime(self.stream_startdate),
            )

        self.varlist2d = ["msl", "10u", "10v"]
        self.varlist3d = ["z"]

        self.reader2d = Reader(
            model=self.model,
            exp=self.exp,
            source=self.source2d,
            catalog=self.catalog,
            regrid=self.lowgrid,
            loglevel=self.loglevel,
            startdate=self.startdate,
            enddate=self.enddate,
            engine=self.engine,
            **self.reader_kwargs,
        )
        self.reader3d = Reader(
            model=self.model,
            exp=self.exp,
            source=self.source3d,
            catalog=self.catalog,
            regrid=self.lowgrid,
            loglevel=self.loglevel,
            startdate=self.startdate,
            enddate=self.enddate,
            engine=self.engine,
            **self.reader_kwargs,
        )

        if self.write_fullres:
            self.reader_fullres = Reader(
                model=self.model,
                exp=self.exp,
                source=self.source2d,
                catalog=self.catalog,
                regrid=self.highgrid,
                loglevel=self.loglevel,
                startdate=self.startdate,
                enddate=self.enddate,
                engine=self.engine,
                **self.reader_kwargs,
            )

        if self.orography:
            if not self.orography_file:
                self.reader_oro = Reader(
                    model=self.model,
                    exp=self.exp,
                    source=self.source_oro,
                    catalog=self.catalog,
                    regrid=self.lowgrid,
                    loglevel=self.loglevel,
                    engine=self.engine,
                )
                self.orog = self.reader_oro.retrieve(var=self.var_oro).isel(time=0)
                self.logger.debug("Orography retrieved from catalog source %s", self.source_oro)
            else:
                self.orog = xr.open_dataset(self.orography_file)
                self.logger.debug("Orography retrieved from file %s", self.orography_file)

            # in this diagnostic we need orography to be called zs
            rename_dict = {
                k: v
                for k, v in {"z": "zs", "oromea": "zs", "orog": "zs", "longitude": "lon", "latitude": "lat"}.items()
                if k in self.orog
            }
            if rename_dict:
                self.orog = self.orog.rename(rename_dict)

    def data_retrieve(self, reset_stream=False):
        """Retrieve fields once and select the next lazy processing window.

        Args:
            reset_stream (bool, optional): Restart window iteration over the cached
                datasets without retrieving them again. Defaults to False.

        Returns:
            bool: True when data is available, False when the stream is exhausted
                or either atmospheric dataset is missing or empty.
        """
        if not hasattr(self, "_data2d"):
            self._data2d = self.reader2d.retrieve(var=self.varlist2d)
            self._data3d = self.reader3d.retrieve(var=self.varlist3d)
            if self._data3d is not None:
                self._data3d = self._data3d.sel(plev=[30000, 50000], method="nearest")
            if self.write_fullres:
                self._fullres = self.reader_fullres.retrieve(var=self.var2store)

        if self._data2d is None or self._data3d is None:
            return False
        if not self._data2d.sizes.get("time", 0) or not self._data3d.sizes.get("time", 0):
            return False

        if not self.streaming:
            self.data2d = self._data2d.aqua.set_default(self.reader2d)
            self.data3d = self._data3d.aqua.set_default(self.reader3d)
            if self.write_fullres:
                self.fullres = self._fullres.aqua.set_default(self.reader_fullres)
            return True

        if reset_stream or not hasattr(self, "_stream"):
            data = self._data2d.sel(time=slice(self._initial_stream_startdate, self.enddate))
            self._stream = iter(data.resample(time=self._stream_step, origin="start")) if data.sizes["time"] else iter(())

        try:
            _, window = next(self._stream)
        except StopIteration:
            return False

        times = window.time
        self.stream_startdate = window.indexes["time"][0]
        self.stream_enddate = window.indexes["time"][-1]
        fields = [("data2d", self.reader2d), ("data3d", self.reader3d)]
        if self.write_fullres:
            fields.append(("fullres", self.reader_fullres))
        for name, reader in fields:
            data = getattr(self, "_" + name).sel(time=times)
            log_history(data, "Selected Tropical Cyclones processing window")
            setattr(self, name, data.aqua.set_default(reader))
        return True

    def store_fullres_field(self, xfield, nodes):
        """
        Create xarray object that keep only the values of a field around the TC nodes

        Args:
            mfield: xarray object (set to 0 at the first timestep of a loop)
            xfield: xarray object to be concatenated with mfield
            nodes: dictionary with date, lon, lat of the TCs centres
            boxdim: length of the lat lon box (required for lonlatbox funct)

        Returns:
            outfield: xarray object with values only in the box around the TC nodes centres for all time steps
        """

        mask = xfield * 0
        for k in range(0, len(nodes["lon"])):
            # add safe condition: keep only data between 50S and 50N
            # if (float(nodes['lat'][k]) > -50) and (float(nodes['lat'][k]) < 50):
            box = lonlatbox(nodes["lon"][k], nodes["lat"][k], self.boxdim)
            mask = mask + xr.where(
                (xfield.lon > box[0]) & (xfield.lon < box[1]) & (xfield.lat > box[2]) & (xfield.lat < box[3]), True, False
            )

        outfield = xfield.where(mask > 0)

        # if isinstance(mfield, xr.DataArray):
        #    outfield = xr.concat([mfield, outfield], dim = 'time')

        return outfield
