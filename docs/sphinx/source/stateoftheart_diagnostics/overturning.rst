.. _overturning:

Overturning Diagnostic
==================================================

Description
-----------

The **Overturning** tool computes the meridional overturning streamfunction of an ocean basin in depth coordinates,
from the northward sea water velocity :math:`v`:

.. math::

    \psi(y, z) = \int_z^0 \int_{x_W}^{x_E} v \, dx \, dz'

in Sverdrup (1 Sv = 10\ :sup:`6` m\ :sup:`3` s\ :sup:`-1`). With this sign convention :math:`\psi` is positive for
northward flow above southward flow, as in the upper cell of the Atlantic Meridional Overturning Circulation (AMOC).
The tool also computes an **overturning index**: the maximum of :math:`\psi` below a minimum depth (500 m by default,
to leave out the wind-driven Ekman cell) at given latitudes, such as 26.5°N, the latitude of the RAPID array.

The zonal integral is computed along latitude rows, so the data must be on a grid made of iso-latitude rows:
regular lon/lat grids and HEALPix grids, where every ring of cells has the same latitude.
Native ocean grids (tripolar, unstructured) must be regridded first (``regrid``).

Two approximations should be kept in mind:

* The velocity interpolated to a regular or HEALPix grid does not conserve volume, so :math:`\psi` does not vanish
  at the bottom. The largest net transport across a section is logged; with ``mass_correction: true`` the net
  transport is removed at each latitude, spread uniformly over the section, as done for the RAPID estimate.
* The layer thickness is derived from the depth of the layer centres (interfaces at the midpoints), and
  :math:`\psi` is given at the lower interface of each layer. Partial bottom cells are not known.

The basin is a region of the regions file. ``atlantic_moc`` is the Atlantic with the marginal seas carrying its
boundary currents (Gulf of Mexico, Caribbean, Labrador Sea, ...), without the Mediterranean, the Nordic Seas and the
Arctic. The section goes from 34°S, the southern tip of Africa, to 70°N by default.

Classes
-------

There is one class for the analysis and one for the plotting:

* **Overturning**: retrieves the meridional velocity of one dataset, computes the streamfunction and the index and
  saves them as NetCDF files. The heavy step, the zonal integral over the basin, reduces the data from grid cells to
  latitude rows lazily (a sparse matrix product on HEALPix grids), so that the full 3D field is read only once.
  The results are ``streamfunction`` (time mean, depth and latitude), ``index_monthly`` and ``index_annual``
  (time and latitude), with their ``AQUA_*`` provenance attributes.
  They can be read back with ``load()``, which makes plot-only runs possible.

* **PlotOverturning**: draws the time series of the index at each latitude (monthly and annual, optionally with a
  reference) and the latitude-depth section of the time-mean streamfunction for each dataset, and saves them through
  ``OutputSaver``. It never accesses the data.

.. note::

    The minimum amount of data is ``MINIMUM_MONTHS_REQUIRED`` (12 months: one annual value of the index).

File structure
--------------

* The tool is located in the ``aqua/diagnostics/overturning/`` directory, which contains the source code and the
  command line interface (CLI) script.
* A template configuration file is available at ``aqua/diagnostics/templates/collections/config-overturning.yaml``.
* Region definitions, including the ``atlantic_moc`` basin, are available in
  ``aqua/diagnostics/config/definitions/regions.yaml``.

The source code directory contains the following files:

* ``overturning.py``: the analysis class ``Overturning``;
* ``plot_overturning.py``: the plot class ``PlotOverturning``;
* ``cli_overturning.py``: the command line interface.

Input variables and datasets
----------------------------

* ``von`` (northward sea water velocity, GRIB parameter 263505 ``avg_von``, CMOR ``vo``), in m s\ :sup:`-1`, on depth
  levels (``depth``, m, positive down).

For the ClimateDT experiments the 3D ocean sources on HEALPix (e.g. ``monthly-hpz7-o3d``) can be used directly.

One dataset is analysed at a time. A reference is optional: a dataset with the meridional velocity is analysed with
the same class and its index is drawn together with the dataset.

Basic usage
-----------

.. code-block:: python

    from aqua.diagnostics import Overturning, PlotOverturning

    amoc = Overturning(
        catalog='climatedt-gen2',
        model='IFS-NEMO-10km',
        exp='baseline-cont',
        source='monthly-hpz7-o3d',
        startdate='1990-01-01',
        enddate='1999-12-31',
        region='atlantic_moc',
        loglevel='INFO'
    )
    amoc.run(var='von', latitudes=[26.5, 45], min_depth=500, lat_range=[-34, 70], outputdir='./')

    PlotOverturning(
        streamfunction=amoc.streamfunction,
        index_monthly=amoc.index_monthly,
        index_annual=amoc.index_annual,
        outputdir='./'
    ).plot()

.. note::

    If not specified otherwise, plots are saved in PNG, PDF and SVG format in the output directory.

CLI usage
---------

The tool can be run from the command line interface (CLI) with the following command:

.. code-block:: bash

    cd $AQUA/aqua/diagnostics/overturning
    python cli_overturning.py --config <path_to_config_file>

The CLI accepts the arguments common to all the tools, see :ref:`diagnostics-cli-arguments`.

Configuration file structure
----------------------------

Most of the settings are common to all the tools (see :ref:`diagnostics-configuration-files`).
Here we describe only the specific settings of the Overturning tool.

* ``overturning``: a block (nested in the ``diagnostics`` block) containing the options of the tool.

    * ``run``: enable/disable the tool.
    * ``diagnostic_name``: prefix of every output file. ``overturning`` by default, the collection name when the tool
      runs inside a collection.
    * ``variables``: the northward velocity, ``['von']``.
    * ``params``: computation options, with a ``default`` entry and per-variable overrides:

        * ``regions``: the basins (``atlantic_moc``; ``null`` is the global ocean);
        * ``latitudes``: the latitudes of the index;
        * ``min_depth``: the index is the maximum of :math:`\psi` below this depth (m);
        * ``lat_range``: the southern and northern limits of the section;
        * ``mass_correction``: remove the net transport across each section;
        * ``vert_coord``: the name of the vertical coordinate.

    * ``plot_params``: ``vmin``, ``vmax`` and ``nlevels`` of the section (Sv).

* ``references``: optional block (top level) with a reference dataset with the meridional velocity.
  Only the first one is used. It keeps its own ``regrid``, ``startdate`` and ``enddate``.

.. code-block:: yaml

    diagnostics:
      overturning:
        run: true
        diagnostic_name: 'overturning'
        variables: ['von']
        params:
          default:
            regions: ['atlantic_moc']
            latitudes: [26.5, 45.0]
            min_depth: 500
            lat_range: [-34, 70]
            mass_correction: false
        plot_params:
          default:
            vmin: -20
            vmax: 20
            nlevels: 21

Output
------

The tool produces the following outputs, named ``<diagnostic_name>.<product>.<catalog>.<model>.<exp>.<realization>...``:

* ``netcdf/``: the products ``streamfunction``, ``index_monthly`` and ``index_annual``, with the ``AQUA_*`` attributes;
* ``png/``, ``pdf/``, ``svg/``: the figures ``index`` (one per latitude) and ``streamfunction`` (one per dataset),
  with the caption stored in the metadata.

With ``plot_only: true`` in the ``output`` block the figures are produced again from the NetCDF files,
without recomputing (see :ref:`diagnostics-plot-only`).

Observations
------------

No reference is configured yet. The natural one for the index at 26.5°N is the RAPID-MOCHA time series
(April 2004 onwards), which is not yet available in the AQUA catalogs.

References
----------

* Danabasoglu, G., et al. (2014). North Atlantic simulations in Coordinated Ocean-ice Reference Experiments phase II
  (CORE-II). Part I: Mean states. *Ocean Modelling*, 73, 76-107. doi:10.1016/j.ocemod.2013.10.005
* Griffies, S. M., et al. (2016). OMIP contribution to CMIP6: experimental and diagnostic protocol for the physical
  component of the Ocean Model Intercomparison Project. *Geosci. Model Dev.*, 9, 3231-3296. doi:10.5194/gmd-9-3231-2016
* McCarthy, G. D., et al. (2015). Measuring the Atlantic Meridional Overturning Circulation at 26°N.
  *Progress in Oceanography*, 130, 91-111. doi:10.1016/j.pocean.2014.10.006
* Frajka-Williams, E., et al. (2019). Atlantic Meridional Overturning Circulation: Observed Transport and Variability.
  *Frontiers in Marine Science*, 6, 260. doi:10.3389/fmars.2019.00260

Authors and contributors
------------------------

This tool is maintained by Paolo Davini.
Contributions are welcome, please open an issue or a pull request.
For questions or suggestions, contact the AQUA team or the maintainer.

Detailed API
------------

This section provides a detailed reference for the Application Programming Interface (API) of the ``overturning`` tool,
generated from the function docstrings.

.. automodule:: aqua.diagnostics.overturning
    :members:
    :undoc-members:
    :show-inheritance:
