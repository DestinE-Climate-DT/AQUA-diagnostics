.. _analysis_collections:

Analysis collections: kind, jinja and legacy files
===================================================

``aqua analysis`` (see :ref:`aqua_analysis`) runs the diagnostics through *collections*.
A collection is a set of configuration files, grouped by the physical domain they check
(``atmosphere2d``, ``ocean3d``, ``radiation_toa``, ...).
Each configuration file is read by one or more diagnostic CLIs, and one CLI can read more than one file:

- ``config-atmosphere2d-surface`` has a ``biases``, a ``seasonalcycles`` and a ``timeseries`` section.
  It is given to the ``biases`` CLI and to the ``timeseries`` CLI, and each CLI runs the sections it knows.
- The ``biases`` CLI of ``atmosphere2d`` receives two files (``-temperature`` and ``-surface``) and runs once for each of them.

The list of the CLIs to run, and which file(s) each of them reads, is written in the analysis configuration file
(``config.aqua-analysis.yaml``).
This page explains how the configuration files are defined, how they are used, how to modify them and how to create a new collection.

All files live in the ``aqua/diagnostics/config`` folder (installed in ``$AQUA_CONFIG``):

.. code-block:: text

    config/
    ├── analysis/
    │   ├── config.aqua-analysis.yaml          # which CLIs to run and which files they read (jinja mode)
    │   ├── config.aqua-analysis-legacy.yaml   # same, with the legacy files
    │   └── climatedt-experiment-kind.yaml     # kind file: the values filled in the jinja files
    └── collections/
        ├── jinja/<collection>/config-<collection>-<quantity>.j2
        └── legacy/<collection>/config-<collection>-<quantity>.yaml

The jinja and legacy files of a collection have the **same names** (only the extension changes).

.. note::

    The generic configuration file of each diagnostic, with all its options and short comments,
    is in ``aqua/diagnostics/templates/collections`` (e.g. ``config-biases.yaml``).
    It is the best starting point to write a new collection file.

Concepts
--------

Kind file
^^^^^^^^^

``climatedt-experiment-kind.yaml`` is a YAML file where each top-level key is an *experiment kind*
(e.g. ``climatedt-hist-cmip6``, ``highresmip-hist-cmip7``, ``storyline``).
A kind describes which reference datasets, which date ranges and which general settings
are appropriate for a family of experiments.
The kind file is declared in the analysis configuration file (``job.experiment_kind``)
and the kind to use is chosen from the command line.

The kind ``default`` is mandatory and contains *every* definition used by the jinja files.
All the other kinds only list the keys they want to change:
aqua-core deep-merges them on top of ``default`` before rendering.
The content of ``default`` is:

``regrid``
    Target grid of the reference datasets (model datasets are regridded through the CLI ``--regrid`` option).

``loglevel``
    Value of ``setup.loglevel`` in the jinja files.

``output``
    Output settings shared by all the jinja files: ``rebuild``, ``save_netcdf``, ``save_format`` and ``dpi``.
    Only ``outputdir`` is written in each file.

``sources``
    Every observational dataset available, listed once as ``{catalog, model, exp, source}``
    with a YAML anchor of the same name (``era5``, ``ceres_toa``, ...).
    The jinja files never read this block: it only exists to define the datasets in one place
    and reuse them through anchors. It must live inside ``default`` because aqua-core treats every top-level key of the
    file as a kind, so a top-level ``sources`` would become a selectable ``--kind sources``.

``references``
    One *slot* for each jinja file, pointing to one of the anchors of ``sources``.
    This is the block the jinja files read to know which dataset to compare with
    (see :ref:`analysis_collections_naming` for the slot names).
    A kind swaps the reference of a single file by pointing its slot to another anchor.

``period``
    Date ranges of the references, in three independent blocks:
    ``standard`` (used by most analyses), ``radiation`` (radiation analyses against CERES,
    which only starts in March 2000) and ``timeseries``.
    Each block has ``startdate``, ``enddate`` and the ``std_startdate``, ``std_enddate`` used to compute the
    standard deviation of the reference.

The kind name is also passed to the jinja files as ``kind``. This name is reserved and cannot be defined in the kind file.

.. note::

    The deep merge only completes a kind starting from ``default``. It acts *inside* each ``period`` block
    (a kind can override only ``period.standard.enddate`` and keep the other three keys),
    but it does **not** act between the blocks: ``period.timeseries`` and ``period.radiation`` never inherit from
    ``period.standard``, even if the latter is the one named "standard".
    A kind that changes the dates of all the analyses must therefore override all the blocks.
    The HighResMIP kinds do this with a YAML anchor shared by the three blocks.
    Lists (e.g. ``output.save_format``) are replaced as a whole, not merged.

Jinja files
^^^^^^^^^^^

A jinja file (``.j2``) is a configuration file in which the values that depend on the experiment kind
are replaced by jinja expressions reading the resolved kind, for instance:

.. code-block:: yaml

    references:
      - catalog: '{{ references.atmosphere2d_surface.catalog }}'
        model: '{{ references.atmosphere2d_surface.model }}'
        exp: '{{ references.atmosphere2d_surface.exp }}'
        source: '{{ references.atmosphere2d_surface.source }}'
        regrid: '{{ regrid }}'
        startdate: '{{ period.standard.startdate }}'
        enddate: '{{ period.standard.enddate }}'

    setup:
      loglevel: '{{ loglevel }}'

    output:
      outputdir: "./"
      rebuild: {{ output.rebuild }}
      save_netcdf: {{ output.save_netcdf }}
      save_format: {{ output.save_format }}
      dpi: {{ output.dpi }}

The rendering is strict: referring to a key that the resolved kind does not define is an error,
so a file cannot silently use a missing value.
The rendered files are written in a temporary folder and passed to the diagnostic CLIs.
Values that do not depend on the kind (variables, plot options, ``outputdir``, ...) are written directly in the file.

Legacy files
^^^^^^^^^^^^

A legacy file (``.yaml``) is the explicit version of the **default jinja case**:
it has the same name and the same content as the jinja file rendered with the ``default`` kind,
with all values written out (dataset names, dates, regrid, output settings).
There is no kind file, so a change in a reference dataset or in a date range has to be repeated in every file.
Legacy files are not touched by the kind mechanism.

The test ``tests/config/test_analysis_config_contract.py`` checks that every legacy file is equal to the
rendering of the corresponding jinja file with the ``default`` kind.
When a jinja file or the ``default`` kind changes, the legacy file must be updated with the same change.

Using the files
---------------

Jinja mode
^^^^^^^^^^

``config.aqua-analysis.yaml`` is the default analysis configuration. It points to the kind file
and, for each CLI, to the jinja file(s) to read:

.. code-block:: yaml

    job:
      experiment_kind: "${AQUA_CONFIG}/analysis/climatedt-experiment-kind.yaml"

    diagnostics:
      atmosphere2d:
        biases:
          config: ["${AQUA_CONFIG}/collections/jinja/atmosphere2d/config-atmosphere2d-temperature.j2",
                   "${AQUA_CONFIG}/collections/jinja/atmosphere2d/config-atmosphere2d-surface.j2"]
        timeseries:
          config: ["${AQUA_CONFIG}/collections/jinja/atmosphere2d/config-atmosphere2d-temperature.j2",
                   "${AQUA_CONFIG}/collections/jinja/atmosphere2d/config-atmosphere2d-surface.j2"]

A list of files runs the CLI once for each file.
The analysis file is given with ``--config`` and the kind with ``--kind``:

.. code-block:: bash

    aqua analysis --config $AQUA_CONFIG/analysis/config.aqua-analysis.yaml --kind climatedt-hist-cmip7 <options>

Without ``--kind`` the ``default`` kind is used, without ``--config`` the default analysis file is used.
An unknown kind raises an error listing the available ones.
The kind ``scenario``, available in previous versions, has been replaced by ``climatedt-projections-cmip6``.

Legacy mode
^^^^^^^^^^^

``config.aqua-analysis-legacy.yaml`` has the same structure but points to the files in ``collections/legacy``
and does not define ``experiment_kind``, so no templating is performed and ``--kind`` has no effect (aqua-core only logs a warning):

.. code-block:: bash

    aqua analysis --config $AQUA_CONFIG/analysis/config.aqua-analysis-legacy.yaml <options>

Modifying existing files
------------------------

Change a reference dataset or a date range for a family of experiments
    Edit (or add) the kind in the kind file. Override only the keys that change.
    For instance, to use ERA5 sea ice concentration in a new kind:

    .. code-block:: yaml

        my-new-kind:
          references:
            seaice_concentration_nh: *era5
            seaice_concentration_sh: *era5
          period:
            standard:
              startdate: "2000-01-01"

    Remember that ``period.radiation`` and ``period.timeseries`` do not follow ``period.standard``.

Add a new observational dataset
    Add it to ``default.sources`` with its own anchor, then point the slots of the kinds to it.

Change a diagnostic option (variables, plots, ...)
    Edit the jinja file **and** the legacy file with the same name, so that they stay equivalent
    (the test mentioned above fails otherwise).

Run a different set of diagnostics
    Edit the ``run`` and ``diagnostics`` sections of the analysis configuration file
    (see :ref:`aqua_analysis`).

Creating a new collection
-------------------------

The example creates a collection ``snow``, with one file ``config-snow-cover`` that is read by two CLIs:
``biases`` (maps of the difference with a reference) and ``timeseries``.
The reference is ERA5 snow depth. Replace names and variables with yours.

**1. Choose the names.**
The collection is ``snow``, the quantity is ``cover``: the file is ``config-snow-cover`` (see :ref:`analysis_collections_naming`).
The reference slot is ``snow_cover``.

**2. Write the file.**
Start from the generic file of the diagnostic (``aqua/diagnostics/templates/collections/config-biases.yaml``)
and keep the sections of the CLIs you need. In jinja mode, the file is
``config/collections/jinja/snow/config-snow-cover.j2``:

.. code-block:: yaml
    :caption: config/collections/jinja/snow/config-snow-cover.j2

    datasets:
      - catalog: null
        model: null
        exp: null
        source: null
        regrid: null
        startdate: null
        enddate: null

    references:
      - catalog: '{{ references.snow_cover.catalog }}'
        model: '{{ references.snow_cover.model }}'
        exp: '{{ references.snow_cover.exp }}'
        source: '{{ references.snow_cover.source }}'
        regrid: '{{ regrid }}'
        startdate: '{{ period.standard.startdate }}'
        enddate: '{{ period.standard.enddate }}'

    setup:
      loglevel: '{{ loglevel }}'

    output:
      outputdir: "./"
      rebuild: {{ output.rebuild }}
      save_netcdf: {{ output.save_netcdf }}
      save_format: {{ output.save_format }}
      dpi: {{ output.dpi }}

    diagnostics:
      biases:
        run: true
        diagnostic_name: 'snow'
        variables: ['sd']
        params:
          default:
            seasons: true
            seasons_stat: 'mean'
            vertical: false
        plot_params:
          default:
            projection: 'robinson'
            projection_params: {}
            show_stats: true

      timeseries:
        run: true
        diagnostic_name: 'snow'
        variables: ['sd']
        center_time: true
        params:
          default:
            hourly: false
            daily: false
            monthly: true
            annual: true
            std_startdate: '{{ period.timeseries.std_startdate }}'
            std_enddate: '{{ period.timeseries.std_enddate }}'

Only the lines with ``{{ ... }}`` depend on the kind. Everything else (variables, plot options) is fixed.
The two sections, ``biases`` and ``timeseries``, are read by the CLI with the same name.

**3. Add the reference slot to the kind file.**
In ``default.references`` of ``climatedt-experiment-kind.yaml`` add the slot, pointing to the anchor of an
observational dataset of ``default.sources`` (add the dataset there if it is missing).
The other kinds inherit the slot from ``default``:

.. code-block:: yaml

    default:
      references:
        snow_cover: *era5     # snow depth

If the new file needs dates different from ``period.standard`` and ``period.timeseries``, add a new block under ``default.period``
and remember to define it also in the kinds that change the dates.

**4. Add the CLIs to the analysis file.**
In ``config.aqua-analysis.yaml`` add the collection under ``diagnostics`` (the file is given to both CLIs)
and add its name to one of the lists of ``run``:

.. code-block:: yaml

    run:
      - ["teleconnections", "climate_metrics", "ocean2d", "snow"]
      # ...

    diagnostics:
      snow:
        biases:
          config: "${AQUA_CONFIG}/collections/jinja/snow/config-snow-cover.j2"
        timeseries:
          config: "${AQUA_CONFIG}/collections/jinja/snow/config-snow-cover.j2"

**5. Create the legacy file.**
Create ``config/collections/legacy/snow/config-snow-cover.yaml`` with the same name.
Copy the jinja file and replace every ``{{ ... }}`` with the value it has in the ``default`` kind
(here ERA5 as reference, ``regrid: r100``, ``loglevel: WARNING``, ``period.standard`` 1990-01-01 / 2014-12-31,
``period.timeseries`` standard deviation 1990-01-01 / 2014-12-31, output settings of ``default.output``):

.. code-block:: yaml
    :caption: config/collections/legacy/snow/config-snow-cover.yaml

    datasets:
      - catalog: null
        model: null
        exp: null
        source: null
        regrid: null
        startdate: null
        enddate: null

    references:
      - catalog: 'obs'
        model: 'ECMWF'
        exp: 'era5'
        source: 'monthly'
        regrid: 'r100'
        startdate: '1990-01-01'
        enddate: '2014-12-31'

    setup:
      loglevel: 'WARNING'

    output:
      outputdir: "./"
      rebuild: true
      save_netcdf: true
      save_format: ['png', 'pdf', 'svg']
      dpi: 300

    diagnostics:
      biases:
        run: true
        diagnostic_name: 'snow'
        variables: ['sd']
        params:
          default:
            seasons: true
            seasons_stat: 'mean'
            vertical: false
        plot_params:
          default:
            projection: 'robinson'
            projection_params: {}
            show_stats: true

      timeseries:
        run: true
        diagnostic_name: 'snow'
        variables: ['sd']
        center_time: true
        params:
          default:
            hourly: false
            daily: false
            monthly: true
            annual: true
            std_startdate: '1990-01-01'
            std_enddate: '2014-12-31'

Then register it in ``config.aqua-analysis-legacy.yaml`` exactly as in step 4,
with the path ``${AQUA_CONFIG}/collections/legacy/snow/config-snow-cover.yaml``.

**6. Check and run.**
The test ``tests/config/test_analysis_config_contract.py`` renders every jinja file with every kind, checks that the slot
exists and is used, and that the legacy file equals the jinja file rendered with ``default``:

.. code-block:: bash

    pytest tests/config/test_analysis_config_contract.py

To try only the new collection, copy the analysis file, keep only ``["snow"]`` in ``run`` and use ``--config``:

.. code-block:: bash

    aqua analysis --config my-analysis.yaml --kind climatedt-hist-cmip6 <options>
    aqua analysis --config my-analysis-legacy.yaml <options>

.. _analysis_collections_naming:

Naming scheme
-------------

Collection
    A folder named after the physical domain, with underscores (``atmosphere2d``, ``radiation_toa``, ``climate_metrics``).
    The same name is used in the ``jinja`` and ``legacy`` folders.

Configuration file
    ``config-<collection>-<quantity>``, with extension ``.j2`` (jinja) or ``.yaml`` (legacy).
    The two files of a pair have the same name. ``<quantity>`` is the physical quantity checked,
    not the reference dataset or the list of variables:
    ``config-atmosphere2d-surface``, ``config-ocean2d-sst``, ``config-radiation_toa-radiative``.
    Names that describe the diagnostic itself (``trends``, ``boxplots``, ``ecmean``, ``gregory``) are also fine.

Reference slots
    The slot of the kind file for a file is its name without ``config-`` and extension, with the hyphen
    between collection and quantity replaced by an underscore:
    ``config-atmosphere2d-surface.j2`` reads ``references.atmosphere2d_surface``.
    A file needing two references uses a suffix (``_nh`` / ``_sh``, ``_t2m`` / ``_toa``).
    Boxplots read two slots, their own (``radiation_toa_boxplots``, the first reference) and the one of the radiative file
    (``radiation_toa_radiative``, the second one).

Anchors in ``sources``
    Lowercase name of the dataset, with the product if there are several (``era5``, ``ceres_toa``, ``ceres_sfc``,
    ``osisaf_ssmis_nh``).

.. note::

    The ensemble legacy files (``*-ensemble-*.yaml``) and ``config_bootstrap.yaml`` have no jinja counterpart and keep their names.
