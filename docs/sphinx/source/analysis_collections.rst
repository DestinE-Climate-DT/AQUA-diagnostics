.. _analysis_collections:

Analysis collections: kind, jinja and legacy files
===================================================

``aqua analysis`` (see :ref:`aqua_analysis`) runs the diagnostics through *collections*:
sets of configuration files, one for each diagnostic CLI, grouped by the physical domain they check
(``atmosphere2d``, ``ocean3d``, ``radiation_toa``, ...).
This page explains how the collections are defined, how they are used and how to modify them or create new ones.

All files live in the ``aqua/diagnostics/config`` folder (installed in ``$AQUA_CONFIG``):

.. code-block:: text

    config/
    ├── analysis/
    │   ├── config.aqua-analysis.yaml          # which diagnostics to run, with which collection files (jinja mode)
    │   ├── config.aqua-analysis-legacy.yaml   # same, with the legacy collection files
    │   └── climatedt-experiment-kind.yaml     # kind file: the values filled in the jinja templates
    └── collections/
        ├── jinja/<collection>/config-<collection>-<quantity>.j2
        └── legacy/<collection>/config-<collection>-<name>.yaml

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

The kind ``default`` is mandatory and contains *every* definition used by the templates.
All the other kinds only list the keys they want to change:
aqua-core deep-merges them on top of ``default`` before rendering.
The content of ``default`` is:

``regrid``
    Target grid of the reference datasets (model datasets are regridded through the CLI ``--regrid`` option).

``loglevel``
    Value of ``setup.loglevel`` in the templates.

``output``
    Figure settings shared by all the templates: ``save_format``, ``dpi`` and ``rebuild``.
    ``outputdir`` and ``save_netcdf`` are not centralised and stay in each template.

``sources``
    Every observational dataset available, listed once as ``{catalog, model, exp, source}``
    with a YAML anchor of the same name (``era5``, ``ceres_toa``, ...).
    The templates never read this block: it only exists to define the datasets in one place
    and be reused through anchors. It must live inside ``default`` because aqua-core treats every top-level key of the
    file as a kind, so a top-level ``sources`` would become a selectable ``--kind sources``.

``references``
    One *slot* for each template, pointing to one of the anchors of ``sources``.
    This is the block the templates read to know which dataset to compare with
    (see :ref:`analysis_collections_naming` for the slot names).
    A kind swaps the reference of a single template by pointing the slot to another anchor.

``period``
    Date ranges of the references, in three independent blocks:
    ``standard`` (used by most analyses), ``radiation`` (radiation analyses against CERES,
    which only starts in March 2000) and ``timeseries``.
    Each block has ``startdate``, ``enddate`` and the ``std_startdate``, ``std_enddate`` used to compute the
    standard deviation of the reference.

The kind name is also passed to the templates as ``kind``. This name is reserved and cannot be defined in the kind file.

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

A jinja collection file (``.j2``) is the configuration file of one diagnostic CLI,
in which the values that depend on the experiment kind are replaced by jinja expressions
reading the resolved kind, for instance:

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
      rebuild: {{ output.rebuild }}
      save_format: {{ output.save_format }}
      dpi: {{ output.dpi }}

The rendering is strict: referring to a key that the resolved kind does not define is an error,
so a template cannot silently use a missing value.
The rendered files are written in a temporary folder and passed to the diagnostic CLI.
Values that do not depend on the kind (variables, plot options, ``outputdir``, ...) are written directly in the template.

Legacy files
^^^^^^^^^^^^

A legacy collection file (``.yaml``) has the same content as the rendered template,
but all values are written explicitly (dataset names, dates, regrid, figure settings).
There is no kind file, so a change in a reference dataset or in a date range has to be repeated in every file.
Legacy files are not touched by the kind mechanism.

Using the files
---------------

Jinja mode
^^^^^^^^^^

``config.aqua-analysis.yaml`` is the default analysis configuration. It points to the kind file
and, for each diagnostic CLI, to the jinja template(s) to use:

.. code-block:: yaml

    job:
      experiment_kind: "${AQUA_CONFIG}/analysis/climatedt-experiment-kind.yaml"

    diagnostics:
      atmosphere2d:
        biases:
          config: ["${AQUA_CONFIG}/collections/jinja/atmosphere2d/config-atmosphere2d-temperature.j2",
                   "${AQUA_CONFIG}/collections/jinja/atmosphere2d/config-atmosphere2d-surface.j2"]

A list of files runs the CLI once for each file.
The kind is selected from the command line:

.. code-block:: bash

    aqua analysis --kind climatedt-hist-cmip7 <options>

Without ``--kind`` the ``default`` kind is used. An unknown kind raises an error listing the available ones.
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
    Edit the jinja template, or the legacy file if you run in legacy mode.
    The same change has to be done in both if they are meant to be equivalent.

Run a different set of diagnostics or collections
    Edit the ``run`` and ``diagnostics`` sections of the analysis configuration file
    (see :ref:`aqua_analysis`).

Creating a new collection
-------------------------

In jinja mode
^^^^^^^^^^^^^

#. Create the folder ``config/collections/jinja/<collection>`` and the template(s)
   ``config-<collection>-<quantity>.j2``, starting from a template of an existing collection
   (or from the diagnostic template in ``aqua/diagnostics/templates/collections``).
#. In the template, replace the kind-dependent values with the expressions of the kind, as in the example above:
   ``references.<collection>_<quantity>.*`` for the reference dataset, ``period.<block>.*`` for the dates,
   ``regrid``, ``loglevel`` and ``output.*``.
#. In ``default.references`` of the kind file add the slot ``<collection>_<quantity>`` pointing to the anchor of the
   dataset to use, with a comment on what it is used for. Add the dataset to ``default.sources`` if it is not there yet.
#. If the reference needs different dates, add a ``period`` block in ``default``
   (and in all the kinds that must change it, since blocks are not inherited).
#. Add the template to the ``diagnostics`` section of ``config.aqua-analysis.yaml``,
   and the collection to the ``run`` list.
#. Check that the template renders for all the kinds, with the test
   ``tests/config/test_analysis_config_contract.py``, which renders every template with every kind in strict mode and checks that
   every slot is a source and is read by at least one template.

In legacy mode
^^^^^^^^^^^^^^

#. Create the folder ``config/collections/legacy/<collection>`` and the file ``config-<collection>-<name>.yaml``,
   copying an existing legacy file of a similar diagnostic.
#. Write all the values explicitly: ``references`` (``catalog``, ``model``, ``exp``, ``source``, ``regrid``, dates),
   ``setup``, ``output`` and the ``diagnostics`` block.
#. Add the file to the ``diagnostics`` section of ``config.aqua-analysis-legacy.yaml`` and the collection to the ``run`` list.

.. _analysis_collections_naming:

Naming scheme
-------------

Collection
    A folder named after the physical domain, with underscores (``atmosphere2d``, ``radiation_toa``, ``climate_metrics``).
    The same name is used in the ``jinja`` and ``legacy`` folders.

Jinja files
    ``config-<collection>-<quantity>.j2``, where ``<quantity>`` is the physical quantity checked,
    not the reference dataset or the list of variables:
    ``config-atmosphere2d-surface.j2``, ``config-ocean2d-sst.j2``, ``config-radiation_toa-radiative.j2``.
    Names that describe the diagnostic itself (``trends``, ``boxplots``, ``ecmean``, ``gregory``) are also fine.

Reference slots
    The slot of the kind file for a template is its name without ``config-`` and ``.j2``, with the hyphen
    between collection and quantity replaced by an underscore:
    ``collections/jinja/atmosphere2d/config-atmosphere2d-surface.j2`` reads ``references.atmosphere2d_surface``.
    A template needing two references uses a suffix (``_nh`` / ``_sh``, ``_t2m`` / ``_toa``).
    Boxplots read two slots, their own (``radiation_toa_boxplots``, the first reference) and the one of the radiative template
    (``radiation_toa_radiative``, the second one).

Legacy files
    ``config-<collection>-<name>.yaml``. Legacy files created before the kind mechanism are named after the
    reference dataset (e.g. ``config-ocean2d-esacci.yaml``) and keep their names.

Anchors in ``sources``
    Lowercase name of the dataset, with the product if there are several (``era5``, ``ceres_toa``, ``ceres_sfc``,
    ``osisaf_ssmis_nh``).
