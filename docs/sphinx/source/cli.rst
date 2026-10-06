.. _cli:

Command Line Interface tools
============================

Setup checker
-------------

The setup checker verifies that a dataset can be retrieved through AQUA's
``Diagnostic`` and ``Reader``. Run it from the installed diagnostics package:

.. code-block:: bash

   python -m aqua.diagnostics.checker.cli_checker \
     --catalog CATALOG --model MODEL --exp EXP --source SOURCE \
     --realization r1

Replace the uppercase placeholders with your catalog entry. No configuration
file is needed: the checker builds one internally from the CLI arguments and
removes it after use.

The checker accepts the shared diagnostics CLI arguments described in
:ref:`diagnostics-cli-arguments` (including ``--reader_kwargs``), plus the
following options of its own:

- ``--catalog``: Optional; when omitted, Reader searches for the model, experiment and source.
- ``--model``, ``--exp``, ``--source``: Required.
- ``--realization``: One invocation checks one realization.
- ``--regrid``: Target regrid resolution. Default is ``r100``.
- ``--no-rebuild``: Reuse existing area and regridding weight files instead of rebuilding them
  (rebuilding is the default). Sets ``rebuild`` in ``reader_kwargs``, so this key is also rejected
  inside ``--reader_kwargs``.
- ``--yaml DIRECTORY``: Write the checked experiment metadata to ``DIRECTORY/experiment.yaml``.
  Unrelated to ``--reader_kwargs``.

Run the command with ``--help`` to see the full list of available flags.

SLURM submission helpers
------------------------

The scripts in the AQUA-diagnostics repository submit SLURM jobs for diagnostic
analyses, publish analysis results, and submit DROP processing jobs. The scripts
are located in ``cli/aqua-web/`` and ``cli/drop/``.

While DROP is part of AQUA-core, the ``cli/drop/`` tools are SLURM submission
helpers for the AQUA-core ``aqua drop`` command.

Repository paths
----------------

The examples below use two placeholder paths. Replace them with the absolute
paths to your local copies of the repositories:

* ``/path/to/AQUA-diagnostics`` contains the submission scripts, job templates,
  example configuration files and publishing tools.
* ``/path/to/AQUA`` contains the core container launcher,
  ``cli/aqua-container/load-aqua-container.sh``, as well as the AQUA-core
``aqua drop`` command used by the DROP submission wrapper.

For aqua-web container jobs, set ``aquadir`` in ``config.aqua-web.yaml`` to
the AQUA-core repository path. For DROP container jobs, set ``aquadir`` under the
``slurm`` section of the DROP configuration, or set the ``AQUA`` environment
variable to that path. The DROP configuration setting takes priority over the
environment variable.

The publishing script is located at:
``/path/to/AQUA-diagnostics/cli/aqua-web/push_analysis.sh``.
The aqua-web submitter locates it in the same directory as
``submit-aqua-web.py`` and passes its absolute path to the job template as
``push_analysis_script``.

The repository paths must be accessible from the compute nodes. For publishing
jobs that run in a container, the diagnostics repository must also be accessible
inside the container at the same absolute path.

Submitting aqua-web jobs
------------------------

Copy ``cli/aqua-web/config.aqua-web.yaml`` to a location for your own
configuration and edit the machine, account, username, partitions, log directories,
output directory and ``aquadir`` for your machine. The ``analysis.memory``
setting controls the analysis job's SLURM memory request. Create the configured
log directories before submitting jobs.

Use ``cli/aqua-web/aqua-web.experiment.list`` as an example when preparing
your experiment list. Pass an absolute path to this list because the generated
jobs change their working directory.

Preview the analysis and publishing jobs with:

.. code-block:: bash

   python /path/to/AQUA-diagnostics/cli/aqua-web/submit-aqua-web.py \
       --config /path/to/config.aqua-web.yaml --dry --push \
       /path/to/aqua-web.experiment.list

The ``--dry`` option prints the generated job scripts without submitting them.
Remove ``--dry`` to submit the jobs to SLURM. The ``--push`` option requests
a publishing job after the analysis jobs finish; omit it to run only the
analyses.

By default, both job types run in the AQUA-diagnostics container. The supplied
job templates select it by passing ``--diagnostics`` to the core container
launcher. Add ``--native`` to the submitter command to run the jobs in the
active AQUA environment instead. See :ref:`container` for direct use of the
launcher.

Publishing also requires storage and repository credentials and the diagnostics
grouping configuration. By default, ``push_analysis.sh`` reads
``$AQUA_CONFIG/analysis/config.grouping.yaml`` when ``AQUA_CONFIG`` is set,
or ``$HOME/.aqua/analysis/config.grouping.yaml`` otherwise.

Configuration and template lookup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default configuration and both job templates are provided in
``AQUA-diagnostics/cli/aqua-web/``. The configuration's ``analysis.template``
and ``push.template`` entries select the analysis and publishing templates.

For configuration and template files, an existing absolute path is used
directly. A relative path or filename is checked in this order:

1. The working directory from which you run the submitter.
2. The AQUA configuration directory, resolved using ``AQUA_CONFIG`` or the
   usual ``$HOME/.aqua`` fallback.
3. The ``cli/aqua-web/`` directory containing ``submit-aqua-web.py``.

Template filenames in the YAML configuration follow the same search order;
they are not automatically resolved relative to the YAML file's directory.
Use absolute template paths when storing custom templates elsewhere.

Custom publishing templates should use ``{{ push_analysis_script }}`` for the
publishing script path. Custom aqua-web container templates must pass
``--diagnostics`` to the launcher.

Submitting DROP jobs
--------------------

Copy ``cli/drop/auto_LRA_overnight.yaml`` and adapt its data selections and
SLURM settings. Create a ``log`` directory in the working directory from which
you will submit jobs.

The wrapper loads the template at
``AQUA-diagnostics/cli/drop/aqua_drop.j2``, in the same directory as
``cli_drop_parallel_slurm.py``. You can therefore invoke the wrapper from
another working directory.

Preview a container job submission with:

.. code-block:: bash

   python /path/to/AQUA-diagnostics/cli/drop/cli_drop_parallel_slurm.py \
       --config /path/to/auto_LRA_overnight.yaml --singularity

Without ``--definitive``, the wrapper checks the SLURM queue and writes
``tempfile.job`` in the working directory, but does not submit jobs.
Each generated job replaces this preview file, so it contains the last job
generated. Previewing still requires access to SLURM's ``squeue`` command.

Add ``--definitive`` to submit the jobs and produce DROP output. With
``--singularity``, jobs use the AQUA-core container, which provides
``aqua drop``. Omit ``--singularity`` to use the active AQUA environment.

The wrapper's ``--workers`` option sets the default worker and SLURM task count.
The ``slurm.ntasks_per_node`` setting overrides that count when present.
The per-source ``nworkers`` setting in the YAML configuration is read by the
core DROP command and can override the worker count. Ensure the SLURM allocation
is sufficient for the number of workers requested.
