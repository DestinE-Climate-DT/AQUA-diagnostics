.. _container:

Container
=========

AQUA provides separate container images for the core framework and for
AQUA-diagnostics.

The shared container launcher is provided by the AQUA-core repository at:

``cli/aqua-container/load-aqua-container.sh``


AQUA-diagnostics container
--------------------------

Use the AQUA-diagnostics container to run diagnostic analyses and publish their
results.

The AQUA-core launcher selects the diagnostics image when ``--diagnostics`` is
provided. For example, on LUMI:

.. code-block:: bash

   bash /path/to/AQUA/cli/aqua-container/load-aqua-container.sh \
       lumi --diagnostics

The aqua-web job templates pass ``--diagnostics`` automatically for both
analysis and publishing jobs.

Their ``aquadir`` configuration setting identifies the AQUA-core checkout
containing the shared container launcher.


AQUA-core container
-------------------

Without ``--diagnostics``, the launcher selects the AQUA-core container.

This is the container used by the DROP SLURM submission wrapper when
``--singularity`` is specified. DROP uses the core image because the
``aqua drop`` command remains part of AQUA-core.

The container choice therefore follows the component being executed:

* aqua-web analysis and publishing use the AQUA-diagnostics container.
* DROP processing uses the AQUA-core container and executes ``aqua drop``.

See :ref:`cli` for submission examples and configuration details.
