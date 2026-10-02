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
