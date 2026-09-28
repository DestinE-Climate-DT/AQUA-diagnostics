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

Replace the uppercase placeholders with your catalog entry. ``--catalog`` is
optional; when omitted, Reader searches for the model, experiment and source.
The checker defaults to regridding to ``r100`` and rebuilding area and weight
files. Use ``--regrid`` to select another grid and ``--no-rebuild`` to reuse
existing files. One invocation checks one realization, selected with
``--realization``.

Additional Reader parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass additional Reader or catalog parameters using ``--reader-kwargs`` with
a JSON object. For example, to select the Polytope engine and set time chunks:

.. code-block:: bash

   python -m aqua.diagnostics.checker.cli_checker \
     --catalog CATALOG --model MODEL --exp EXP --source SOURCE \
     --realization r1 \
     --reader-kwargs '{"engine": "polytope", "chunks": {"time": 12}}'

Quote the entire object with single quotes in the shell and use double quotes
for JSON keys and strings. JSON preserves numbers, booleans (``true`` and
``false``), lists, nested objects and ``null`` (Python ``None``). The selected
Reader/backend determines which additional parameters it supports.

Keep ``catalog``, ``model``, ``exp``, ``source``, ``regrid``, ``startdate``,
``enddate``, ``loglevel`` and ``realization`` in their dedicated CLI flags.
Control ``rebuild`` with ``--no-rebuild`` (rebuilding is enabled by default).
These keys are rejected inside ``--reader-kwargs`` to avoid conflicting
settings. Invalid JSON or a value that is not an object also produces a CLI
error before retrieval.

Existing commands work without ``--reader-kwargs``. Previously, the checker
accepted only its predefined flags, so parameters such as ``engine`` could
not be supplied. The new option adds those parameters to the existing
``reader_kwargs`` dictionary, which is passed to ``Diagnostic.retrieve()``
and then to ``Reader``.

No user-managed configuration file is needed. The checker includes the kwargs
in its existing temporary YAML configuration and removes that file after
configuration preparation. The optional ``--yaml DIRECTORY`` flag still writes
experiment metadata to ``DIRECTORY/experiment.yaml``; it is unrelated to
``--reader-kwargs``. Run the command with ``--help`` to see the available flags.
