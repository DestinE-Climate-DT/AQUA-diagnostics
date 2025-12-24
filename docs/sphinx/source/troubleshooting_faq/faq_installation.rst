.. _faq_installation:

I am getting an error when trying to install AQUA on LUMI
=========================================================

Resource is busy
----------------

It is possible that, if you're recreating the environment, the code breaks while removing the folder ``~/mambaforge/aqua/bin``, complaining the resource is busy.  
In this case you may have some processes running in the background. You can check them with ``ps -ef | grep aqua | grep $USER`` and kill them manually.  
In case of unexpected termination during the reinstallation of the container, it may be necessary to manually check for running open processes and terminate them.  
Finally, retry with the reinstallation.

Pip dependency resolution
-------------------------

You may encounter an error messange indicating a problem with the pip's dependency resolution process,
which may involve outdated dependencies for certain packages, e.g.:
  
.. code-block:: text
  
  ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
  This behaviour is the source of the following dependency conflicts.
  one-pass 0.4.2 requires tqdm, which is not installed..

These dependency resolution issues can often stem from outdated or conflicting information stored within the ``.local`` folder.
A potential solution is to remove the ``.local`` folder located in your home directory.
By doing so, you effectively eliminate any cached data or outdated information associated with Python packages, including dependency metadata and pip will begin its operations with a clean slate.

Permission denied
-----------------

If you are recreating an installation it is possible to encounter an error similar to this:

.. code-block:: bash

  PermissionError: [Errno 13] Permission denied: '/users/lrb_465000454_fdb/mars/versions/current/bin/../bin'
  [ ERROR ] Configuration construction failed 
  [ ERROR ] Set CW_DEBUG_KEEP_FILES env variable to keep build files

This may be due to a previous path added to the ``PATH`` variable in a previous installation and not available anymore.
In this case please remove manually the ``load_aqua.sh`` file from the previous installation and try again.

Intel compatibility build error
-------------------------------

On Intel-based macOS, you may encounter an error when pip tries to build `rasterio` from source because it cannot find `gdal` tools.  
If a similar error appears during the installation:

.. code-block:: text

  Collecting rasterio>=1.3 (from regionmask->aqua-core==0.21.0->-r /path/to/AQUA-diagnostics/condaenv.89sywc42.requirements.txt (line 1))
    Using cached rasterio-1.4.4.tar.gz (445 kB)
    ...

  Pip subprocess error:
    error: subprocess-exited-with-error
    
    x Getting requirements to build wheel did not run successfully.
    │ exit code: 1
    ╰─> [2 lines of output]
        WARNING:root:Failed to get options via gdal-config: [Errno 2] No such file or directory: 'gdal-config'
        ERROR: A GDAL API version must be specified. Provide a path to gdal-config using a GDAL_CONFIG environment variable or use a GDAL_VERSION environment variable.
        [end of output]
    
    note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed to build 'rasterio' when getting requirements to build wheel

  failed

  CondaEnvException: Pip failed
 
you can resolve this issue, by adding the following packages to the ``dependencies`` section in ``environment-dev.yml``:

.. code-block:: yaml

  dependencies:
    ...
    - gdal
    - rasterio
    - regionmask
    - geopandas

This ensure that pre-built binaries are used instead of attempting to build from source.
