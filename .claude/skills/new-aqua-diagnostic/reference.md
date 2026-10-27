# Reference: shared machinery

Condensed from [DIAGNOSTIC_GUIDELINES.md](../../../DIAGNOSTIC_GUIDELINES.md) §3.1, §4.2, §5.2, §6.2 and §8.
Paths are relative to `aqua/diagnostics/` unless stated otherwise. Signatures can move: check the code
(`base/diagnostic.py`, `base/output_saver.py`, `base/cli_base.py`) before relying on a detail.

## `Diagnostic` (`base/diagnostic.py`)

| Member | What it does |
|---|---|
| `__init__(model, exp, source, catalog=None, regrid=None, startdate=None, enddate=None, std_startdate=None, std_enddate=None, loglevel="WARNING")` | Common signature, keep these names. |
| `retrieve(var=None, reader_kwargs={}, months_required=None)` | Builds the `Reader`, retrieves the widest window covering analysis and std period in one call, regrids if `regrid` is set, fills `self.data`, `self.std_data`, `self.reader`, `self.catalog`, `self.realization`. Dates outside the data are **clipped with a warning**. Raises `NotEnoughDataError` below `months_required`. Sets `AQUA_startdate/enddate` (and `AQUA_std_*`). `self.data` is a Dataset: index it with `[var]`. |
| `_retrieve(model, exp, source, var, catalog, startdate, enddate, regrid, months_required, reader_kwargs, loglevel)` | Engine behind `retrieve()` for complex retrievals (several sources, an extra dataset). Same checks, **populates no attribute**, returns `(data, reader, catalog)`. |
| `save_netcdf(data, diagnostic, diagnostic_product, outputdir, rebuild, extra_keys)` | Writes through `OutputSaver`; restores a missing `DataArray.name` from `short_name`/`standard_name` with a warning. |
| `load_netcdf(diagnostic, diagnostic_product, outputdir, extra_keys, as_dataarray, startdate, enddate, expect_netcdf)` | Rebuilds the same filename and opens it, or returns `None`. Works without `retrieve()`. A file whose **dataset-level** `AQUA_startdate/enddate` do not match the requested ones is not loaded (see the gotcha in SKILL.md). |
| `_set_region(region, regions_file_path=None, lon_limits=None, lat_limits=None)` | Resolves a region name against `config/definitions/regions.yaml`, returns `(longname, lon_limits, lat_limits)`. No data needed. |
| `select_region(data, region, drop=True)` | Area selection; sets `AQUA_region`; returns a **dict** `data, region, lon_limits, lat_limits`. |
| `_check_data(data, var, units)` | Unit conversion (factor and offset), updates `units`. |
| `_set_date_attrs()` | Re-applies the `AQUA_*date` attributes, e.g. after Dataset -> DataArray. |

## `OutputSaver` (`base/output_saver.py`)

The only way outputs are written. Filenames:

```
<diagnostic>.<diagnostic_product>.<catalog>.<model>.<exp>.<realization>[.<catalog_ref>.<model_ref>.<exp_ref>][.<extra_key values>...].<ext>
```

in `<outputdir>/netcdf|png|pdf|svg/`. With more than one model (or reference) the triplet becomes `multimodel`
(`multiref`) and the list goes to the metadata. `extra_keys` values are appended **in dict order**: keep the order
stable. Formats: `png`, `pdf`, `svg` (`SAVE_FORMAT` in `base/defaults.py`) and `nc`.
Initialise it with **one dataset for NetCDF** (done by `Diagnostic.save_netcdf`) and with **all models and
references for figures**.

## Contract with `aqua analysis`

```
python <cli> --model M --exp E --source S --outputdir <out>/<collection> \
       [--regrid R] [--nworkers N] [--cluster ADDR] [--catalog C] [--realization R] \
       [--startdate D] [--enddate D] [extra] --config <rendered .j2>
```

All tools of a collection share the output folder: that is why `diagnostic_name` matters.

## Where configuration lives

| Folder in the repo | Installed as | Used for |
|---|---|---|
| `templates/collections/config-<name>.yaml` | `$AQUA_CONFIG/templates/collections/` | Standalone template, the starting point for a user. |
| `config/collections/jinja/<collection>/*.j2` | `$AQUA_CONFIG/collections/jinja/` | Configs run by `aqua analysis`, rendered with the dates of the experiment kind (`{{ default.startdate }}`, `{{ timeseries.std_startdate }}`) from `config/analysis/climatedt-experiment-kind.yaml`. One file can configure several tools of a collection against one reference. |
| `config/collections/legacy/<collection>/*.yaml` | `$AQUA_CONFIG/collections/legacy/` | Plain-YAML versions without Jinja. |
| `config/tools/<name>/` | `$AQUA_CONFIG/tools/<name>/` | Lookup files of one tool. |
| `config/definitions/regions.yaml` | `$AQUA_CONFIG/definitions/` | Shared region registry. |
| `config/analysis/` | `$AQUA_CONFIG/analysis/` | `aqua analysis` orchestration. |

Helpers: `get_diagnostic_configpath(diagnostic, folder=...)`, `load_diagnostic_config(...)`. `.j2` files are only
rendered by `aqua analysis`: a standalone run with a `.j2` keeps the literal `'{{ default.startdate }}'`.
Without `--config`, the default lookup of `DiagnosticCLI` does not resolve for most tools: pass `--config`.

## aqua-core functions worth knowing

**Reader operators**: `fldmean/fldstat/fldsum/fldintg/fldarea` (area-weighted, `lon_limits`, `lat_limits`,
`box_brd`, `dims`), `timmean/timstat/timstd/timsum/timmax/timmin` (`freq`, `exclude_incomplete`,
`center_time`), `select_area`, `regrid`, `vertinterp`, `detrend`, `histogram`. Without a Reader:
`aqua.core.fldstat.AreaSelection/FldStat`, `aqua.core.timstat.TimStat`, `aqua.core.reader.Trender`,
`aqua.core.fixer.EvaluateFormula`, `aqua.core.data_model.CoordIdentifier` (wrapped by `find_vert_coord`).

**`aqua.core.util`**: `to_list`, `get_arg`, `load_yaml`/`dump_yaml`, `time_to_string`, `pandas_freq_to_string`,
`int_month_name`, `select_season`, `convert_units`, `unit_to_latex`, `DEFAULT_REALIZATION`, `get_realizations`,
`format_realization`, `merge_attrs`, `extract_attrs`, `update_metadata`, `lon_to_180/360`, `add_cyclic_lon`,
`evaluate_colorbar_limits`, `cbar_get_label`, `generate_colorbar_ticks`, `get_projection`, `plot_box`,
`strlist_to_phrase`, `lat_to_phrase`, `create_folder`, `clean_filename`.

**Logging, config, exceptions**: `aqua.core.logger.log_configure(log_level, log_name)` and `log_history(data, msg)`;
`aqua.core.configurer.ConfigLocator().configdir`, `ConfigCatalog`; `aqua.core.exceptions`: `NoDataError`,
`NotEnoughDataError`, `NoObservationError`, `NoRegridError`.

**`aqua.core.graphics`**: `plot_single_map`, `plot_single_map_diff`, `plot_maps`, `plot_maps_diff`,
`plot_timeseries`, `plot_seasonalcycle`, `plot_hovmoller`, `plot_vertical_profile`, `plot_vertical_profile_diff`,
`plot_lat_lon_profiles`, `plot_seasonal_lat_lon_profiles`, `plot_vertical_lines`, `plot_histogram`, `boxplot`,
`index_plot`, `indexes_plot`, `plot_gregory_monthly`, `plot_gregory_annual`, `ConfigStyle`.

**`aqua.diagnostics.base`**: `Diagnostic`, `DiagnosticCLI`, `OutputSaver`, `TitleBuilder`,
`template_parse_arguments`, `load_diagnostic_config`, `merge_config_args`, `get_diagnostic_configpath`,
`load_var_config`, `find_vert_coord`, `open_cluster/close_cluster`, `start_end_dates`,
`round_startdate/round_enddate`, `SAVE_FORMAT`; `DEFAULT_OCEAN_VERT_COORD` in `base/defaults.py`.
