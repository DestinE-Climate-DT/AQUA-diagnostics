import os
import sys

from src.tropical_rainfall_cli_class import TropicalRainfallCLI
from src.tropical_rainfall_utils import load_configuration, parse_arguments, validate_arguments

from aqua.core.configurer import ConfigLocator
from aqua.core.logger import log_configure
from aqua.core.util import get_arg

# Initialize logger
logger = log_configure(log_name="Trop. Rainfall CLI", log_level="INFO")


def load_config(args):
    """Load the configuration file."""
    config_path = get_arg(args, "config", None)
    try:
        if config_path is None:
            config_path = os.path.join(
                ConfigLocator(logger=logger).configdir,
                "diagnostics",
                "tropical_rainfall",
                "cli",
                "cli_config_trop_rainfall.yml",
            )
        config = load_configuration(config_path)
        logger.info("Configuration successfully loaded from %s", config_path)
    except FileNotFoundError:
        logger.error("Configuration file not found at %s", config_path)
        sys.exit(2)
    except Exception as e:
        logger.error("An error occurred while loading configuration: %s", e)
        sys.exit(3)

    return config


def main():
    """Main function to orchestrate the tropical rainfall CLI operations."""
    # Parse and validate arguments
    args = parse_arguments(sys.argv[1:])
    validate_arguments(args)

    # Load configuration
    config = load_config(args)

    # Create the CLI object and run operations
    trop_rainfall_cli = TropicalRainfallCLI(config, args)

    try:
        trop_rainfall_cli.calculate_histogram_by_months()
        trop_rainfall_cli.plot_histograms()
        trop_rainfall_cli.average_profiles()
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        sys.exit(4)

    if trop_rainfall_cli.client:
        trop_rainfall_cli.client.close()
        logger.debug("Dask client closed.")

    if trop_rainfall_cli.private_cluster:
        trop_rainfall_cli.cluster.close()
        logger.debug("Dask cluster closed.")

    logger.info("Tropical rainfall diagnostic completed.")


if __name__ == "__main__":
    main()
