#!/bin/bash

# Install AQUA-diagnostics framework.
# First set AQUA_DIAGNOSTICS variable to the path to the AQUA-diagnostics repository
# it is reccomended to set it to the .bashrc file, similarly to how $AQUA is set.

# Usage
# bash lumi_install.sh 
# or
# bash lumi_install.sh --help

set -e

# Check if AQUA is set and the directory exists
# We need AQUA for the logger and for editable install of aqua-core
if [[ -z "$AQUA" ]]; then
    echo -e "\033[0;31mWarning: The AQUA environment variable is not defined."
    echo -e "\033[0;31mWarning: We are assuming AQUA is installed in your HOME, i.e. $HOME/AQUA"
    echo -e "\x1b[38;2;255;165;0mAlternatively, define the AQUA environment variable with the path to your 'AQUA' directory."
    echo -e "For example: export AQUA=/path/to/aqua\033[0m"
    export AQUA=$HOME/AQUA
fi

if [[ ! -d  $AQUA ]] ; then
    echo -e "\033[0;31mWarning: I cannot find AQUA, exiting..."
    exit 1  # Exit with status 1 to indicate an error
else
    source "$AQUA/cli/util/logger.sh"
    log_message INFO "AQUA found in $AQUA"
    log_message INFO "Sourcing logger.sh from: $AQUA/cli/util/logger.sh"
fi

# Check if AQUA_DIAGNOSTICS is set and the directory exists
if [[ -z "$AQUA_DIAGNOSTICS" ]]; then
    echo -e "\033[0;31mWarning: The AQUA_DIAGNOSTICS environment variable is not defined."
    echo -e "\033[0;31mWarning: We are assuming AQUA-diagnostics is installed in your HOME, i.e. $HOME/AQUA-diagnostics"
    echo -e "\x1b[38;2;255;165;0mAlternatively, define the AQUA_DIAGNOSTICS environment variable with the path to your 'AQUA-diagnostics' directory."
    echo -e "For example: export AQUA_DIAGNOSTICS=/path/to/AQUA-diagnostics\033[0m"
    export AQUA_DIAGNOSTICS=$HOME/AQUA-diagnostics
fi

if [[ ! -d  $AQUA_DIAGNOSTICS ]] ; then
    echo -e "\033[0;31mWarning: I cannot find AQUA-diagnostics, exiting..."
    exit 1  # Exit with status 1 to indicate an error
else
    log_message INFO "AQUA-diagnostics found in $AQUA_DIAGNOSTICS"
fi

setup_log_level 2 # 1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR, 5=CRITICAL
#####################################################################
# Begin of user input
user=$USER # change this to your username if automatic detection fails
MAMBADIR="$HOME/mambaforge" #check if $HOME does not exist
load_aqua_diagnostics_file="$HOME/load_aqua_diagnostics.sh" #check if $HOME does not exist
# End of user input
#####################################################################

# define installation path
export INSTALLATION_PATH="$MAMBADIR/aqua-diagnostics"
log_message INFO "Installation path has been set to ${INSTALLATION_PATH}"

# Remove the installation paths from the $PATH. 
# This removes paths containing 'aqua-diagnostics' or 'aqua/bin' (aqua-core)
# to ensure a clean environment before installation

# Split the $PATH into individual components using ":" as the separator
IFS=":" read -ra path_components <<< "$PATH"

# Create a new array to store the modified path components
new_path_components=()
removed_count=0

# Loop through each path component and filter out aqua-related paths
for component in "${path_components[@]}"; do
  if [[ "$component" != *"aqua-diagnostics"* ]] && [[ "$component" != *"aqua/bin"* ]]; then
    # If the component does not contain aqua paths, add it to the new array
    new_path_components+=("$component")
  else
    removed_count=$((removed_count + 1))
  fi
done

# Join the new array back into a single string with ":" as the separator
new_path=$(IFS=":"; echo "${new_path_components[*]}")

# Update the $PATH variable with the new value
export PATH="$new_path"

if [[ $removed_count -gt 0 ]]; then
  log_message INFO "Removed $removed_count AQUA-related path(s) from \$PATH (aqua-diagnostics and aqua/bin)."
else
  log_message INFO "No AQUA-related paths found in \$PATH."
fi

#####################################################################

install_aqua_diagnostics() {
  # clean up environment
  module --force purge
  log_message INFO "Environment has been cleaned up."

  # load modules
  module load LUMI/24.03
  module load lumi-container-wrapper
  log_message INFO "Modules have been loaded."
  
  # install AQUA-diagnostics framework
  conda-containerize new --mamba --prefix "${INSTALLATION_PATH}" "${AQUA_DIAGNOSTICS}/cli/lumi-install/environment_lumi.yml"
  conda-containerize update "${INSTALLATION_PATH}" --post-install "${AQUA_DIAGNOSTICS}/cli/lumi-install/pip_lumi.txt"
  log_message INFO "AQUA-diagnostics framework has been installed."

}

# if INSTALLATION_PATH does not exist, create it
if [[ ! -d "${INSTALLATION_PATH}" ]]; then
  mkdir -p "${INSTALLATION_PATH}"
  log_message INFO "Installation path ${INSTALLATION_PATH} has been created."
else
  log_message INFO "Installation path ${INSTALLATION_PATH} already exists."
fi

# if INSTALLATION_PATH is empty, install AQUA-diagnostics
if [[ -z "$(ls -A ${INSTALLATION_PATH})" ]]; then
  log_message INFO "Installing AQUA-diagnostics..."
  # install AQUA-diagnostics
  install_aqua_diagnostics
else
  log_message INFO "AQUA-diagnostics is already installed."
  # check if reinstallation is wanted

  log_message $next_level_msg_type "Do you want to reinstall AQUA-diagnostics? (y/n) "
  # Read the user's input
  read -n 1 -r
  echo

  if [[ $REPLY =~ ^[Yy]$ ]]
  then
    # run code to reinstall AQUA-diagnostics
    log_message INFO "Removing AQUA-diagnostics..."
    log_message $next_level_msg_type "Are you sure you want to delete ${INSTALLATION_PATH}? (y/n) "
    # Read the user's input
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
      rm -rf "${INSTALLATION_PATH}"
      mkdir -p "${INSTALLATION_PATH}"
    else
      log_message CRITICAL "Deletion cancelled."
      exit 1
    fi
    log_message INFO "Installing AQUA-diagnostics..."
    install_aqua_diagnostics
  else
    log_message ERROR "AQUA-diagnostics will not be reinstalled."
  fi
fi

create_aqua_diagnostics_file() {
  # Create a new file
  touch $load_aqua_diagnostics_file

  #echo '# Use ClimateDT paths' >> $load_aqua_diagnostics_file
  #echo 'module use /project/project_465000454/software/23.09/modules/C' >> $load_aqua_diagnostics_file
  #echo '# Load modules' >> $load_aqua_diagnostics_file
  # Removed, see issue #1195
  #echo 'module purge' >> $load_aqua_diagnostics_file
  #echo 'module load fdb/5.12.1-cpeCray-23.09' >> $load_aqua_diagnostics_file
  # These are loaded automatically with the fdb module
  # echo 'module load eckit/1.26.3-cpeCray-24.03' >> $load_aqua_diagnostics_file
  # echo 'module load metkit/1.11.14-cpeCray-24.03' >> $load_aqua_diagnostics_file
    
  log_message INFO "exports for FDB5 added to load_aqua_diagnostics.sh."

  # Config GSV: check load_modules_lumi.sh on GSV repo https://earth.bsc.es/gitlab/digital-twins/de_340/gsv_interface/-/blob/main/load_modules_lumi.sh
  echo 'export GSV_WEIGHTS_PATH=/scratch/project_465000454/igonzalez/gsv_weights' >>  $load_aqua_diagnostics_file
  echo 'export GSV_TEST_FILES=/scratch/project_465000454/igonzalez/gsv_test_files' >> $load_aqua_diagnostics_file
  echo 'export GRID_DEFINITION_PATH=/scratch/project_465000454/igonzalez/grid_definitions' >>  $load_aqua_diagnostics_file

  # Currently (Feb 2025) this is the recommended setup overcoming lumi modules
  # This points to a stack with fdb 5.14.0 and the required associated modules
  echo 'export PATH=/appl/local/destine/mars/versions/current/bin:$PATH' >>  $load_aqua_diagnostics_file
  echo 'export LD_LIBRARY_PATH=/appl/local/destine/mars/versions/current/lib64:$LD_LIBRARY_PATH' >>  $load_aqua_diagnostics_file

  log_message INFO "export for GSV has been added to load_aqua_diagnostics.sh."

  # Define AQUA installation paths
  echo "# AQUA installation paths" >>  $load_aqua_diagnostics_file
  echo 'export AQUA_DIAGNOSTICS_PATH="'$INSTALLATION_PATH'/bin"' >>  $load_aqua_diagnostics_file
  echo 'export AQUA_CORE_PATH="'$MAMBADIR'/aqua/bin"' >>  $load_aqua_diagnostics_file
  echo '' >>  $load_aqua_diagnostics_file
  
  # Function to switch between AQUA environments
  echo '# Function to switch between AQUA environments' >>  $load_aqua_diagnostics_file
  echo '# Usage: switch_aqua [-v|--verbose] [diagnostics|core]' >>  $load_aqua_diagnostics_file
  echo 'switch_aqua() {' >>  $load_aqua_diagnostics_file
  echo '  local verbose=false' >>  $load_aqua_diagnostics_file
  echo '  local target=""' >>  $load_aqua_diagnostics_file
  echo '  ' >>  $load_aqua_diagnostics_file
  echo '  # Parse arguments for verbose flag and target' >>  $load_aqua_diagnostics_file
  echo '  for arg in "$@"; do' >>  $load_aqua_diagnostics_file
  echo '    if [[ "$arg" == "-v" ]] || [[ "$arg" == "--verbose" ]]; then' >>  $load_aqua_diagnostics_file
  echo '      verbose=true' >>  $load_aqua_diagnostics_file
  echo '    elif [[ "$arg" == "diagnostics" ]] || [[ "$arg" == "core" ]]; then' >>  $load_aqua_diagnostics_file
  echo '      target="$arg"' >>  $load_aqua_diagnostics_file
  echo '    fi' >>  $load_aqua_diagnostics_file
  echo '  done' >>  $load_aqua_diagnostics_file
  echo '  ' >>  $load_aqua_diagnostics_file
  echo '  # Validate: Error if any argument is not allowed' >>  $load_aqua_diagnostics_file
  echo '  for arg in "$@"; do' >>  $load_aqua_diagnostics_file
  echo '    if [[ "$arg" != "-v" && "$arg" != "--verbose" && "$arg" != "diagnostics" && "$arg" != "core" ]]; then' >>  $load_aqua_diagnostics_file
  echo '      echo "Error: Invalid argument '\''$arg'\''"' >>  $load_aqua_diagnostics_file
  echo '      echo "Usage: switch_aqua [-v|--verbose] [diagnostics|core]"' >>  $load_aqua_diagnostics_file
  echo '      return 1' >>  $load_aqua_diagnostics_file
  echo '    fi' >>  $load_aqua_diagnostics_file
  echo '  done' >>  $load_aqua_diagnostics_file
  echo '  ' >>  $load_aqua_diagnostics_file
  echo '  # Default to diagnostics if no target specified' >>  $load_aqua_diagnostics_file
  echo '  if [[ -z "$target" ]]; then' >>  $load_aqua_diagnostics_file
  echo '    target="diagnostics"' >>  $load_aqua_diagnostics_file
  echo '  fi' >>  $load_aqua_diagnostics_file
  echo '  ' >>  $load_aqua_diagnostics_file
  echo '  # Store original PATH for verbose output' >>  $load_aqua_diagnostics_file
  echo '  local original_path="$PATH"' >>  $load_aqua_diagnostics_file
  echo '  local original_first=$(echo "$PATH" | cut -d: -f1)' >>  $load_aqua_diagnostics_file
  echo '  ' >>  $load_aqua_diagnostics_file
  echo '  # Remove both aqua paths from PATH' >>  $load_aqua_diagnostics_file
  echo '  # Split PATH and filter out paths containing aqua-diagnostics or aqua/bin' >>  $load_aqua_diagnostics_file
  echo '  local cleaned_path=""' >>  $load_aqua_diagnostics_file
  echo '  local removed_paths=()' >>  $load_aqua_diagnostics_file
  echo '  IFS=":" read -ra path_components <<< "$PATH"' >>  $load_aqua_diagnostics_file
  echo '  for component in "${path_components[@]}"; do' >>  $load_aqua_diagnostics_file
  echo '    if [[ "$component" == *"aqua-diagnostics"* ]] || [[ "$component" == *"aqua/bin"* ]]; then' >>  $load_aqua_diagnostics_file
  echo '      removed_paths+=("$component")' >>  $load_aqua_diagnostics_file
  echo '    else' >>  $load_aqua_diagnostics_file
  echo '      if [[ -z "$cleaned_path" ]]; then' >>  $load_aqua_diagnostics_file
  echo '        cleaned_path="$component"' >>  $load_aqua_diagnostics_file
  echo '      else' >>  $load_aqua_diagnostics_file
  echo '        cleaned_path="$cleaned_path:$component"' >>  $load_aqua_diagnostics_file
  echo '      fi' >>  $load_aqua_diagnostics_file
  echo '    fi' >>  $load_aqua_diagnostics_file
  echo '  done' >>  $load_aqua_diagnostics_file
  echo '  export PATH="$cleaned_path"' >>  $load_aqua_diagnostics_file
  echo '  ' >>  $load_aqua_diagnostics_file
  echo '  # Add the selected environment to PATH' >>  $load_aqua_diagnostics_file
  echo '  if [[ "$target" == "diagnostics" ]]; then' >>  $load_aqua_diagnostics_file
  echo '    export PATH="$AQUA_DIAGNOSTICS_PATH:$PATH"' >>  $load_aqua_diagnostics_file
  echo '    if [[ "$verbose" == "true" ]]; then' >>  $load_aqua_diagnostics_file
  echo '      echo "=== AQUA Environment Switch ==="' >>  $load_aqua_diagnostics_file
  echo '      echo "Target: AQUA-diagnostics"' >>  $load_aqua_diagnostics_file
  echo '      if [[ ${#removed_paths[@]} -gt 0 ]]; then' >>  $load_aqua_diagnostics_file
  echo '        echo "Removed paths:"' >>  $load_aqua_diagnostics_file
  echo '        for path in "${removed_paths[@]}"; do' >>  $load_aqua_diagnostics_file
  echo '          echo "  - $path"' >>  $load_aqua_diagnostics_file
  echo '        done' >>  $load_aqua_diagnostics_file
  echo '      else' >>  $load_aqua_diagnostics_file
  echo '        echo "Removed paths: (none)"' >>  $load_aqua_diagnostics_file
  echo '      fi' >>  $load_aqua_diagnostics_file
      echo '      echo "Added path: \$AQUA_DIAGNOSTICS_PATH ($AQUA_DIAGNOSTICS_PATH)"' >>  $load_aqua_diagnostics_file
      echo '      echo "Previous PATH component: $original_first"' >>  $load_aqua_diagnostics_file
      echo '      echo "New PATH component: $(echo "$PATH" | cut -d: -f1)"' >>  $load_aqua_diagnostics_file
  echo '      echo "========================================"' >>  $load_aqua_diagnostics_file
  echo '    fi' >>  $load_aqua_diagnostics_file
  echo '    echo "Switched to AQUA-diagnostics environment"' >>  $load_aqua_diagnostics_file
  echo '  elif [[ "$target" == "core" ]]; then' >>  $load_aqua_diagnostics_file
  echo '    export PATH="$AQUA_CORE_PATH:$PATH"' >>  $load_aqua_diagnostics_file
  echo '    if [[ "$verbose" == "true" ]]; then' >>  $load_aqua_diagnostics_file
  echo '      echo "=== AQUA Environment Switch ==="' >>  $load_aqua_diagnostics_file
  echo '      echo "Target: AQUA-core"' >>  $load_aqua_diagnostics_file
  echo '      if [[ ${#removed_paths[@]} -gt 0 ]]; then' >>  $load_aqua_diagnostics_file
  echo '        echo "Removed paths:"' >>  $load_aqua_diagnostics_file
  echo '        for path in "${removed_paths[@]}"; do' >>  $load_aqua_diagnostics_file
  echo '          echo "  - $path"' >>  $load_aqua_diagnostics_file
  echo '        done' >>  $load_aqua_diagnostics_file
  echo '      else' >>  $load_aqua_diagnostics_file
  echo '        echo "Removed paths: (none)"' >>  $load_aqua_diagnostics_file
  echo '      fi' >>  $load_aqua_diagnostics_file
      echo '      echo "Added path: \$AQUA_CORE_PATH ($AQUA_CORE_PATH)"' >>  $load_aqua_diagnostics_file
      echo '      echo "Previous PATH component: $original_first"' >>  $load_aqua_diagnostics_file
      echo '      echo "New PATH component: $(echo "$PATH" | cut -d: -f1)"' >>  $load_aqua_diagnostics_file
  echo '      echo "========================================"' >>  $load_aqua_diagnostics_file
  echo '    fi' >>  $load_aqua_diagnostics_file
  echo '    echo "Switched to AQUA-core environment"' >>  $load_aqua_diagnostics_file
  echo '  fi' >>  $load_aqua_diagnostics_file
  echo '}' >>  $load_aqua_diagnostics_file
  echo '' >>  $load_aqua_diagnostics_file
  
  # Function to check which AQUA environment is currently active
  echo '# Function to check which AQUA environment is currently active' >>  $load_aqua_diagnostics_file
  echo '# Returns: "aqua-diagnostics", "aqua-core", or "none"' >>  $load_aqua_diagnostics_file
  echo '# Usage: which_aqua' >>  $load_aqua_diagnostics_file
  echo 'which_aqua() {' >>  $load_aqua_diagnostics_file
  echo '  # Check the first component of PATH (has highest priority)' >>  $load_aqua_diagnostics_file
  echo '  local first_path=$(echo "$PATH" | cut -d: -f1)' >>  $load_aqua_diagnostics_file
  echo '  if [[ "$first_path" == "$AQUA_DIAGNOSTICS_PATH" ]]; then' >>  $load_aqua_diagnostics_file
  echo '    echo "aqua-diagnostics"' >>  $load_aqua_diagnostics_file
  echo '    return 0' >>  $load_aqua_diagnostics_file
  echo '  elif [[ "$first_path" == "$AQUA_CORE_PATH" ]]; then' >>  $load_aqua_diagnostics_file
  echo '    echo "aqua-core"' >>  $load_aqua_diagnostics_file
  echo '    return 0' >>  $load_aqua_diagnostics_file
  echo '  else' >>  $load_aqua_diagnostics_file
  echo '    # If neither is first, check if any aqua path exists in PATH' >>  $load_aqua_diagnostics_file
  echo '    # This handles edge cases where PATH was manually modified' >>  $load_aqua_diagnostics_file
  echo '    if [[ "$PATH" == *"aqua-diagnostics"* ]]; then' >>  $load_aqua_diagnostics_file
  echo '      echo "aqua-diagnostics"' >>  $load_aqua_diagnostics_file
  echo '      return 0' >>  $load_aqua_diagnostics_file
  echo '    elif [[ "$PATH" == *"aqua/bin"* ]]; then' >>  $load_aqua_diagnostics_file
  echo '      echo "aqua-core"' >>  $load_aqua_diagnostics_file
  echo '      return 0' >>  $load_aqua_diagnostics_file
  echo '    else' >>  $load_aqua_diagnostics_file
  echo '      echo "none"' >>  $load_aqua_diagnostics_file
  echo '      return 1' >>  $load_aqua_diagnostics_file
  echo '    fi' >>  $load_aqua_diagnostics_file
  echo '  fi' >>  $load_aqua_diagnostics_file
  echo '}' >>  $load_aqua_diagnostics_file
  echo '' >>  $load_aqua_diagnostics_file
  
  # Default: load AQUA-diagnostics (for backward compatibility)
  echo '# Default: load AQUA-diagnostics environment' >>  $load_aqua_diagnostics_file
  echo 'export PATH="$AQUA_DIAGNOSTICS_PATH:$PATH"' >>  $load_aqua_diagnostics_file
  log_message INFO "AQUA environment switch function and PATH exports have been added to load_aqua_diagnostics.sh."
  log_message INFO "Please run 'source $load_aqua_diagnostics_file' to load the new configuration."
  log_message INFO "Use 'switch_aqua diagnostics' or 'switch_aqua core' to switch between environments. Use 'switch_aqua -v diagnostics' or 'switch_aqua -v core' for verbose output showing path changes."
  log_message INFO "Use 'which_aqua' to check which AQUA environment is currently active."
}

# check if load_aqua_diagnostics_file exist and clean it
if [ -f "$load_aqua_diagnostics_file" ]; then
  log_message $next_level_msg_type "Existing ${load_aqua_diagnostics_file} found. Would you like to remove it? Safer to say yes (y/n) " 
  read -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm $load_aqua_diagnostics_file
    log_message INFO "Existing ${load_aqua_diagnostics_file} removed."

    # Creating the new file
    create_aqua_diagnostics_file
  elif [[ $REPLY =~ ^[Nn]$ ]]; then
    log_message WARNING "Keeping the old $load_aqua_diagnostics_file file. Please make sure it is up to date."
  else
    log_message ERROR "Invalid response. Please enter 'y' or 'n'."
  fi
else
  # Creating the new file
  create_aqua_diagnostics_file
fi

# ask if you want to add this to the bash profile
while true; do
  log_message $next_level_msg_type "Would you like to source $load_aqua_diagnostics_file in your .bash_profile? (y/n) "
  # Read the user's input
  read -n 1 -r
  echo
  case $REPLY in
    [Yy])
      if ! grep -q "source  $load_aqua_diagnostics_file" ~/.bash_profile; then
        echo "source  $load_aqua_diagnostics_file" >> ~/.bash_profile
        log_message INFO 'load_aqua_diagnostics.sh added to your .bash_profile.'
      else
        log_message WARNING 'load_aqua_diagnostics.sh is already in your bash profile, not adding it again!'
      fi
      break
      ;;
    [Nn])
      log_message WARNING "source load_aqua_diagnostics.sh not added to .bash_profile"
      break
      ;;
    *)
      log_message ERROR "Invalid response. Please enter 'y' or 'n'."
      ;;
  esac
done

log_message WARNING "AQUA-diagnostics environment has been installed. Both AQUA (aqua-core) and AQUA-diagnostics are installed in editable mode."
log_message WARNING "You can modify both repositories and changes will be reflected immediately."

