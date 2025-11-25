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

# Remove the installation path from the $PATH. 
# This is AI-based block which creates a new $PATH removing path including 'aqua-diagnostics'

# Word to check and remove from $PATH
word_to_remove="aqua-diagnostics"

# Function to check if a path contains the specified word
contains_word() {
  [[ "$1" == *"$word_to_remove"* ]]
}

# Split the $PATH into individual components using ":" as the separator
IFS=":" read -ra path_components <<< "$PATH"

# Create a new array to store the modified path components
new_path_components=()

# Loop through each path component and check if it contains the specified word
for component in "${path_components[@]}"; do
  if ! contains_word "$component"; then
    # If the component does not contain the word, add it to the new array
    new_path_components+=("$component")
  fi
done

# Join the new array back into a single string with ":" as the separator
new_path=$(IFS=":"; echo "${new_path_components[*]}")

# Update the $PATH variable with the new value
export PATH="$new_path"

log_message INFO "Paths containing '$word_to_remove' have been removed from \$PATH."

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
    
  log_message INFO "exports for FDB5 added to load_aqua_diagnostics.sh. Please run 'source $load_aqua_diagnostics_file' to load the new configuration."

  # Config GSV: check load_modules_lumi.sh on GSV repo https://earth.bsc.es/gitlab/digital-twins/de_340/gsv_interface/-/blob/main/load_modules_lumi.sh
  echo 'export GSV_WEIGHTS_PATH=/scratch/project_465000454/igonzalez/gsv_weights' >>  $load_aqua_diagnostics_file
  echo 'export GSV_TEST_FILES=/scratch/project_465000454/igonzalez/gsv_test_files' >> $load_aqua_diagnostics_file
  echo 'export GRID_DEFINITION_PATH=/scratch/project_465000454/igonzalez/grid_definitions' >>  $load_aqua_diagnostics_file

  # Currently (Feb 2025) this is the recommended setup overcoming lumi modules
  # This points to a stack with fdb 5.14.0 and the required associated modules
  echo 'export PATH=/appl/local/destine/mars/versions/current/bin:$PATH' >>  $load_aqua_diagnostics_file
  echo 'export LD_LIBRARY_PATH=/appl/local/destine/mars/versions/current/lib64:$LD_LIBRARY_PATH' >>  $load_aqua_diagnostics_file

  log_message INFO "export for GSV has been added to load_aqua_diagnostics.sh. Please run 'source $load_aqua_diagnostics_file' to load the new configuration."

  # Install path
  echo "# AQUA-diagnostics installation path" >>  $load_aqua_diagnostics_file
  echo 'export PATH="'$INSTALLATION_PATH'/bin:$PATH"' >>  $load_aqua_diagnostics_file
  log_message INFO "export PATH has been added to load_aqua_diagnostics.sh. Please run 'source $load_aqua_diagnostics_file' to load the new configuration."
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

log_message WARNING "AQUA-diagnostics environment has been installed."
log_message WARNING "Remember: Both AQUA (aqua-core) and AQUA-diagnostics are installed in editable mode."
log_message WARNING "You can modify both repositories and changes will be reflected immediately."

