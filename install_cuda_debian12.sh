#!/bin/bash
# Script to install NVIDIA CUDA Toolkit on Debian 12

# --- Configuration ---
CUDA_VERSION="12.8.1"
CUDA_DEB_INSTALLER="cuda-repo-debian12-12-8-local_12.8.1-570.124.06-1_amd64.deb"
CUDA_DEB_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_DEB_INSTALLER}"
CUDA_PATH_DIR="/usr/local/cuda-${CUDA_VERSION%.*}" # e.g., /usr/local/cuda-12.8
CUDA_REPO_DIR="/var/cuda-repo-debian12-12-8-local" # Directory where the deb package extracts files
CUDA_GPG_KEYRING_NAME="cuda-930170B2-keyring.gpg" # Specific GPG key name from NVIDIA's installer
CUDA_GPG_KEYRING_PATH="/usr/share/keyrings/${CUDA_GPG_KEYRING_NAME}"


# --- Functions ---

# Function to display error messages and exit
error_exit() {
    echo -e "\nERROR: $1" >&2
    exit 1
}

# Function to check for root privileges
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root. Please use 'sudo ./install_cuda.sh'"
    fi
}

# Function to check for NVIDIA GPU presence
check_nvidia_gpu() {
    echo "Checking for NVIDIA GPU..."
    if ! lspci | grep -q -i nvidia; then
        echo "WARNING: No NVIDIA GPU detected. CUDA installation might not be useful without compatible hardware."
        read -p "Do you want to continue anyway? (y/N): " response
        if [[ ! "$response" =~ ^[yY]$ ]]; then
            error_exit "Installation aborted by user."
        fi
    else
        echo "NVIDIA GPU detected. Proceeding with installation."
    fi
}

# Function to update APT sources
add_apt_sources() {
    echo "Adding 'contrib non-free' components to /etc/apt/sources.list..."
    # Check if 'contrib non-free' are already present
    if ! grep -q "contrib non-free" /etc/apt/sources.list; then
        # Use sed to add 'contrib non-free' after 'main' in lines starting with 'deb '
        sed -i '/^deb /s/ main/ main contrib non-free/' /etc/apt/sources.list
        echo "Updated /etc/apt/sources.list."
    else
        echo "'contrib non-free' already present in /etc/apt/sources.list."
    fi
    echo "Updating apt package lists..."
    apt update || error_exit "Failed to update apt package lists."
}

# Function to install common prerequisites
install_prerequisites() {
    echo "Installing common prerequisites..."
    # Install wget if not present
    if ! command -v wget &> /dev/null; then
        apt install -y wget || error_exit "Failed to install wget."
    fi

    # Install build tools and kernel headers
    # dkms is crucial for automatic kernel module recompilation on kernel updates
    apt install -y build-essential cmake dkms linux-headers-"$(uname -r)" || error_exit "Failed to install build tools and kernel headers."

    # Install additional libraries for CUDA samples (as per typical guides)
    apt install -y freeglut3-dev libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglfw3-dev libgles2-mesa-dev libglx-dev libopengl-dev || error_exit "Failed to install graphics libraries."
}

# Function to copy the CUDA repository GPG key
copy_cuda_keyring() {
    echo "Copying CUDA repository GPG key..."
    # Find the .gpg key file in the extracted repository directory
    # Using the specific name as it appears in the error, for robustness
    local key_source_file="${CUDA_REPO_DIR}/${CUDA_GPG_KEYRING_NAME}"

    if [[ ! -f "${key_source_file}" ]]; then
        # Fallback to generic find if specific name not found (though less likely for this exact error)
        key_source_file=$(find "${CUDA_REPO_DIR}" -name "cuda-*-keyring.gpg" -print -quit)
        if [[ -z "${key_source_file}" ]]; then
            error_exit "Could not find CUDA GPG key file in ${CUDA_REPO_DIR}. Dpkg installation might have failed to extract it."
        fi
    fi

    cp "${key_source_file}" "${CUDA_GPG_KEYRING_PATH}" || error_exit "Failed to copy CUDA GPG keyring: ${key_source_file} to ${CUDA_GPG_KEYRING_PATH}"
    echo "CUDA GPG key copied successfully: ${key_source_file}"
}

# Function to install CUDA Toolkit
install_cuda() {
    echo "Downloading NVIDIA CUDA Toolkit local installer..."
    # Using --no-clobber to prevent re-download if already exists
    wget -c "${CUDA_DEB_URL}" -O "/tmp/${CUDA_DEB_INSTALLER}" || error_exit "Failed to download CUDA installer."

    # --- CRITICAL FIX: Clean up any old conflicting APT source files ---
    local conflicting_repo_list_file_dpkg="/etc/apt/sources.list.d/cuda-repo-debian12-12-8-local.list"
    local conflicting_repo_list_file_script="/etc/apt/sources.list.d/cuda-repository.list" # From previous script version
    echo "Cleaning up any old or conflicting APT repository list files for CUDA..."
    rm -f "${conflicting_repo_list_file_dpkg}" "${conflicting_repo_list_file_script}" || echo "No old CUDA repo list files to remove (or failed to remove some)."

    echo "Installing CUDA repository package (this unpacks local packages and extracts the GPG key)..."
    dpkg -i --force-overwrite "/tmp/${CUDA_DEB_INSTALLER}" || error_exit "Failed to install CUDA debian package."

    # Copy the GPG key that was extracted by the deb package
    copy_cuda_keyring # This must happen BEFORE creating the .list file with Signed-By
    
    # --- ROBUSTNESS IMPROVEMENT: Manually create/correct the repository .list file with Signed-By ---
    # Using a generic name (cuda-local.list) to avoid direct conflict with dpkg's potential naming.
    # We will ensure this is the *only* active CUDA local repo definition.
    local repo_list_file="/etc/apt/sources.list.d/cuda-local.list" 
    local expected_repo_line="deb [signed-by=${CUDA_GPG_KEYRING_PATH}] file://${CUDA_REPO_DIR}/ /"

    echo "Ensuring APT repository list file (${repo_list_file}) is correctly set up with Signed-By option..."
    # Always recreate this file to ensure it's clean and has the correct Signed-By
    echo "${expected_repo_line}" | tee "${repo_list_file}" > /dev/null || error_exit "Failed to write repository list file."
    echo "Repository list file created/corrected successfully."
    echo "Contents of ${repo_list_file}:"
    cat "${repo_list_file}" # Print contents for debugging

    echo "Cleaning apt cache before update..."
    apt clean # Clear old package lists to ensure a fresh index

    echo "Updating apt package lists for CUDA repository..."
    # Ensure apt update is successful and includes the new repo
    if ! apt update; then
        error_exit "Failed to update apt after adding CUDA repo. The 'E: Conflicting values set for option Signed-By' error suggests an issue with the repository file or keys. Please check ${repo_list_file} and ${CUDA_GPG_KEYRING_PATH} manually."
    fi
    
    echo "Verifying 'cuda' package is discoverable by APT..."
    if ! apt-cache show cuda &> /dev/null; then
        error_exit "Package 'cuda' is still not found in apt cache after update. This indicates the local repository might not be correctly registered or accessible by apt. Check the 'apt update' output carefully for the CUDA repository line and the contents of ${repo_list_file}."
    fi
    echo "'cuda' package found in APT cache. Proceeding with installation."

    echo "Installing CUDA toolkit and NVIDIA drivers..."
    apt install -y cuda || error_exit "Failed to install CUDA toolkit and NVIDIA drivers."
}

# Function to blacklist Nouveau and update initramfs
blacklist_nouveau_and_update_initramfs() {
    echo -e "\nBlacklisting Nouveau (open-source NVIDIA driver) to prevent conflicts..."
    local nouveau_blacklist_file="/etc/modprobe.d/blacklist-nouveau.conf"
    
    if [ ! -f "${nouveau_blacklist_file}" ] || ! grep -q "blacklist nouveau" "${nouveau_blacklist_file}"; then
        echo -e "blacklist nouveau\noptions nouveau modeset=0" | tee "${nouveau_blacklist_file}" > /dev/null || error_exit "Failed to create Nouveau blacklist file."
        echo "Nouveau blacklisted. Rebuilding initramfs..."
        update-initramfs -u || error_exit "Failed to update initramfs after blacklisting Nouveau."
    else
        echo "Nouveau already blacklisted. Skipping initramfs update for Nouveau."
    fi

    echo "Running 'modprobe -r nouveau' to unload Nouveau if it's currently loaded..."
    modprobe -r nouveau 2>/dev/null || echo "Nouveau module not loaded or could not be unloaded."

    echo "Running 'modprobe nvidia_uvm' to attempt loading NVIDIA modules immediately..."
    modprobe nvidia_uvm 2>/dev/null || echo "nvidia_uvm module could not be loaded. This might be expected if a full reboot is needed."

}


# Function to configure environment variables for CUDA
configure_environment() {
    echo "Configuring CUDA environment variables..."
    local profile_file="/etc/profile.d/cuda.sh"
    if [ -f "$profile_file" ]; then
        echo "Existing CUDA environment configuration found in $profile_file. Overwriting."
        # Backup existing file before overwriting, if desired
        # mv "$profile_file" "${profile_file}.bak.$(date +%Y%m%d%H%M%S)"
    fi
    
    # Using 'cat <<EOF' for multi-line export for clarity and robustness
    cat <<EOF > "$profile_file"
export PATH=${CUDA_PATH_DIR}/bin:\$PATH
export LD_LIBRARY_PATH=${CUDA_PATH_DIR}/lib64:\$LD_LIBRARY_PATH
EOF
    
    echo "CUDA environment variables set in $profile_file. Please source this file or reboot for them to take effect."
    echo "Example: source $profile_file"
}

# Function to verify CUDA installation
verify_cuda_installation() {
    echo -e "\nVerifying CUDA installation..."
    echo "Running nvidia-smi:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi || echo "nvidia-smi command failed. Driver issue? A reboot might be required for the driver to load correctly."
        return 1 # Return 1 if nvidia-smi fails
    else
        echo "nvidia-smi not found. NVIDIA driver might not be installed correctly."
        return 1
    fi

    echo -e "\nRunning nvcc --version:"
    # To ensure nvcc is found, we temporarily add CUDA to PATH for this check
    # In a real session, the user would need to source the profile or reboot.
    local temp_path="${CUDA_PATH_DIR}/bin:$PATH"
    if PATH="$temp_path" command -v nvcc &> /dev/null; then
        PATH="$temp_path" nvcc --version || echo "nvcc command failed. CUDA Toolkit issue?"
    else
        echo "nvcc not found. CUDA Toolkit might not be installed correctly or path not set."
        return 1
    fi

    # Only report success if both commands actually work
    if nvidia-smi &> /dev/null && PATH="$temp_path" nvcc --version &> /dev/null; then
        echo -e "\nCUDA installation verification complete. Looks good!"
        return 0
    else
        echo -e "\nCUDA installation verification failed. Please check the logs above for errors."
        return 1
    fi
}

# --- Main Script Execution ---
main() {
    # Set strict error handling
    set -euo pipefail

    check_root
    check_nvidia_gpu

    echo "Starting NVIDIA CUDA Toolkit installation on Debian 12..."

    add_apt_sources
    install_prerequisites
    install_cuda
    blacklist_nouveau_and_update_initramfs
    configure_environment

    echo -e "\n--- Installation Summary ---"
    echo "Attempting immediate post-installation verification (might fail if reboot is needed for drivers)."
    verify_cuda_installation || echo "Initial verification failed. This is common before a reboot."

    echo -e "\nInstallation of NVIDIA CUDA Toolkit is largely complete."
    echo "It is ABSOLUTELY ESSENTIAL to REBOOT your system NOW for all changes (especially kernel modules and Nouveau blacklisting) to take full effect."
    echo "After reboot, you can verify the installation by running 'nvidia-smi' and 'nvcc --version'."
    echo "You may need to log out and log back in, or run 'source /etc/profile.d/cuda.sh' to update your shell's environment variables immediately after reboot."
    echo -e "\nImportant: If 'nvidia-smi' still fails after reboot, check your BIOS/UEFI settings for 'Secure Boot'. If enabled, it might prevent unsigned NVIDIA kernel modules from loading. You may need to disable Secure Boot or manually sign the modules (an advanced topic beyond this script)."

    read -p "Do you want to reboot now? (y/N): " reboot_response
    if [[ "$reboot_response" =~ ^[yY]$ ]]; then
        echo "Rebooting system..."
        reboot
    else
        echo "Reboot postponed. Please remember to reboot soon to finalize the installation."
    fi

    echo -e "\nOptional: To compile and run NVIDIA CUDA samples (for advanced verification):"
    echo "  1. Install Git: sudo apt install -y git"
    echo "  2. Clone samples: git clone https://github.com/nvidia/cuda-samples.git"
    echo "  3. Navigate to a sample (e.g., cd cuda-samples/Samples/5_Domain_Specific/nbody)"
    echo "  4. Build and run: cmake . && make && ./nbody -benchmark"
}

# Execute the main function
main

exit 0
