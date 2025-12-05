#!/bin/bash

set -euo pipefail

####################################################################################
##### Marimo Setup for Brev
####################################################################################
# Defaults to cloning marimo-team/examples repository
# Set MARIMO_REPO_URL to use your own notebooks repository
# Set MARIMO_REPO_URL="" to skip cloning entirely
####################################################################################

# Detect the actual Brev user dynamically
# This handles ubuntu, nvidia, shadeform, or any other user
# Uses Brev-specific markers to identify the correct user
if [ -z "${USER:-}" ] || [ "${USER:-}" = "root" ]; then
    # Check if run via sudo first
    if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
        USER="$SUDO_USER"
    else
        # Find actual Brev user by checking for Brev-specific markers:
        # 1. .lifecycle-script-ls-*.log files (unique to Brev user)
        # 2. .verb-setup.log file (Brev-specific)
        # 3. .cache symlink to /ephemeral/cache
        DETECTED_USER=""
        
        # First pass: Look for Brev lifecycle script logs (most reliable)
        for user_home in /home/*; do
            username=$(basename "$user_home")
            # Check for Brev lifecycle script log files
            if ls "$user_home"/.lifecycle-script-ls-*.log 2>/dev/null | grep -q .; then
                DETECTED_USER="$username"
                break
            fi
            # Check for Brev verb setup log
            if [ -f "$user_home/.verb-setup.log" ]; then
                DETECTED_USER="$username"
                break
            fi
        done
        
        # Second pass: Check for .cache symlink to /ephemeral/cache
        if [ -z "$DETECTED_USER" ]; then
            for user_home in /home/*; do
                username=$(basename "$user_home")
                if [ -L "$user_home/.cache" ] && [ "$(readlink "$user_home/.cache")" = "/ephemeral/cache" ]; then
                    DETECTED_USER="$username"
                    break
                fi
            done
        fi
        
        # Third pass: Use UID check, but skip known service users
        if [ -z "$DETECTED_USER" ]; then
            for user_home in /home/*; do
                username=$(basename "$user_home")
                # Skip known service users
                if [ "$username" = "launchpad" ]; then
                    continue
                fi
                # Check if user has UID >= 1000 (interactive user)
                if id "$username" &>/dev/null; then
                    user_uid=$(id -u "$username" 2>/dev/null || echo 0)
                    if [ "$user_uid" -ge 1000 ]; then
                        DETECTED_USER="$username"
                        break
                    fi
                fi
            done
        fi
        
        # Fall back to known common users if all detection fails
        if [ -z "$DETECTED_USER" ]; then
            if [ -d "/home/nvidia" ]; then
                DETECTED_USER="nvidia"
            elif [ -d "/home/ubuntu" ]; then
                DETECTED_USER="ubuntu"
            else
                DETECTED_USER="ubuntu"
            fi
        fi
        USER="$DETECTED_USER"
    fi
fi

# Force HOME to be the detected user's home directory
# Don't use ${HOME:-...} because HOME is already set to /root when running as root
HOME="/home/$USER"

REPO_URL="${MARIMO_REPO_URL:-https://github.com/marimo-team/examples.git}"
NOTEBOOKS_DIR="${MARIMO_NOTEBOOKS_DIR:-sglang-brevdev}"
NOTEBOOKS_COPIED=0

(echo ""; echo "##### Detected Environment #####"; echo "";)
(echo "User: $USER"; echo "";)
(echo "Home: $HOME"; echo "";)

##### Detect and configure CUDA environment #####
(echo ""; echo "##### Detecting CUDA installation #####"; echo "";)

CUDA_HOME_FOUND=""
CUDA_LIB_PATH=""

# Wrap CUDA detection in error handling to prevent bootstrap failure
set +e  # Don't exit on errors

# Check if CUDA runtime library exists
if ls /usr/lib/x86_64-linux-gnu/libcudart.so* 2>/dev/null | grep -q .; then
    CUDA_HOME_FOUND="/usr"
    CUDA_LIB_PATH="/usr/lib/x86_64-linux-gnu"
    echo "✅ Found CUDA runtime library at: $CUDA_LIB_PATH"
else
    # Check common CUDA installation paths
    CUDA_PATHS=(
        "/usr/local/cuda"
        "/usr/local/cuda-12.1"
        "/usr/local/cuda-12.0"
        "/usr/local/cuda-11.8"
        "/opt/cuda"
    )
    
    for cuda_path in "${CUDA_PATHS[@]}"; do
        if [ -d "$cuda_path" ] && [ -d "$cuda_path/lib64" ]; then
            if ls "$cuda_path/lib64"/libcudart.so* 2>/dev/null | grep -q .; then
                CUDA_HOME_FOUND="$cuda_path"
                CUDA_LIB_PATH="$cuda_path/lib64"
                echo "✅ Found CUDA installation at: $CUDA_HOME_FOUND"
                break
            fi
        fi
    done
    
    # If still not found, try to install CUDA runtime library (non-blocking)
    if [ -z "$CUDA_HOME_FOUND" ]; then
        echo "⚠️  CUDA runtime library not found, attempting to install..."
        sudo apt-get update -qq 2>&1 | grep -v "WARNING: apt does not have a stable CLI interface" || true
        if sudo apt-get install -y libcudart11.0 2>&1 | grep -v "WARNING: apt does not have a stable CLI interface" | grep -q "Setting up"; then
            CUDA_HOME_FOUND="/usr"
            CUDA_LIB_PATH="/usr/lib/x86_64-linux-gnu"
            echo "✅ Installed CUDA runtime library (libcudart11.0)"
        else
            echo "⚠️  Could not install CUDA runtime library automatically"
            echo "   Notebooks use Triton backend which doesn't require CUDA runtime library"
        fi
    fi
fi

set -e  # Re-enable error exit

# Set CUDA environment variables if found
if [ -n "$CUDA_HOME_FOUND" ]; then
    export CUDA_HOME="$CUDA_HOME_FOUND"
    export LD_LIBRARY_PATH="$CUDA_LIB_PATH:${LD_LIBRARY_PATH:-}"
    
    # Add to shell config files for interactive use
    if ! grep -q "CUDA_HOME" "$HOME/.bashrc" 2>/dev/null; then
        echo "export CUDA_HOME=\"$CUDA_HOME_FOUND\"" >> "$HOME/.bashrc"
        echo "export LD_LIBRARY_PATH=\"$CUDA_LIB_PATH:\$LD_LIBRARY_PATH\"" >> "$HOME/.bashrc"
    fi
    if ! grep -q "CUDA_HOME" "$HOME/.zshrc" 2>/dev/null; then
        echo "export CUDA_HOME=\"$CUDA_HOME_FOUND\"" >> "$HOME/.zshrc"
        echo "export LD_LIBRARY_PATH=\"$CUDA_LIB_PATH:\$LD_LIBRARY_PATH\"" >> "$HOME/.zshrc"
    fi
    
    echo "✅ CUDA environment variables configured"
    echo "   CUDA_HOME=$CUDA_HOME"
    echo "   LD_LIBRARY_PATH includes $CUDA_LIB_PATH"
fi

# Check for CUDA compiler (nvcc) - optional since notebooks use Triton backend
NVCC_PATH=""
set +e  # Don't exit on errors

if command -v nvcc &> /dev/null; then
    NVCC_PATH=$(command -v nvcc)
    echo "✅ CUDA compiler (nvcc) found: $NVCC_PATH"
else
    echo ""
    echo "ℹ️  CUDA compiler (nvcc) not found"
    echo "   Notebooks use Triton backend which doesn't require nvcc"
    echo "   (Optional) Attempting to install CUDA development toolkit..."
    
    # Try to install nvcc via nvidia-cuda-toolkit package (non-blocking)
    sudo apt-get update -qq 2>&1 | grep -v "WARNING: apt does not have a stable CLI interface" || true
    INSTALL_OUTPUT=$(sudo apt-get install -y nvidia-cuda-toolkit 2>&1)
    INSTALL_EXIT=$?
    
    if [ $INSTALL_EXIT -eq 0 ] && echo "$INSTALL_OUTPUT" | grep -q "Setting up"; then
        # Installation succeeded - refresh PATH and check again
        export PATH="/usr/local/cuda/bin:/usr/local/cuda-12.1/bin:/usr/local/cuda-12.0/bin:/usr/local/cuda-11.8/bin:$PATH"
        if command -v nvcc &> /dev/null; then
            NVCC_PATH=$(command -v nvcc)
            echo "✅ Installed CUDA toolkit (nvcc available at: $NVCC_PATH)"
        else
            # Check common installation locations after apt install
            for cuda_bin in /usr/local/cuda/bin/nvcc /usr/local/cuda-12.1/bin/nvcc /usr/local/cuda-12.0/bin/nvcc /usr/local/cuda-11.8/bin/nvcc /usr/bin/nvcc; do
                if [ -f "$cuda_bin" ]; then
                    NVCC_PATH="$cuda_bin"
                    echo "✅ Found nvcc at: $NVCC_PATH"
                    break
                fi
            done
        fi
    else
        echo "⚠️  Could not install CUDA toolkit automatically (this is optional)"
        echo "   Checking for existing nvcc in common locations..."
        
        # Check common installation locations even if apt install failed
        for cuda_bin in /usr/local/cuda/bin/nvcc /usr/local/cuda-12.1/bin/nvcc /usr/local/cuda-12.0/bin/nvcc /usr/local/cuda-11.8/bin/nvcc; do
            if [ -f "$cuda_bin" ]; then
                NVCC_PATH="$cuda_bin"
                echo "✅ Found nvcc at: $NVCC_PATH"
                break
            fi
        done
        
        # Try conda if available
        if [ -z "$NVCC_PATH" ] && command -v conda &> /dev/null; then
            echo "   Attempting to install via conda..."
            if conda install -c nvidia -y cuda-toolkit 2>/dev/null; then
                export PATH="$CONDA_PREFIX/bin:$PATH"
                if command -v nvcc &> /dev/null; then
                    NVCC_PATH=$(command -v nvcc)
                    echo "✅ Installed CUDA toolkit via conda (nvcc available at: $NVCC_PATH)"
                fi
            fi
        fi
        
        if [ -z "$NVCC_PATH" ]; then
            echo "ℹ️  nvcc not found (not required - notebooks use Triton backend)"
            echo "   Options:"
            echo "   1. Install manually: sudo apt-get install -y nvidia-cuda-toolkit"
            echo "   2. Install via conda: conda install -c nvidia cuda-toolkit"
            echo "   3. Download CUDA toolkit from: https://developer.nvidia.com/cuda-downloads"
            echo "   4. Notebooks already use Triton backend (no action needed)"
        fi
    fi
fi

set -e  # Re-enable error exit

# Ensure nvcc is accessible at /usr/bin/nvcc (FlashInfer hardcodes this path)
if [ -n "$NVCC_PATH" ] && [ -f "$NVCC_PATH" ]; then
    NVCC_BIN_DIR=$(dirname "$NVCC_PATH")
    
    # Add CUDA bin directory to PATH
    export PATH="$NVCC_BIN_DIR:$PATH"
    
    # Create symlink if nvcc is not at /usr/bin/nvcc
    if [ "$NVCC_PATH" != "/usr/bin/nvcc" ]; then
        echo "   Creating symlink: /usr/bin/nvcc -> $NVCC_PATH"
        if sudo ln -sf "$NVCC_PATH" /usr/bin/nvcc 2>/dev/null; then
            echo "   ✅ Symlink created successfully"
        else
            echo "   ⚠️  Failed to create symlink (may need manual creation)"
        fi
    else
        echo "   ✅ nvcc already at /usr/bin/nvcc"
    fi
    
    # Verify symlink exists
    if [ ! -f "/usr/bin/nvcc" ] && [ ! -L "/usr/bin/nvcc" ]; then
        echo "   ⚠️  Warning: /usr/bin/nvcc still not accessible"
    fi
    
    # Update CUDA_HOME_FOUND to include bin directory for PATH
    if [ -z "$CUDA_HOME_FOUND" ]; then
        CUDA_HOME_FOUND=$(dirname "$NVCC_BIN_DIR")
        if [ -d "$CUDA_HOME_FOUND/lib64" ]; then
            CUDA_LIB_PATH="$CUDA_HOME_FOUND/lib64"
        elif [ -d "$CUDA_HOME_FOUND/lib" ]; then
            CUDA_LIB_PATH="$CUDA_HOME_FOUND/lib"
        fi
    fi
fi

##### Install Python and pip if not available #####
if ! command -v pip3 &> /dev/null; then
    (echo ""; echo "##### Installing Python and pip3 #####"; echo "";)
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv
fi

##### Install Marimo #####
(echo ""; echo "##### Installing Marimo #####"; echo "";)
pip3 install --upgrade marimo

##### Add to PATH #####
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc" 2>/dev/null || true
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc" 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

##### Clone notebooks if URL provided #####
if [ -n "$REPO_URL" ]; then
    (echo ""; echo "##### Cloning notebooks from $REPO_URL #####"; echo "";)
    cd "$HOME"
    git clone "$REPO_URL" "$NOTEBOOKS_DIR" 2>/dev/null || echo "Repository already exists"
    
    # Install dependencies if requirements.txt exists
    if [ -f "$HOME/$NOTEBOOKS_DIR/requirements.txt" ]; then
        (echo ""; echo "##### Installing additional dependencies from requirements.txt #####"; echo "";)
        pip3 install -r "$HOME/$NOTEBOOKS_DIR/requirements.txt"
    fi
fi

##### Install PyTorch with CUDA support and common packages #####
(echo ""; echo "##### Installing PyTorch with CUDA support #####"; echo "";)
# Install latest PyTorch from CUDA 12.1 index (torch-tensorrt compatibility handled separately)
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Fix PyTorch 2.9.1 + CuDNN compatibility issue
# PyTorch 2.9.1 has a known bug with CuDNN < 9.15 that causes performance degradation
# Upgrade CuDNN to 9.16.0.29 to resolve this issue (required for SGLang)
(echo ""; echo "##### Fixing PyTorch 2.9.1 + CuDNN compatibility #####"; echo "";)
pip3 install --no-cache-dir nvidia-cudnn-cu12==9.16.0.29 2>&1 | grep -v -E "ERROR: pip's dependency resolver" || true

# Install common packages for marimo examples
(echo ""; echo "##### Installing common packages for marimo examples #####"; echo "";)
# Note: openai pinned to 2.6.1 for sglang compatibility, jsonschema>=4.0.0 for sglang
# Note: nvidia-ml-py replaces deprecated pynvml package
pip3 install --no-cache-dir --upgrade \
    polars altair plotly pandas numpy scipy scikit-learn \
    matplotlib seaborn pyarrow "openai==2.6.1" anthropic requests \
    beautifulsoup4 pillow 'marimo[sql]' duckdb sqlalchemy \
    instructor mohtml openai-whisper opencv-python python-dotenv \
    wigglystuff yt-dlp psutil pynvml GPUtil nvidia-ml-py \
    transformers networkx diffusers accelerate safetensors \
    "jsonschema>=4.0.0"

# Optional: Install TensorRT-related packages if CUDA is available
(echo ""; echo "##### Installing optional NVIDIA packages (TensorRT, etc.) #####"; echo "";)
# Suppress dependency conflict warnings (torch-tensorrt may have version constraints)
set +o pipefail  # Temporarily disable pipefail to allow grep filtering
pip3 install --no-cache-dir --upgrade torch-tensorrt 2>&1 | grep -v -E "WARNING: Error parsing dependencies|ERROR: pip's dependency resolver" || true; TENSORRT_EXIT=${PIPESTATUS[0]}
set -o pipefail  # Re-enable pipefail
if [ "${TENSORRT_EXIT:-0}" -ne 0 ]; then
    echo "  torch-tensorrt not available (needs TensorRT installed or version conflict)"
fi

# Note: RAPIDS packages (cudf, cugraph) require conda installation
# These are optional - notebooks will fall back to CPU equivalents if not available
# To install RAPIDS:
#   conda install -c rapidsai -c conda-forge -c nvidia cudf=24.08 cugraph=24.08 python=3.11 cuda-version=12.0

##### Ensure notebooks directory exists with proper permissions #####
(echo ""; echo "##### Ensuring notebooks directory exists #####"; echo "";)

# Create directory as the target user to avoid permission issues
if [ "$(id -u)" -eq 0 ] && [ -n "$USER" ]; then
    # Running as root - create directory as target user
    # First ensure HOME directory exists and has proper permissions
    mkdir -p "$HOME"
    chown "$USER:$USER" "$HOME" 2>/dev/null || true
    
    # Create notebooks directory as the target user
    sudo -u "$USER" mkdir -p "$HOME/$NOTEBOOKS_DIR" 2>/dev/null || mkdir -p "$HOME/$NOTEBOOKS_DIR"
    
    # Ensure proper ownership and permissions
    chown -R "$USER:$USER" "$HOME/$NOTEBOOKS_DIR"
    chmod -R 755 "$HOME/$NOTEBOOKS_DIR"
    echo "Created $HOME/$NOTEBOOKS_DIR as user $USER"
else
    # Running as regular user
    mkdir -p "$HOME/$NOTEBOOKS_DIR"
    echo "Created $HOME/$NOTEBOOKS_DIR"
fi

##### Create systemd service for Marimo #####
(echo ""; echo "##### Setting up Marimo systemd service #####"; echo "";)

# Build environment variables for systemd service
SERVICE_ENV="Environment=\"PATH=/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin\"
Environment=\"HOME=$HOME\"
Environment=\"MARIMO_PORT=${MARIMO_PORT:-8080}\""

# Add CUDA environment variables if CUDA was found
if [ -n "$CUDA_HOME_FOUND" ]; then
    # Determine CUDA bin directory
    CUDA_BIN_DIR=""
    if command -v nvcc &> /dev/null; then
        NVCC_PATH=$(command -v nvcc)
        CUDA_BIN_DIR=$(dirname "$NVCC_PATH")
    elif [ -d "$CUDA_HOME_FOUND/bin" ]; then
        CUDA_BIN_DIR="$CUDA_HOME_FOUND/bin"
    fi
    
    SERVICE_ENV="$SERVICE_ENV
Environment=\"CUDA_HOME=$CUDA_HOME_FOUND\"
Environment=\"LD_LIBRARY_PATH=$CUDA_LIB_PATH:\${LD_LIBRARY_PATH:-}\""
    
    # Add CUDA bin to PATH if found
    if [ -n "$CUDA_BIN_DIR" ]; then
        SERVICE_ENV="$SERVICE_ENV
Environment=\"PATH=$CUDA_BIN_DIR:/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin\""
    fi
fi

sudo tee /etc/systemd/system/marimo.service > /dev/null << EOF
[Unit]
Description=Marimo Notebook Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/$NOTEBOOKS_DIR
$SERVICE_ENV
ExecStart=$HOME/.local/bin/marimo edit --host 0.0.0.0 --port \${MARIMO_PORT} --headless --no-token
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=marimo

[Install]
WantedBy=multi-user.target
EOF

##### Fix ownership of shell config files if running as root #####
if [ "$(id -u)" -eq 0 ] && [ -n "$USER" ]; then
    chown -R "$USER:$USER" "$HOME/.bashrc" "$HOME/.zshrc" 2>/dev/null || true
fi

##### Enable and start Marimo service #####
(echo ""; echo "##### Enabling and starting Marimo service #####"; echo "";)
sudo systemctl daemon-reload
sudo systemctl enable marimo.service 2>/dev/null || true
# Use restart to ensure service picks up new environment variables (including CUDA vars)
sudo systemctl restart marimo.service 2>/dev/null || sudo systemctl start marimo.service

# Wait for service to start
sleep 2

(echo ""; echo ""; echo "==============================================================="; echo "";)
(echo "  Setup Complete! Marimo is now running"; echo "";)
(echo "==============================================================="; echo "";)
(echo ""; echo "Notebooks Location: $HOME/$NOTEBOOKS_DIR"; echo "";)
(echo "Access URL: http://localhost:${MARIMO_PORT:-8080}"; echo "";)
if [ "$NOTEBOOKS_COPIED" -gt 0 ]; then
    (echo "Custom Notebooks: $NOTEBOOKS_COPIED notebook(s) added"; echo "";)
fi
(echo ""; echo "⚠️  OPEN THIS PORT ON BREV: ${MARIMO_PORT:-8080}/tcp"; echo "";)
(echo ""; echo "Useful commands:"; echo "";)
(echo "  - Check status:  sudo systemctl status marimo"; echo "";)
(echo "  - View logs:     sudo journalctl -u marimo -f"; echo "";)
(echo "  - Restart:       sudo systemctl restart marimo"; echo "";)
(echo ""; echo "==============================================================="; echo "";)

##### Install sglang and FlashInfer #####
(echo ""; echo "##### Installing FlashInfer and sglang #####"; echo "";)
# Upgrade pip first (as recommended by SGLang docs)
pip3 install --upgrade pip 2>&1 | grep -v "WARNING: Error parsing dependencies" || true

# Install FlashInfer (default attention kernel backend for SGLang)
# FlashInfer supports sm75 and above (T4, A10, A100, L4, L40S, H100, etc.)
# Note: If you encounter FlashInfer issues on L40S or other sm75+ devices,
# you can switch to alternative backends when running SGLang:
#   --attention-backend triton --sampling-backend pytorch
# Installation is non-blocking - SGLang can use alternative backends if FlashInfer fails
set +o pipefail  # Temporarily disable pipefail to allow grep filtering
pip3 install --no-cache-dir --upgrade flashinfer-python 2>&1 | grep -v "WARNING: Error parsing dependencies"; PIPESTATUS_EXIT=${PIPESTATUS[0]}
set -o pipefail  # Re-enable pipefail
if [ "$PIPESTATUS_EXIT" -ne 0 ]; then
    echo "  Warning: FlashInfer installation failed or encountered issues"
    echo "  SGLang will use alternative backends (triton/pytorch) if needed"
    echo "  You can retry FlashInfer later or use --attention-backend triton when running SGLang"
fi

# Install SGLang
# Using --pre to allow pre-release versions as recommended
# Suppress dependency conflict warnings (openai version is pinned above)
pip3 install --no-cache-dir --upgrade --pre "sglang" 2>&1 | grep -v -E "WARNING: Error parsing dependencies|ERROR: pip's dependency resolver" || true
pip install --upgrade filelock

# Note: If FlashInfer has issues, you can reinstall it with:
# pip3 install --upgrade flashinfer-python --force-reinstall --no-deps
# Then delete cache: rm -rf ~/.cache/flashinfer
