#!/bin/bash

src=$(dirname "$(realpath "$0")")
source $src/helpers/common.sh

download_libtorch() {
    (
        cd $src/source
        rm -rf libtorch*
        wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.3.0%2Bcu121.zip
        unzip libtorch-shared-with-deps-2.3.0+cu121.zip
    )
}

install_sarathi_lean() {
    (
        cd $root/sarathi-lean
        pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
        #pip install -r requirements.txt --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
        #python setup.py develop
    )
}

install_vattention() {
    (
        cd $root/vattention
        LIBTORCH_PATH=$src/source/libtorch python setup.py install
    )
}

install_vllm() {
    (
        cd $src/source
        rm -rf vllm
        git clone --branch v0.3.0 https://github.com/vllm-project/vllm.git
        cd vllm
        git switch -c v0.3.0
        git am $src/source/patches/*.patch
        python setup.py install
    )
}

echo "========================================"
echo "Step 1: Downloading libtorch..."
echo "========================================"
download_libtorch || { echo "Error: Failed to download libtorch"; exit 1; }

echo "========================================"
echo "Step 2: Installing sarathi-lean..."
echo "========================================"
install_sarathi_lean || { echo "Error: Failed to install sarathi-lean"; exit 1; }

echo "========================================"
echo "Step 3: Installing vattention..."
echo "========================================"
install_vattention || { echo "Error: Failed to install vattention"; exit 1; }

echo "========================================"
echo "Step 4: Installing vllm..."
echo "========================================"
install_vllm || { echo "Error: Failed to install vllm"; exit 1; }

echo "========================================"
echo "Installation completed successfully!"
echo "========================================"
