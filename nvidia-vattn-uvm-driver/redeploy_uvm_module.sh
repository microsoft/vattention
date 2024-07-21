#!/bin/bash

# This script must be used while development.
# It only removes the loaded nvidia-uvm module and redeploys it
# Assumption: nvidia-uvm and nvidia are already custom deployed modules

# make modules
make modules -j4

# Remove nvidia_uvm module
sudo rmmod nvidia_uvm

# Insert newly compiled module
sudo insmod kernel-open/nvidia-uvm.ko
