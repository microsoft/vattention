#!/bin/bash

# This script must be used on a system reboot to replace propietary nvidia modules
# to custom modules. Just to be safe, this recompiles the modules again --- but this
# step can be skipped.

# Notice the order in which modules are removed (rmmod) and inserted (insmod)

make modules -j4

# Remove nvidia_uvm module
sudo rmmod nvidia_modeset
sudo rmmod nvidia_uvm
sudo rmmod nvidia

# Insert newly compiled module
sudo insmod kernel-open/nvidia.ko
sudo insmod kernel-open/nvidia-uvm.ko
sudo insmod kernel-open/nvidia-modeset.ko
