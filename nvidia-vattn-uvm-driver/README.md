# NVIDIA Linux Open GPU Kernel Module Source for vAttention

This is the source release of the NVIDIA Linux open GPU kernel modules version 545.23.06 for vAttention.

Note: The commands below require sudo privileges.


## Pre-requisite to install vAttention driver module

Uninstall existing driver installations:

```sh
sudo nvidia-uninstall
or
sudo apt-get remove --purge '^nvidia-.*'
```

Download the run file of NVIDIA driver with the same version (545.23.06)

```sh
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
```

Install nvidia driver using the run file. While installation, ensure that the open-source version is enabled. Follow the steps described next after running the following command.

```sh
sudo sh ./cuda_12.3.0_545.23.06_linux.run
```

This will bring the menu to accept terms and conditions. Type accept and continue.
In the next window, move the cursor below to 'Options,' 'Driver Options.'
In this section check the option 'Install the kernel open module flavor,' and exit the menu.
Reference for more details on installing NVIDIA driver from run file:
https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Driver_Installation_Quickstart.pdf


## Build and install driver module for vAttention

Reboot the machine:
```sh
sudo reboot
```


Use the deploy_nvidia_modules.sh script. This will first (1) Build the open-source kernel modules,
(2) Replace installed NVIDIA kernel modules with the modules built in the previous step.

```sh
cd vattention/nvidia-vattn-uvm-driver
sudo ./deploy_nvidia_modules.sh
```

**Note:** Some other modules (e.g., nvidia-drm) may conflict with this step. You can update the script to remove the offending module and insert the built version of it. Check the script for directions.


## Development

Changes to nvidia-uvm module does not require the entire installation process to be repeated. Only the uvm module has to be replaced. Use the redeploy_uvm_modules.sh script. This script will
(1) Build the uvm module, (2) Replace the installed uvm module.

```sh
cd vattention/nvidia-vattn-uvm-driver
sudo ./redeploy_uvm_modules.sh
```

## Verify if the UVM driver has been set up properly

Ensure that path to nvcc is added to your PATH environment variable. Go to the tests directory, compile and run the vattn test script. 

```sh
# temporarily add nvcc to PATH
export PATH=/usr/local/cuda/bin:$PATH
or
# permanently add nvcc to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc

# compile and run unittest
cd vattention/nvidia-vattn-uvm-driver/tests/
make
./vattn
```

If you see **SUCCESS!** at the end, then vAttention is able to allocate smaller (e.g., 64KB) pages and you are set to run e2e LLM serving tests now!
