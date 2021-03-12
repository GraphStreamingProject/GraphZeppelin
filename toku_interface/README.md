# Toku Interface
This directory includes the relevant files for building tokudb and interfacing with it to accomplish graph streaming.

# Installation

## Configuration
### Dependencies
The following dependencies must be installed
- zlib1g-dev 
- libbz4-dev (or libbz2-dev depending on your OS)
- cmake

### Hugepages
Transparent hugepages must be disabled for tokudb to run properly. The following commands run as root will disable them:
```
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag
```

Alternatively, to disable hugepages by default, add the following to the `/etc/rc.local` file:
```
if test -f /sys/kernel/mm/transparent_hugepage/enabled; then
   echo never > /sys/kernel/mm/transparent_hugepage/enabled
fi
if test -f /sys/kernel/mm/transparent_hugepage/defrag; then
   echo never > /sys/kernel/mm/transparent_hugepage/defrag
fi
```

## Compile
Once the configuration is complete you can install tokudb and compile the codebase by running `make install`. This will download the tokudb source code and warn you if your system is not properly configured to run it.

### OS
You will likely need to run this code within linux or some other unix derivative. It got mad at me on OSX.

I have specifically tested the code using Ubuntu_16.04 but it is likely that other versions would work too.

# Known Issues
Sometimes the installation script does not properly setup the environment for running toku. The consequence is that running the `main` executable will throw `error while loading shared libraries`.

To fix this issue run the following command from the `toku_interface` directory (you may want to verify that `LD_LIBRARY_PATH` is empty first):
```
export PATH_TO_INTER=$PWD
export LD_LIBRARY_PATH=$PATH_TO_INTER/ft-index/lib:$PATH_TO_INTER/ft-index/build:$PATH_TO_INTER/ft-index/build/src:$PATH_TO_INTER/ft-index/build/portability
```
