#!/bin/bash
set -ex

#install xxhash
git clone https://github.com/Cyan4973/xxHash.git
cd XXHash
mkdir build
cd build
cmake ../cmake_unofficial
cmake --build .
export xxHash_DIR="$PWD"
cd ../..

#install criterion
sudo add-apt-repository -y ppa:snaipewastaken/ppa
sudo apt-get update
sudo apt-get install criterion-dev

#install StreamingGraphAlgo
git clone https://github.com/Abi1024/StreamingGraphAlgo.git
cd StreamingGraphAlgo
mkdir build
cd build
cmake ..
make

#to run, simply do:
<<<<<<< HEAD
=======
./l0sampling
>>>>>>> d1ab12581f7e9708c5507070b367163e2d5765aa
./tests
