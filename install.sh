#!/bin/bash
set -ex

#install xxhash
git clone https://github.com/Cyan4973/xxHash.git
cd XXHash
mkdir build
cd build
cmake ../cmake_unofficial
cmake --build .
exoirt xxHash_DIR="$PWD"
cd ../..


#install StreamingGraphAlgo
git clone https://github.com/Abi1024/StreamingGraphAlgo.git
cd StreamingGraphAlgo
mkdir build
cd build
cmake ..
make

#to run, simply do:
./l0sampling
