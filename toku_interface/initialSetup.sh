#!/bin/bash
set -e # script errors on any failed command
set -o pipefail  # pipeline fails if any part fails
# No -u, we want to take env vars from some places
# No -f, we need to do file globbing a couple times

function add_path()
{
	# $1 path variable
	# $2 path to add
	if [ -d "$2" ] && [[ ":$1:" != *":$2:"* ]]; then
		echo "$1:$2"
	else
		echo "$1"
	fi
}


A=${BASH_SOURCE[0]}
BASE_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "$BASE_DIR"

echo "cflags: '$SCFLAGS'"
echo "basedir: '$BASE_DIR'"

TOKUDB_DIR="$BASE_DIR/ft-index"
BOOST_DIR="$BASE_DIR/boostPrefix"

export CFLAGS="$CFLAGS -I$TOKUDB_DIR/prefix/include"
export CXXFLAGS="$CXXFCLAGS -I$TOKUDB_DIR/prefix/include"
#export CGO_LDFLAGS="$LDFLAGS -L$TOKUDB_DIR/build/portability -L$TOKUDB_DIR/build/src -L$TOKUDB_DIR/lib -ltokuportability -ltokufractaltree"
export LD_LIBRARY_PATH=$(add_path $LD_LIBRARY_PATH $TOKUDB_DIR/lib:$TOKUDB_DIR/build:$TOKUDB_DIR/build/src:$TOKUDB_DIR/build/portability)
#export DYLD_LIBRARY_PATH=$(add_path $DYLD_LIBRARY_PATH $TOKUDB_DIR/lib)

echo "LD LIB PATH"
echo $LD_LIBRARY_PATH

# ======================== BUILD FT-INDEX

#clone ft index and dependencies
if [ ! -d "$TOKUDB_DIR" ]; then
	git clone "git://github.com/Tokutek/ft-index.git" "ft-index"
fi


if [ ! -d "$TOKUDB_DIR/third_party/jemalloc" ]; then
	cd "$TOKUDB_DIR"
	git clone "git://github.com/Tokutek/jemalloc.git" "third_party/jemalloc" 
fi


# Check for dependencies
dependencies="zlib1g-dev libbz2-dev cmake"
unmet=""
for dep in $dependencies; do
	if [[ $(dpkg -s $dep 2>/dev/null) != *"Status: install ok installed"* ]]; then
		unmet+="$dep "
	fi
done
if [ ! -z "$unmet" ]; then
	echo -e "\nError initializing: please install the following dependencies"
	for dep in $unmet; do
		echo -e "\t$dep"
	done
	cd $cwd
	exit 1
fi


#if ft-index is not already built, then build it
if [ ! -d "$TOKUDB_DIR/build" ]; then
	mkdir "$TOKUDB_DIR/build"
	cd "$TOKUDB_DIR/build"
	#Old way CC=gcc47 CXX=g++47 cmake
	#CC=gcc CXX=g++ cmake -D CMAKE_BUILD_TYPE=Debug -D BUILD_TESTING=OFF -D USE_VALGRIND=OFF -D CMAKE_INSTALL_PREFIX=../prefix/ .. 
	CC=gcc CXX=g++ cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTING=OFF -D USE_VALGRIND=OFF -D CMAKE_INSTALL_PREFIX=../prefix/ .. 
	cmake --build . --target install
fi


cd "$BASE_DIR"

# ======================== KINDLY REMIND USER ABOUT HUGEPAGES

thp_cmds=""
if [ -f "/sys/kernel/mm/transparent_hugepage/enabled" ] && [ "$(cat /sys/kernel/mm/transparent_hugepage/enabled)" != "always madvise [never]" ]; then
	thp_cmds+=$'\techo never > /sys/kernel/mm/transparent_hugepage/enabled\n'
fi
if [ -f "/sys/kernel/mm/transparent_hugepage/defrag" ] && [ "$(cat /sys/kernel/mm/transparent_hugepage/defrag)" != "always madvise [never]" ] && [ "$(cat /sys/kernel/mm/transparent_hugepage/defrag)" != "always defer defer+madvise madvise [never]" ]; then
	thp_cmds+=$'\techo never > /sys/kernel/mm/transparent_hugepage/defrag'
fi
if [ ! -z "$thp_cmds" ]; then
	echo -e "\nError initializing: please disable transparent hugepages"
	echo "Run the following commands as root to do so: "
	echo "$thp_cmds"
	exit
fi


# ======================== BUILD CODE
cd "$BASE_DIR"

# Set up build
if [ ! -d "build" ]; then
	mkdir build
fi

make
