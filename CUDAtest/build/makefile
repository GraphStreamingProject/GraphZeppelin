file = main
cudaFile = cudaSketch
xxhash_path = "../../build/_deps/xxhash-src"
xxhashlib_path = "../../build/lib/libxxhash.a"
gtest_path = "../../build/_deps/googletest-src/googletest/include"
gtestlib_path = "../../build/lib/libgtest.a"

all:
	g++ -c $(file).cpp -I$(xxhash_path) -I$(gtest_path)
	nvcc -c ../src/$(cudaFile).cu
	nvcc -o $(file) $(file).o $(cudaFile).o $(xxhashlib_path) $(gtestlib_path)

clean:
	rm -f $(file) $(file).o $(cudaFile).o
