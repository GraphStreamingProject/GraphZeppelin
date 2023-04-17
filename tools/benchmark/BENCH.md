# GraphStreamingCC Benchmarking
This file provides documentation for the various benchmarks that measure the performance of the GraphZeppelin library. 
These benchmarks use [Google Benchmark](https://github.com/google/benchmark) and are built when the cmake variable BUILD_BENCH is true.

## Usage
`./bench_cc [googlebench args]`  
The optional google benchmark options control which benchmarks are executed, how many times, along with other options.
See their documentation for more details or type `./bench_cc --help` for more usage information.

## Output
`bench_cc` first outputs the name of the benchmark followed by arguments `{BenchName}/{Arg1}/{Arg2}...`.
Then the absolute `Time` and `CPU` time per iteration in addition to the number of iterations performed.
Finally, `UserCounters` gives performance information unique to each benchmark.

## Benchmarks
### Hashing
Measures the performance of a variety of hashing methods against the current method used by `Bucket_Boruvka`. 
All methods hash a single 64 bit input 8 times using different hash seeds.

Example output:
```
--------------------------------------------------------------------------------------
Benchmark                            Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------
BM_Hash_XXH64/1                   20.7 ns         20.7 ns     33589611 Hashes=48.214M/s
BM_Hash_XXH64/100                 1943 ns         1943 ns       361388 Hashes=51.4613M/s
BM_Hash_XXH64/10000             191324 ns       191322 ns         3632 Hashes=52.2679M/s
```
These results indicate that XXH64 can perform 48.2 million hashes per second when hashing 1 update per hash seed.
Additionally they tell us that better hash performance is found when hashing many updates, that all share a hash seed, one after the other.

### Sketch Updates
This benchmark tests the performance of performing sketch updates serially or batched with vectors of different sizes.

Example output:
```
--------------------------------------------------------------------------------------
Benchmark                            Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------
BM_Sketch_Update/16384/0    1083702438 ns   1083665555 ns            1 Update_Rate=9.22794M/s
BM_Sketch_Update/65536/0    1122453196 ns   1122395844 ns            1 Update_Rate=8.90951M/s
BM_Sketch_Update/16384/1    1008273781 ns   1008226959 ns            1 Update_Rate=9.9184M/s
BM_Sketch_Update/65536/1    1005989121 ns   1005924881 ns            1 Update_Rate=9.9411M/s
```
As expected performing updates upon a vector of size 16384 is faster than that of 65536 when updates are applied serially(0).
However, the size of the vector seems not to make the same impact when batching updates to the sketches(1).

### Sketch Queries
Tests the performance of sketch queries with different numbers of updates applied. 
The minimum number of updates per sketch is 1.
In this benchmark 100 sketches with a vector size of approximately 32 thousand each are queried.

Example output:
```
--------------------------------------------------------------------------------------
Benchmark                            Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------
BM_Sketch_Query/0                  502 ns          502 ns      1373192 Query_Rate=199.148M/s
BM_Sketch_Query/10                5365 ns         5365 ns       131777 Query_Rate=18.641M/s
```
These results indicate that somewhat small and sparsely populated sketches can be queried with relatively low latency (5ns).
Once the number of non-zero elements grows this query latency grows quickly.

### DSU Merging
In this test we merge elements in a DSU in a binary tree pattern. 
We first merge singletons, then groups of two, then 4, ...
The number of elements in the DSU is approximately 16 million.

To test the performance of the DSU we create this tree by following two different update patterns. 
The first is 'Adversarial' in that we purposefully join elements that are not roots.
The other one 'Root' is friendly and only joins directly by root.

Example output:
```
------------------------------------------------------------------------------------
Benchmark                          Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------
BM_DSU_Adversarial        2494977485 ns   2494916611 ns            1 Merge_Latency=148.709ns
BM_DSU_Root               1859009603 ns   1858916553 ns            1 Merge_Latency=110.8ns
```
This result tells us that a friendly access pattern that only joins by roots results in an average merge latency reduction of roughly 30%.
This latency is the time required, on average, to merge two sets in the DSU.

### File Ingestion
Tests the speed of reading a graph stream from a file with a variety of buffer sizes.
By default these benchmarks are not enabled. 
There is a flag at the top of `graphcc_bench.cpp` called `FILE_INGEST_F`. 
Defining this flag enables this tests.
This benchmark requires root privileges to flush the file system cache between iterations.

Example output:
```
--------------------------------------------------------------------------------------
Benchmark                            Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------
BM_FileIngest/4096          18296837484 ns   11498983304 ns            1 Ingestion_Rate=97.3513M/s
```
Indicates that a `BinaryGraphStream` with a buffer of 4KiB is capable of ingesting 97 million updates per second.
