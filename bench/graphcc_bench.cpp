#include <benchmark/benchmark.h>
#include "binary_graph_stream.h"

constexpr uint64_t KB = 1024;

// Test the speed of reading all the data in the kron15 graph stream
static void BM_FileIngest(benchmark::State &state) {
	BinaryGraphStream stream("/mnt/ssd2/binary_streams/kron_15_stream_binary", state.range(0));

	uint64_t m = stream.edges();
	GraphUpdate upd;
	while (m--) {
		benchmark::DoNotOptimize(upd = stream.get_edge());
	}
}

BENCHMARK(BM_FileIngest)->Arg(KB)->Arg(32 * KB)->Arg(64 * KB);


BENCHMARK_MAIN();
