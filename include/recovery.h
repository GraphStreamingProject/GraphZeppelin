#include "bucket.h"
#include "sketch.h"

class SparseRecovery {
    private:
        size_t universe_size;
        size_t max_recovery_size;
        size_t cleanup_sketch_support;
        static constexpr double reduction_factor = 0.82; 
        // approx 1-1/2e. TODO - can do better. closer to 1-1/e with right
        // bounding parameters
        // TODO - rewrite this for better locality
        // should just be a single array, maybe with a lookup set of pointers for the start of each
        std::vector<std::vector<Bucket>> recovery_buckets;
        // TODO - see if we want to continue maintaining the deterministic bucket
        Bucket deterministic_bucket;
        Sketch cleanup_sketch;
    public:
        SparseRecovery(size_t universe_size, size_t max_recovery_size, double cleanup_sketch_support_factor, uint64_t seed):
            // TODO - ugly constructor
        cleanup_sketch(universe_size, seed, ceil(cleanup_sketch_support_factor * log2(universe_size)) * 2, 1)
         {
            universe_size = universe_size;
            max_recovery_size = max_recovery_size;
            cleanup_sketch_support = ceil(cleanup_sketch_support_factor * log2(universe_size));
            size_t current_cfr_size = max_recovery_size;
            while (current_cfr_size > cleanup_sketch_support) {
                // doing it this way also deals with zero-initialization
                recovery_buckets.push_back(std::vector<Bucket>(current_cfr_size));
                current_cfr_size = ceil(current_cfr_size * reduction_factor);
            }
        };
        void update(const vec_t update) {
            // TODO - checksum seed agreement. 
              vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update,0);
            for (size_t cfr_idx=0; cfr_idx < recovery_buckets.size(); cfr_idx++) {
                // TODO - get this with an actual function
                size_t hash_index = Bucket_Boruvka::get_index_hash(update, cfr_idx * 1231) % recovery_buckets[cfr_idx].size();
                // recovery_buckets[cfr_idx][hash_index] ^= update;
                Bucket_Boruvka::update(recovery_buckets[cfr_idx][hash_index], update, checksum);
            }
            cleanup_sketch.update(update);
        }
        void reset() {
            // zero contents of the CFRs
            cleanup_sketch.zero_contents();
        };
        // NOTE THAT THIS IS A DESTRUCTIVE OPERATION AT THE MOMENT.
        std::vector<Bucket> recover() {
            std::vector<Bucket> recovered_indices;
            for (size_t cfr_idx=0; cfr_idx < recovery_buckets.size(); cfr_idx++) {
                // first, remove all the already recovered indices
                for (auto recov: recovered_indices) {
                    size_t hash_index = Bucket_Boruvka::get_index_hash(recov.alpha, cfr_idx * 1231) % recovery_buckets[cfr_idx].size();
                    recovery_buckets[cfr_idx][hash_index] ^= recov;
                }
                // now go hunting for good buckets
                for (size_t bucket_idx=0; bucket_idx < recovery_buckets[cfr_idx].size(); bucket_idx++) {
                    Bucket &bucket = recovery_buckets[cfr_idx][bucket_idx];
                    if (Bucket_Boruvka::is_good(bucket, 0)) {
                        recovered_indices.push_back(bucket);
                    }
                }
                // ... repeat until we cleared all the cfrs
            }
            // now, recover from the sketches
            for (auto recov: recovered_indices) {
                cleanup_sketch.update(recov.alpha);
            }
            size_t i=0;
            for (; i < cleanup_sketch.get_num_samples(); i++) {
                ExhaustiveSketchSample sample = cleanup_sketch.exhaustive_sample();
                if (sample.result == ZERO) {
                    break;
                }
                for (auto idx: sample.idxs) {
                    // todo - checksum stuff. tihs is bad code writing but whatever, anything
                    // to get out of writing psuedocode...
                    recovered_indices.push_back({idx, Bucket_Boruvka::get_index_hash(idx, 0)});
                    // todo - this is inefficient. we are recalculating the bucket hash
                    // for literally no reason
                    cleanup_sketch.update(idx);
                }
            }
            if (i == cleanup_sketch.get_num_samples()) {
                // we ran out of samples
                // TODO - UNDO YOUR RECOVERY!!!
            }
            return recovered_indices;
        };
        void merge(const SparseRecovery &other) {
            // TODO - xor together all the CFRs
            cleanup_sketch.merge(other.cleanup_sketch);
        };
        ~SparseRecovery();
};