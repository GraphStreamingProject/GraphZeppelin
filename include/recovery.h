#include "bucket.h"
#include "sketch.h"
#include <span>

enum RecoveryResultTypes {
    SUCCESS,
    FAILURE
};
struct RecoveryResult {
    RecoveryResultTypes result;
    std::vector<vec_t> recovered_indices;
};


class SparseRecovery {
    private:
        size_t universe_size;
        size_t max_recovery_size;
        size_t cleanup_sketch_support;
        // 1 - 1/2e. TODO - can do better. closer to 1-1/e. for the power-of-two-rounding, 
        // I'm gonna propose 0.69 (comfortably below sqrt(2) so we decrease the size every two levels)
        static constexpr double reduction_factor = 0.82;
        static constexpr double reduction_factor = 0.69;
        uint64_t checksum_seed;
        uint64_t seed;
        // approx 1-1/2e. TODO - can do better. closer to 1-1/e with right
        // bounding parameters
        // TODO - rewrite this for better locality
        // should just be a single array, maybe with a lookup set of pointers for the start of each
        std::vector<Bucket> recovery_buckets;
        std::vector<size_t> starter_indices;        
        Sketch cleanup_sketch;
        // TODO - see if we want to continue maintaining the deterministic bucket
        Bucket deterministic_bucket;
    public:
        SparseRecovery(size_t universe_size, size_t max_recovery_size, double cleanup_sketch_support_factor, uint64_t seed):
            // TODO - ugly constructor
        cleanup_sketch(universe_size, seed, ceil(cleanup_sketch_support_factor * log2(universe_size)) * 2, 1)
         {
            // TODO - define the seed better
            checksum_seed = seed;
            seed = seed * seed + 13;
            universe_size = universe_size;
            max_recovery_size = max_recovery_size;
            starter_indices.reserve(2 + ceil(log2(universe_size) - log2(log2( cleanup_sketch_support_factor * universe_size))));
            starter_indices.push_back(0);
            cleanup_sketch_support = ceil(cleanup_sketch_support_factor * log2(universe_size));
            size_t current_cfr_size = max_recovery_size;
            size_t current_cfr_idx = 0;
            while (current_cfr_size > cleanup_sketch_support) {
                size_t power_of_two_rounded_size = 1 << (size_t) ceil(log2(current_cfr_size));
                // TODO - examine whether it's better to do something else.
                // ROUND THE SIZE TO A POWER OF TWO -- important for maintaining uniformity.
                auto current_start_idx = starter_indices[current_cfr_idx++] + power_of_two_rounded_size;
                starter_indices.push_back(current_start_idx);
                current_cfr_size = ceil(current_cfr_size * reduction_factor);
            }
            auto full_storage_size = starter_indices.back();
            // starter_indices.pop_back();
            recovery_buckets.resize(full_storage_size);
        };
    private:
        size_t num_levels() const {
            return starter_indices.size() - 1;
        }
        size_t get_cfr_size(size_t level) const {
            assert(level < starter_indices.size() - 1);
            return starter_indices[level+1] - starter_indices[level];
        }
        Bucket& get_cfr_bucket(size_t row, size_t col) {
            size_t cfr_start_idx = starter_indices[row];
            return recovery_buckets[cfr_start_idx + col];
        }

    public:
        inline uint64_t get_seed() const { return seed; }
        inline uint64_t level_seed(size_t level) const {
          return seed * (2 + seed) + level * 30;
        }
        inline size_t checksum_seed() const { return seed; }
        void update(const vec_t update) {
            for (size_t cfr_idx=0; cfr_idx < recovery_buckets.size(); cfr_idx++) {
                size_t hash_index = Bucket_Boruvka::get_index_hash(update, cfr_idx * 1231) % get_cfr_size(cfr_idx);
                Bucket_Boruvka::update(get_cfr_bucket(cfr_idx, hash_index), update, checksum_seed());
            }
            cleanup_sketch.update(update);
            Bucket_Boruvka::update(deterministic_bucket, update, checksum_seed());
        }
        void reset() {
            // zero contents of the CFRs
            for (size_t i=0; i < recovery_buckets.size(); i++) {
                recovery_buckets[i] = {0, 0};
            }
            cleanup_sketch.zero_contents();
        };
        // NOTE THAT THIS IS A DESTRUCTIVE OPERATION AT THE MOMENT.
        RecoveryResult recover() {
            std::vector<vec_t> recovered_indices;
            for (size_t cfr_idx=0; cfr_idx < num_levels(); cfr_idx++) {
                // go hunting for good buckets
                auto cfr_size = get_cfr_size(cfr_idx);
                for (size_t bucket_idx=0; bucket_idx < cfr_size; bucket_idx++) {
                    // Bucket &bucket = recovery_buckets[cfr_idx][bucket_idx];
                    Bucket &bucket = get_cfr_bucket(cfr_idx, bucket_idx);
                    if (Bucket_Boruvka::is_good(bucket, checksum_seed())) {
                        recovered_indices.push_back(bucket.alpha);
                        // update it out of the sketch everywhere.
                        this->update(bucket.alpha);

                        // EARLY EXIT CONDITION: deterministic bucket empty
                        if (Bucket_Boruvka::is_empty(deterministic_bucket)) {
                            return {SUCCESS, recovered_indices};
                        }
                    }
                }
                // repeat until we cleared out all the sketches.
            }
            size_t i=0;
            for (; i < cleanup_sketch.get_num_samples(); i++) {
                ExhaustiveSketchSample sample = cleanup_sketch.exhaustive_sample();
                if (sample.result == ZERO) {
                    return {SUCCESS, recovered_indices};
                }
                for (auto idx: sample.idxs) {
                    // todo - checksum stuff. tihs is bad code writing but whatever, anything
                    // to get out of writing psuedocode...
                    recovered_indices.push_back(idx);
                    // todo - this is inefficient. we are recalculating the bucket hash
                    // for literally no reason
                    // but doing things this way is important for undoing our recovery!
                    // otherwise, we're stuck with a bunch of extra bookkeeping 
                    this->update(idx);
                }
            }
            if (i == cleanup_sketch.get_num_samples()) {
                // we ran out of samples
                // TODO - UNDO YOUR RECOVERY!!!
                for (auto idx: recovered_indices) {
                    this->update(idx);
                }
                recovered_indices.clear();
            }
            return {FAILURE, recovered_indices};
        };
        void merge(const SparseRecovery &other) {
            assert(other.recovery_buckets.size() == recovery_buckets.size());
            for (size_t i=0; i < recovery_buckets.size(); i++) {
                recovery_buckets[i] ^= other.recovery_buckets[i];
            }
            cleanup_sketch.merge(other.cleanup_sketch);
        };
        ~SparseRecovery();
};