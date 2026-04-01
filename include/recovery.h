#include "bucket.h"
#include "sketch.h"
#include "recovery_types.h"
#include "iblt_cascade.h"
#include <vector>
#include <span>
#include <cmath>

template<typename ItemType = vec_t, typename HashType = vec_hash_t>
class CFRChain {
    private:
        struct RecoveredBucket {
            ItemType alpha;
            HashType gamma;
        };

        size_t universe_size;
        size_t max_recovery_size;
        size_t cleanup_sketch_support;
        bool has_cleanup_sketch;
        bool owns_cleanup_sketch;
        size_t updates_since_recovery_attempt = 0;
        // 1 - 1/2e. TODO - can do better. closer to 1-1/e. for the power-of-two-rounding, 
        // I'm gonna propose 0.69 (comfortably below sqrt(2) so we decrease the size every two levels)
        // static constexpr double reduction_factor = 0.82;
        static constexpr double reduction_factor = 0.69;
        uint64_t _checksum_seed;
        uint64_t seed;
        std::vector<ItemType> recovery_alphas;
        std::vector<HashType> recovery_gammas;
        std::vector<size_t> starter_indices;        
        ItemType deterministic_alpha;
        HashType deterministic_gamma;
        static constexpr double inv_two_e = 1.0 / (2.0 * 2.71828182845904523536);

        inline HashType get_index_hash(const ItemType index, const uint64_t hash_seed) const {
            return (HashType)vec_hash(&index, sizeof(ItemType), hash_seed);
        }

        inline bool is_empty_bucket(size_t level, size_t col) const {
            size_t idx = starter_indices[level] + col;
            return (recovery_alphas[idx] | recovery_gammas[idx]) == 0;
        }

        inline bool is_good_bucket(size_t level, size_t col) const {
            if (is_empty_bucket(level, col)) {
                return false;
            }
            size_t idx = starter_indices[level] + col;
            return recovery_gammas[idx] == get_index_hash(recovery_alphas[idx], checksum_seed());
        }

        inline void xor_bucket(size_t level, size_t col, ItemType alpha, HashType gamma) {
            size_t idx = starter_indices[level] + col;
            recovery_alphas[idx] ^= alpha;
            recovery_gammas[idx] ^= gamma;
        }

        // Shared recovery implementation. If allow_partial is true, may return PARTIAL_RECOVERY.
        RecoveryResult recover_internal(bool allow_partial) {
            if (cleanup_sketch != nullptr) {
                cleanup_sketch->reset_sample_state();
            }
            updates_since_recovery_attempt = 0;

            std::vector<RecoveredBucket> recovered_indices;
            std::vector<vec_t> recovered_return_vals;
            ItemType working_det_alpha = 0;
            HashType working_det_gamma = 0;
            bool met_partial_threshold = true;

            for (size_t cfr_idx=0; cfr_idx < num_levels(); cfr_idx++) {
                auto cfr_size = get_cfr_size(cfr_idx);
                size_t previously_recovered = recovered_indices.size();
                for (size_t i=0; i < previously_recovered; i++) {
                    auto location = get_level_placement(recovered_indices[i].alpha, cfr_idx);
                    xor_bucket(cfr_idx, location, recovered_indices[i].alpha, recovered_indices[i].gamma);
                }

                for (size_t bucket_idx=0; bucket_idx < cfr_size; bucket_idx++) {
                    if (is_good_bucket(cfr_idx, bucket_idx)) {
                        size_t idx = starter_indices[cfr_idx] + bucket_idx;
                        ItemType alpha = recovery_alphas[idx];
                        HashType gamma = recovery_gammas[idx];
                        recovered_indices.push_back({alpha, gamma});
                        recovered_return_vals.push_back((vec_t)alpha);
                        working_det_alpha ^= alpha;
                        working_det_gamma ^= gamma;
                    }
                }

                for (size_t i=0; i < previously_recovered; i++) {
                    auto location = get_level_placement(recovered_indices[i].alpha, cfr_idx);
                    xor_bucket(cfr_idx, location, recovered_indices[i].alpha, recovered_indices[i].gamma);
                }

                // If this level peels too little, we classify this as overloaded recovery.
                size_t recovered_this_level = recovered_indices.size() - previously_recovered;
                size_t min_for_partial = (size_t) std::ceil(inv_two_e * (double) cfr_size);
                if (recovered_this_level < min_for_partial) {
                    met_partial_threshold = false;
                }

                // Early exit: deterministic bucket says all remaining items are recovered.
                if (working_det_alpha == deterministic_alpha && working_det_gamma == deterministic_gamma) {
                    return {SUCCESS, recovered_return_vals};
                }
            }

            // If a cleanup sketch exists, try to finish with it.
            if (cleanup_sketch != nullptr) {
                for (auto idx: recovered_return_vals) {
                    this->update(idx);
                }

                for (size_t i = 0; i < cleanup_sketch->get_num_samples(); i++) {
                    ExhaustiveSketchSample sample = cleanup_sketch->exhaustive_sample();
                    if (sample.result == ZERO) {
                        for (auto idx: recovered_return_vals) {
                            this->update(idx);
                        }
                        return {SUCCESS, recovered_return_vals};
                    }
                    for (auto idx: sample.idxs) {
                        recovered_return_vals.push_back(idx);
                        this->update(idx);
                    }
                }

                // Undo the temporary removals from cleanup probing.
                for (auto idx: recovered_return_vals) {
                    this->update(idx);
                }
            }

            if (allow_partial && met_partial_threshold) {
                return {PARTIAL_RECOVERY, recovered_return_vals};
            }
            return {FAILURE, recovered_return_vals};
        }
    public:
        Sketch *cleanup_sketch;
        CFRChain(size_t universe_size, size_t max_recovery_size, double cleanup_sketch_support_factor, uint64_t seed,
                       bool include_cleanup_sketch = true, Sketch *borrowed_cleanup_sketch = nullptr)
            // TODO - ugly constructor
        // cleanup_sketch(universe_size, seed, ceil(cleanup_sketch_support_factor * log2(universe_size)) * 2, 1)
         {
            this->universe_size = universe_size;
            this->max_recovery_size = max_recovery_size;
            this->seed = seed;

            cleanup_sketch_support = (size_t) std::ceil(cleanup_sketch_support_factor * std::log2((double) universe_size));
            has_cleanup_sketch = false;
            owns_cleanup_sketch = false;
            cleanup_sketch = nullptr;
            if (borrowed_cleanup_sketch != nullptr) {
                cleanup_sketch = borrowed_cleanup_sketch;
                has_cleanup_sketch = true;
            } else if (include_cleanup_sketch && cleanup_sketch_support > 0) {
                cleanup_sketch = new Sketch(universe_size, seed, cleanup_sketch_support, 1);
                has_cleanup_sketch = true;
                owns_cleanup_sketch = true;
            }
            // TODO - define the seed better
            _checksum_seed = this->seed;
            this->seed = this->seed * this->seed + 13;
            starter_indices.reserve(64);
            starter_indices.push_back(0);
            size_t terminal_size = has_cleanup_sketch ? cleanup_sketch_support : 1;
            size_t current_cfr_size = max_recovery_size;
            while (current_cfr_size > 2 * terminal_size) {
                // size_t power_of_two_rounded_size = 1 << (size_t) ceil(log2(current_cfr_size));
                // TODO - examine whether it's better to do something else.
                // ROUND THE SIZE TO A POWER OF TWO -- important for maintaining uniformity.
                // auto current_start_idx = starter_indices[current_cfr_idx++] + power_of_two_rounded_size;
                auto current_start_idx = starter_indices.back() + current_cfr_size;
                starter_indices.push_back(current_start_idx);
                size_t next_size = (size_t) std::ceil(current_cfr_size * reduction_factor);
                if (next_size >= current_cfr_size) {
                    next_size = current_cfr_size - 1;
                }
                current_cfr_size = next_size;
            }
            auto full_storage_size = starter_indices.back();
                        recovery_alphas.resize(full_storage_size);
                        recovery_gammas.resize(full_storage_size);
            reset();
        };

                CFRChain(const CFRChain &other)
            : universe_size(other.universe_size),
              max_recovery_size(other.max_recovery_size),
              cleanup_sketch_support(other.cleanup_sketch_support),
              has_cleanup_sketch(other.has_cleanup_sketch),
              owns_cleanup_sketch(other.owns_cleanup_sketch),
              updates_since_recovery_attempt(other.updates_since_recovery_attempt),
              _checksum_seed(other._checksum_seed),
              seed(other.seed),
                            recovery_alphas(other.recovery_alphas),
                            recovery_gammas(other.recovery_gammas),
              starter_indices(other.starter_indices),
                            deterministic_alpha(other.deterministic_alpha),
                            deterministic_gamma(other.deterministic_gamma),
              cleanup_sketch(nullptr) {
            if (other.cleanup_sketch != nullptr) {
                if (other.owns_cleanup_sketch) {
                    cleanup_sketch = new Sketch(*other.cleanup_sketch);
                    owns_cleanup_sketch = true;
                } else {
                    cleanup_sketch = other.cleanup_sketch;
                    owns_cleanup_sketch = false;
                }
            }
        }

        CFRChain& operator=(const CFRChain &other) {
            if (this == &other) {
                return *this;
            }

            if (owns_cleanup_sketch) {
                delete cleanup_sketch;
            }
            cleanup_sketch = nullptr;

            universe_size = other.universe_size;
            max_recovery_size = other.max_recovery_size;
            cleanup_sketch_support = other.cleanup_sketch_support;
            has_cleanup_sketch = other.has_cleanup_sketch;
            owns_cleanup_sketch = other.owns_cleanup_sketch;
            updates_since_recovery_attempt = other.updates_since_recovery_attempt;
            _checksum_seed = other._checksum_seed;
            seed = other.seed;
            recovery_alphas = other.recovery_alphas;
            recovery_gammas = other.recovery_gammas;
            starter_indices = other.starter_indices;
            deterministic_alpha = other.deterministic_alpha;
            deterministic_gamma = other.deterministic_gamma;

            if (other.cleanup_sketch != nullptr) {
                if (other.owns_cleanup_sketch) {
                    cleanup_sketch = new Sketch(*other.cleanup_sketch);
                    owns_cleanup_sketch = true;
                } else {
                    cleanup_sketch = other.cleanup_sketch;
                    owns_cleanup_sketch = false;
                }
            }
            return *this;
        }

        CFRChain(CFRChain &&other) noexcept
            : universe_size(other.universe_size),
              max_recovery_size(other.max_recovery_size),
              cleanup_sketch_support(other.cleanup_sketch_support),
              has_cleanup_sketch(other.has_cleanup_sketch),
              owns_cleanup_sketch(other.owns_cleanup_sketch),
              updates_since_recovery_attempt(other.updates_since_recovery_attempt),
              _checksum_seed(other._checksum_seed),
              seed(other.seed),
              recovery_alphas(std::move(other.recovery_alphas)),
              recovery_gammas(std::move(other.recovery_gammas)),
              starter_indices(std::move(other.starter_indices)),
              deterministic_alpha(other.deterministic_alpha),
              deterministic_gamma(other.deterministic_gamma),
              cleanup_sketch(other.cleanup_sketch) {
            other.cleanup_sketch = nullptr;
            other.owns_cleanup_sketch = false;
            other.has_cleanup_sketch = false;
        }

        CFRChain& operator=(CFRChain &&other) noexcept {
            if (this == &other) {
                return *this;
            }

            if (owns_cleanup_sketch) {
                delete cleanup_sketch;
            }

            universe_size = other.universe_size;
            max_recovery_size = other.max_recovery_size;
            cleanup_sketch_support = other.cleanup_sketch_support;
            has_cleanup_sketch = other.has_cleanup_sketch;
            owns_cleanup_sketch = other.owns_cleanup_sketch;
            updates_since_recovery_attempt = other.updates_since_recovery_attempt;
            _checksum_seed = other._checksum_seed;
            seed = other.seed;
            recovery_alphas = std::move(other.recovery_alphas);
            recovery_gammas = std::move(other.recovery_gammas);
            starter_indices = std::move(other.starter_indices);
            deterministic_alpha = other.deterministic_alpha;
            deterministic_gamma = other.deterministic_gamma;
            cleanup_sketch = other.cleanup_sketch;

            other.cleanup_sketch = nullptr;
            other.owns_cleanup_sketch = false;
            other.has_cleanup_sketch = false;
            return *this;
        }
    private:
        size_t num_levels() const {
            return starter_indices.size() - 1;
        }
        size_t get_cfr_size(size_t level) const {
            assert(level < starter_indices.size() - 1);
            return starter_indices[level+1] - starter_indices[level];
        }

    public:
        size_t space_usage_bytes() const {
            size_t total = sizeof(CFRChain);
            total += recovery_alphas.capacity() * sizeof(ItemType);
            total += recovery_gammas.capacity() * sizeof(HashType);
            if (cleanup_sketch != nullptr) {
                total += sizeof(Sketch);
                total += cleanup_sketch->bucket_array_bytes();
            }
            return total;
        }
        size_t space_usage_bytes_nocleanup() const {
            size_t total = sizeof(CFRChain);
            total += recovery_alphas.capacity() * sizeof(ItemType);
            total += recovery_gammas.capacity() * sizeof(HashType);
            return total;
        }
        inline uint64_t get_seed() const { return seed; }
        inline uint64_t level_seed(size_t level) const {
          return seed * (2 + seed) + level * 30;
        }
        inline size_t checksum_seed() const { return _checksum_seed; }
        // where in the level this coordinate would go:
        size_t get_level_placement(ItemType coordinate, size_t level) {
            size_t level_size = get_cfr_size(level);
            HashType hash = get_index_hash(coordinate, level_seed(level));
            return hash % level_size;
        }
        void update(const vec_t update) {
            updates_since_recovery_attempt++;
            ItemType item = (ItemType)update;
            HashType checksum = get_index_hash(item, checksum_seed());
            deterministic_alpha ^= item;
            deterministic_gamma ^= checksum;
            for (size_t cfr_idx=0; cfr_idx < num_levels(); cfr_idx++) {
                size_t bucket_idx = get_level_placement(item, cfr_idx);
                xor_bucket(cfr_idx, bucket_idx, item, checksum);
            }
            if (cleanup_sketch != nullptr) {
                cleanup_sketch->update(update);
            }
        }
        void reset() {
            // zero contents of the CFRs
            updates_since_recovery_attempt = 0;
            deterministic_alpha = 0;
            deterministic_gamma = 0;
            std::fill(recovery_alphas.begin(), recovery_alphas.end(), 0);
            std::fill(recovery_gammas.begin(), recovery_gammas.end(), 0);
            if (cleanup_sketch != nullptr) {
                cleanup_sketch->zero_contents();
            }
        };
        

        // THIS IS A NON_DESTRUCTIVE OPERATION
        // (but cannot be marked const)
        RecoveryResult recover() {
            return recover_internal(false);
        };

        // If allow_partial is true, the call can return PARTIAL_RECOVERY.
        RecoveryResult recover(bool allow_partial) {
            return recover_internal(allow_partial);
        }
        void merge(const CFRChain &other) {
            updates_since_recovery_attempt += other.updates_since_recovery_attempt;
            assert(other.recovery_alphas.size() == recovery_alphas.size());
            assert(other.recovery_gammas.size() == recovery_gammas.size());
            for (size_t i=0; i < recovery_alphas.size(); i++) {
                recovery_alphas[i] ^= other.recovery_alphas[i];
                recovery_gammas[i] ^= other.recovery_gammas[i];
            }
            if (cleanup_sketch != nullptr && other.cleanup_sketch != nullptr) {
                cleanup_sketch->merge(*other.cleanup_sketch);
            }
        };

        bool worth_recovery_attempt() const {
            // TODO - remove magic number; more complicated logic, etc.
            // note that this could be done by looking at the cleanup sketch for a cardinality estimate
            return updates_since_recovery_attempt > 1000;
        };

        ~CFRChain() {
            if (owns_cleanup_sketch) {
                delete cleanup_sketch;
            }
        };
};

using SparseRecoveryCFRChain = CFRChain<vec_t, vec_hash_t>;
using SparseRecoveryIBLT = IBLT<vec_t, vec_hash_t>;
using SparseRecoveryIBLTCascade = IBLTCascade<vec_t, vec_hash_t>;

// Backward-compatible default.
using SparseRecovery = SparseRecoveryCFRChain;