#pragma once

#include <vector>
#include <xxhash.h>
#include <iostream>
#include <cstdint>
#include <cassert>
#include <cmath>
#include "types.h"
#include "recovery_types.h"
#include "sketch.h"

// Template for ItemType to allow 32-bit vs 64-bit elements (e.g. node_id_t vs vec_t)
template<typename ItemType = vec_t, typename HashType = vec_hash_t>
class IBLT {
private:
    size_t capacity;
    size_t num_hashes;
    size_t universe_size;
    size_t max_recovery_size;
    size_t cleanup_sketch_support;
    bool has_cleanup_sketch;
    bool owns_cleanup_sketch;
    long seed;

    static constexpr double capacity_factor = 1.35;
    // static constexpr double capacity_factor = 1.001;

    std::vector<ItemType> alphas;
    std::vector<HashType> gammas;

    // Deterministic checking bucket
    ItemType det_alpha;
    HashType det_gamma;

    // Hashes an item into a HashType checksum (e.g. 64-bit)
    inline HashType get_item_hash(const ItemType item_idx) const {
        return (HashType)(XXH3_128bits_withSeed(&item_idx, sizeof(ItemType), seed)).low64;
    }

    // Computes k unique bucket indices for an item, re-hashing on collision.
    inline void get_bucket_indices(const ItemType item_idx, size_t *indices) const {
        assert(capacity > 0);
        assert(num_hashes > 0);
        assert(capacity > num_hashes);
        for (size_t i = 0; i < num_hashes; ++i) {
            size_t attempt = i;
            bool unique;
            do {
                auto hash = XXH3_128bits_withSeed(&item_idx, sizeof(ItemType), seed + attempt);
                indices[i] = hash.low64 % capacity;
                unique = true;
                for (size_t j = 0; j < i; ++j) {
                    if (indices[j] == indices[i]) {
                        unique = false;
                        attempt += num_hashes;
                        break;
                    }
                }
            } while (!unique);
        }
    }

    inline bool is_empty(size_t idx) const {
        return (alphas[idx] | gammas[idx]) == 0;
    }

    inline bool is_good(size_t idx) const {
        return !is_empty(idx) && get_item_hash(alphas[idx]) == gammas[idx];
    }

    // Shared recovery implementation. If allow_partial is true, may return PARTIAL_RECOVERY.
    RecoveryResult recover_internal(bool allow_partial) {
        if (cleanup_sketch != nullptr) {
            cleanup_sketch->reset_sample_state();
        }

        std::vector<vec_t> recovered_items;
        std::vector<size_t> good_buckets;

        ItemType working_det_alpha = 0;
        HashType working_det_gamma = 0;

        // push good buckets onto queue
        for (size_t i = 0; i < capacity; ++i) {
            if (is_good(i)) {
                good_buckets.push_back(i);
            }
        }

        // Peel good buckets
        while (!good_buckets.empty()) {
            size_t idx = good_buckets.back();
            good_buckets.pop_back();

            if (!is_good(idx)) continue;

            ItemType item = alphas[idx];
            HashType item_hash = get_item_hash(item);

            recovered_items.push_back((vec_t)item);
            working_det_alpha ^= item;
            working_det_gamma ^= item_hash;

            // Remove from IBLT buckets
            size_t indices[num_hashes];
            get_bucket_indices(item, indices);
            for (size_t i = 0; i < num_hashes; ++i) {
                alphas[indices[i]] ^= item;
                gammas[indices[i]] ^= item_hash;
            }

            // Check if removing this item created new good buckets
            for (size_t i = 0; i < num_hashes; ++i) {
                if (is_good(indices[i])) {
                    good_buckets.push_back(indices[i]);
                }
            }

            // Early exit: deterministic bucket says all items recovered
            if (working_det_alpha == det_alpha && working_det_gamma == det_gamma) {
                // Undo changes so recover is non-destructive
                for (auto ri : recovered_items) {
                    ItemType it = (ItemType)ri;
                    HashType ih = get_item_hash(it);
                    size_t ri_indices[num_hashes];
                    get_bucket_indices(it, ri_indices);
                    for (size_t i = 0; i < num_hashes; ++i) {
                        alphas[ri_indices[i]] ^= it;
                        gammas[ri_indices[i]] ^= ih;
                    }
                }
                return {SUCCESS, recovered_items};
            }
        }

        // Undo peeling changes so recover is non-destructive
        for (auto ri : recovered_items) {
            ItemType it = (ItemType)ri;
            HashType ih = get_item_hash(it);
            size_t ri_indices[num_hashes];
            get_bucket_indices(it, ri_indices);
            for (size_t i = 0; i < num_hashes; ++i) {
                alphas[ri_indices[i]] ^= it;
                gammas[ri_indices[i]] ^= ih;
            }
        }

        if (working_det_alpha == det_alpha && working_det_gamma == det_gamma) {
            return {SUCCESS, recovered_items};
        }

        // If a cleanup sketch exists, try to finish with it.
        if (cleanup_sketch != nullptr) {
            // Temporarily remove recovered items from the structure
            for (auto idx : recovered_items) {
                this->update(idx);
            }

            for (size_t i = 0; i < cleanup_sketch->get_num_samples(); i++) {
                ExhaustiveSketchSample sample = cleanup_sketch->exhaustive_sample();
                if (sample.result == ZERO) {
                    // Undo temporary removals
                    for (auto idx : recovered_items) {
                        this->update(idx);
                    }
                    return {SUCCESS, recovered_items};
                }
                for (auto idx : sample.idxs) {
                    recovered_items.push_back(idx);
                    this->update(idx);
                }
            }

            // Undo the temporary removals from cleanup probing
            for (auto idx : recovered_items) {
                this->update(idx);
            }
        }

        if (allow_partial && !recovered_items.empty()) {
            return {PARTIAL_RECOVERY, recovered_items};
        }
        return {FAILURE, recovered_items};
    }

public:
    Sketch *cleanup_sketch;

    IBLT() : capacity(0), num_hashes(0), universe_size(0), max_recovery_size(0),
             cleanup_sketch_support(0), has_cleanup_sketch(false), owns_cleanup_sketch(false),
             seed(0), det_alpha(0), det_gamma(0), cleanup_sketch(nullptr) {}

    IBLT(size_t universe_size, size_t max_recovery_size, double cleanup_sketch_support_factor,
         uint64_t seed, bool include_cleanup_sketch = true, Sketch *borrowed_cleanup_sketch = nullptr,
         size_t k = 3)
        : capacity((size_t)std::ceil(capacity_factor * max_recovery_size)),
          num_hashes(k), universe_size(universe_size), max_recovery_size(max_recovery_size),
          seed(seed),
          alphas((size_t)std::ceil(capacity_factor * max_recovery_size), 0),
          gammas((size_t)std::ceil(capacity_factor * max_recovery_size), 0),
          det_alpha(0), det_gamma(0) {

                assert(capacity > 0);
                assert(num_hashes > 0);
                assert(capacity > num_hashes);

        cleanup_sketch_support = (size_t)std::ceil(cleanup_sketch_support_factor * std::log2((double)universe_size));
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
    }

    IBLT(const IBLT &other)
        : capacity(other.capacity), num_hashes(other.num_hashes),
          universe_size(other.universe_size), max_recovery_size(other.max_recovery_size),
          cleanup_sketch_support(other.cleanup_sketch_support),
          has_cleanup_sketch(other.has_cleanup_sketch), owns_cleanup_sketch(false),
          seed(other.seed),
          alphas(other.alphas), gammas(other.gammas),
          det_alpha(other.det_alpha), det_gamma(other.det_gamma),
          cleanup_sketch(nullptr) {
        if (other.cleanup_sketch != nullptr) {
            if (other.owns_cleanup_sketch) {
                cleanup_sketch = new Sketch(*other.cleanup_sketch);
                owns_cleanup_sketch = true;
            } else {
                cleanup_sketch = other.cleanup_sketch;
            }
        }
    }

    IBLT &operator=(const IBLT &other) {
        if (this == &other) return *this;
        if (owns_cleanup_sketch) delete cleanup_sketch;
        cleanup_sketch = nullptr;

        capacity = other.capacity;
        num_hashes = other.num_hashes;
        universe_size = other.universe_size;
        max_recovery_size = other.max_recovery_size;
        cleanup_sketch_support = other.cleanup_sketch_support;
        has_cleanup_sketch = other.has_cleanup_sketch;
        owns_cleanup_sketch = false;
        seed = other.seed;
        alphas = other.alphas;
        gammas = other.gammas;
        det_alpha = other.det_alpha;
        det_gamma = other.det_gamma;

        if (other.cleanup_sketch != nullptr) {
            if (other.owns_cleanup_sketch) {
                cleanup_sketch = new Sketch(*other.cleanup_sketch);
                owns_cleanup_sketch = true;
            } else {
                cleanup_sketch = other.cleanup_sketch;
            }
        }
        return *this;
    }

    IBLT(IBLT &&other) noexcept
        : capacity(other.capacity), num_hashes(other.num_hashes),
          universe_size(other.universe_size), max_recovery_size(other.max_recovery_size),
          cleanup_sketch_support(other.cleanup_sketch_support),
          has_cleanup_sketch(other.has_cleanup_sketch), owns_cleanup_sketch(other.owns_cleanup_sketch),
          seed(other.seed),
          alphas(std::move(other.alphas)), gammas(std::move(other.gammas)),
          det_alpha(other.det_alpha), det_gamma(other.det_gamma),
          cleanup_sketch(other.cleanup_sketch) {
        other.cleanup_sketch = nullptr;
        other.owns_cleanup_sketch = false;
        other.has_cleanup_sketch = false;
    }

    IBLT &operator=(IBLT &&other) noexcept {
        if (this == &other) return *this;
        if (owns_cleanup_sketch) delete cleanup_sketch;

        capacity = other.capacity;
        num_hashes = other.num_hashes;
        universe_size = other.universe_size;
        max_recovery_size = other.max_recovery_size;
        cleanup_sketch_support = other.cleanup_sketch_support;
        has_cleanup_sketch = other.has_cleanup_sketch;
        owns_cleanup_sketch = other.owns_cleanup_sketch;
        seed = other.seed;
        alphas = std::move(other.alphas);
        gammas = std::move(other.gammas);
        det_alpha = other.det_alpha;
        det_gamma = other.det_gamma;
        cleanup_sketch = other.cleanup_sketch;

        other.cleanup_sketch = nullptr;
        other.owns_cleanup_sketch = false;
        other.has_cleanup_sketch = false;
        return *this;
    }

    ~IBLT() {
        if (owns_cleanup_sketch) {
            delete cleanup_sketch;
        }
    }

    void update(const vec_t update_idx) {
        ItemType item = (ItemType)update_idx;
        HashType item_hash = get_item_hash(item);

        det_alpha ^= item;
        det_gamma ^= item_hash;

        // TODO - do we want to keep using
        // variable length arrays?
        size_t indices[num_hashes];
        get_bucket_indices(item, indices);
        for (size_t i = 0; i < num_hashes; ++i) {
            alphas[indices[i]] ^= item;
            gammas[indices[i]] ^= item_hash;
        }
        if (cleanup_sketch != nullptr) {
            cleanup_sketch->update(update_idx);
        }
    }

    void reset() {
        det_alpha = 0;
        det_gamma = 0;
        std::fill(alphas.begin(), alphas.end(), 0);
        std::fill(gammas.begin(), gammas.end(), 0);
        if (cleanup_sketch != nullptr) {
            cleanup_sketch->zero_contents();
        }
    }

    // Non-destructive recovery
    RecoveryResult recover() {
        return recover_internal(false);
    }

    RecoveryResult recover(bool allow_partial) {
        return recover_internal(allow_partial);
    }

    void merge(const IBLT &other) {
        assert(other.capacity == capacity);
        det_alpha ^= other.det_alpha;
        det_gamma ^= other.det_gamma;
        for (size_t i = 0; i < capacity; ++i) {
            alphas[i] ^= other.alphas[i];
            gammas[i] ^= other.gammas[i];
        }
        if (cleanup_sketch != nullptr && other.cleanup_sketch != nullptr) {
            cleanup_sketch->merge(*other.cleanup_sketch);
        }
    }

    size_t space_usage_bytes() const {
        size_t total = sizeof(IBLT)
            + alphas.capacity() * sizeof(ItemType)
            + gammas.capacity() * sizeof(HashType);
        if (owns_cleanup_sketch && cleanup_sketch != nullptr) {
            total += sizeof(Sketch);
            total += cleanup_sketch->bucket_array_bytes();
        }
        return total;
    }

    inline long get_seed() const { return seed; }
};
