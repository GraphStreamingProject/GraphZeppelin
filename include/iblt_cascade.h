#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "iblt.h"
#include "recovery_types.h"
#include "sketch.h"
#include "types.h"

template<typename ItemType = vec_t, typename HashType = vec_hash_t>
class IBLTCascade {
private:
    static constexpr double primary_capacity_factor = 1.25;

    size_t universe_size;
    size_t max_recovery_size;
    size_t cleanup_sketch_support;
    bool has_cleanup_sketch;
    bool owns_cleanup_sketch;
    uint64_t seed;

    double secondary_ratio;
    double log_floor_factor;
    size_t num_hashes;

    IBLT<ItemType, HashType> primary;
    IBLT<ItemType, HashType> secondary;

    size_t min_recovery_size_for_k(size_t k) const {
        // Need ceil(1.3 * r) > k  => r > k / 1.3
        return std::max<size_t>(1, (size_t)std::floor(((double)k / primary_capacity_factor) + 1.0));
    }

    size_t secondary_recovery_size() const {
        // Target secondary capacity: secondary_ratio * R, but apply a log2 floor.
        const double target_cap = secondary_ratio * (double)max_recovery_size;
        const double log_floor_cap = log_floor_factor * std::log2(std::max<size_t>(2, universe_size));
        const double chosen_cap = std::max(target_cap, log_floor_cap);

        // IBLT capacity is ceil(1.3 * recovery_size), so invert.
        size_t rec_size = (size_t)std::ceil(chosen_cap / primary_capacity_factor);
        rec_size = std::max(rec_size, min_recovery_size_for_k(num_hashes));
        return rec_size;
    }

    uint64_t secondary_seed() const {
        //randomly generated 64 bit number
        return (seed * seed) + 0x91f14a89adb336a0ULL;
    }

    RecoveryResult recover_internal(bool allow_partial) {
        if (cleanup_sketch != nullptr) {
            cleanup_sketch->reset_sample_state();
        }

        std::vector<vec_t> recovered;

        // Alternate peeling between tables until no progress.
        while (true) {
            size_t before = recovered.size();

            RecoveryResult p = primary.recover(true);
            for (vec_t idx : p.recovered_indices) {
                recovered.push_back(idx);
                primary.update(idx);
                secondary.update(idx);
                if (cleanup_sketch != nullptr) {
                    cleanup_sketch->update(idx);
                }
            }

            RecoveryResult s = secondary.recover(true);
            for (vec_t idx : s.recovered_indices) {
                recovered.push_back(idx);
                primary.update(idx);
                secondary.update(idx);
                if (cleanup_sketch != nullptr) {
                    cleanup_sketch->update(idx);
                }
            }

            if (recovered.size() == before) {
                break;
            }
        }

        // Success check: if both tables are fully peeled after temporary removals.
        bool fully_recovered = (primary.recover(false).result == SUCCESS) &&
                               (secondary.recover(false).result == SUCCESS);
        if (fully_recovered) {
            for (vec_t idx : recovered) {
                primary.update(idx);
                secondary.update(idx);
                if (cleanup_sketch != nullptr) {
                    cleanup_sketch->update(idx);
                }
            }
            return {SUCCESS, recovered};
        }

        // Optional cleanup sketch fallback.
        if (cleanup_sketch != nullptr) {
            for (size_t i = 0; i < cleanup_sketch->get_num_samples(); i++) {
                ExhaustiveSketchSample sample = cleanup_sketch->exhaustive_sample();
                if (sample.result == ZERO) {
                    for (vec_t idx : recovered) {
                        primary.update(idx);
                        secondary.update(idx);
                        cleanup_sketch->update(idx);
                    }
                    return {SUCCESS, recovered};
                }
                for (vec_t idx : sample.idxs) {
                    recovered.push_back(idx);
                    primary.update(idx);
                    secondary.update(idx);
                    cleanup_sketch->update(idx);
                }
            }
        }

        for (vec_t idx : recovered) {
            primary.update(idx);
            secondary.update(idx);
            if (cleanup_sketch != nullptr) {
                cleanup_sketch->update(idx);
            }
        }

        if (allow_partial && !recovered.empty()) {
            return {PARTIAL_RECOVERY, recovered};
        }
        return {FAILURE, recovered};
    }

public:
    Sketch *cleanup_sketch;

    IBLTCascade()
        : universe_size(0), max_recovery_size(0), cleanup_sketch_support(0),
          has_cleanup_sketch(false), owns_cleanup_sketch(false), seed(0),
          secondary_ratio(0.05), log_floor_factor(1.0), num_hashes(3),
          primary(), secondary(), cleanup_sketch(nullptr) {}

    IBLTCascade(size_t universe_size,
                size_t max_recovery_size,
                double cleanup_sketch_support_factor,
                uint64_t seed,
                bool include_cleanup_sketch = true,
                Sketch *borrowed_cleanup_sketch = nullptr,
                size_t k = 3,
                double secondary_ratio = 0.05,
                double log_floor_factor = 1.0)
        : universe_size(universe_size),
          max_recovery_size(max_recovery_size),
          cleanup_sketch_support(0),
          has_cleanup_sketch(false),
          owns_cleanup_sketch(false),
          seed(seed),
          secondary_ratio(secondary_ratio),
          log_floor_factor(log_floor_factor),
          num_hashes(k),
          primary(universe_size, max_recovery_size, 0.0, seed, false, nullptr, k),
          secondary(),
          cleanup_sketch(nullptr) {
        assert(max_recovery_size > 0);
        assert(num_hashes > 0);

        size_t secondary_rec_size = secondary_recovery_size();
        secondary = IBLT<ItemType, HashType>(
            universe_size, secondary_rec_size, 0.0, secondary_seed(), false, nullptr, num_hashes);

        cleanup_sketch_support = (size_t)std::ceil(
            cleanup_sketch_support_factor * std::log2((double)std::max<size_t>(2, universe_size)));

        if (borrowed_cleanup_sketch != nullptr) {
            cleanup_sketch = borrowed_cleanup_sketch;
            has_cleanup_sketch = true;
        } else if (include_cleanup_sketch && cleanup_sketch_support > 0) {
            cleanup_sketch = new Sketch(universe_size, seed, cleanup_sketch_support, 1);
            has_cleanup_sketch = true;
            owns_cleanup_sketch = true;
        }
    }

    IBLTCascade(const IBLTCascade &other)
        : universe_size(other.universe_size),
          max_recovery_size(other.max_recovery_size),
          cleanup_sketch_support(other.cleanup_sketch_support),
          has_cleanup_sketch(other.has_cleanup_sketch),
          owns_cleanup_sketch(false),
          seed(other.seed),
          secondary_ratio(other.secondary_ratio),
          log_floor_factor(other.log_floor_factor),
          num_hashes(other.num_hashes),
          primary(other.primary),
          secondary(other.secondary),
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

    IBLTCascade &operator=(const IBLTCascade &other) {
        if (this == &other) return *this;
        if (owns_cleanup_sketch) delete cleanup_sketch;
        cleanup_sketch = nullptr;

        universe_size = other.universe_size;
        max_recovery_size = other.max_recovery_size;
        cleanup_sketch_support = other.cleanup_sketch_support;
        has_cleanup_sketch = other.has_cleanup_sketch;
        owns_cleanup_sketch = false;
        seed = other.seed;
        secondary_ratio = other.secondary_ratio;
        log_floor_factor = other.log_floor_factor;
        num_hashes = other.num_hashes;
        primary = other.primary;
        secondary = other.secondary;

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

    IBLTCascade(IBLTCascade &&other) noexcept
        : universe_size(other.universe_size),
          max_recovery_size(other.max_recovery_size),
          cleanup_sketch_support(other.cleanup_sketch_support),
          has_cleanup_sketch(other.has_cleanup_sketch),
          owns_cleanup_sketch(other.owns_cleanup_sketch),
          seed(other.seed),
          secondary_ratio(other.secondary_ratio),
          log_floor_factor(other.log_floor_factor),
          num_hashes(other.num_hashes),
          primary(std::move(other.primary)),
          secondary(std::move(other.secondary)),
          cleanup_sketch(other.cleanup_sketch) {
        other.cleanup_sketch = nullptr;
        other.owns_cleanup_sketch = false;
        other.has_cleanup_sketch = false;
    }

    IBLTCascade &operator=(IBLTCascade &&other) noexcept {
        if (this == &other) return *this;
        if (owns_cleanup_sketch) delete cleanup_sketch;

        universe_size = other.universe_size;
        max_recovery_size = other.max_recovery_size;
        cleanup_sketch_support = other.cleanup_sketch_support;
        has_cleanup_sketch = other.has_cleanup_sketch;
        owns_cleanup_sketch = other.owns_cleanup_sketch;
        seed = other.seed;
        secondary_ratio = other.secondary_ratio;
        log_floor_factor = other.log_floor_factor;
        num_hashes = other.num_hashes;
        primary = std::move(other.primary);
        secondary = std::move(other.secondary);
        cleanup_sketch = other.cleanup_sketch;

        other.cleanup_sketch = nullptr;
        other.owns_cleanup_sketch = false;
        other.has_cleanup_sketch = false;
        return *this;
    }

    ~IBLTCascade() {
        if (owns_cleanup_sketch) {
            delete cleanup_sketch;
        }
    }

    void update(const vec_t update_idx) {
        primary.update(update_idx);
        secondary.update(update_idx);
        if (cleanup_sketch != nullptr) {
            cleanup_sketch->update(update_idx);
        }
    }

    void reset() {
        primary.reset();
        secondary.reset();
        if (cleanup_sketch != nullptr) {
            cleanup_sketch->zero_contents();
        }
    }

    RecoveryResult recover() {
        return recover_internal(false);
    }

    RecoveryResult recover(bool allow_partial) {
        return recover_internal(allow_partial);
    }

    void merge(const IBLTCascade &other) {
        primary.merge(other.primary);
        secondary.merge(other.secondary);
        if (cleanup_sketch != nullptr && other.cleanup_sketch != nullptr) {
            cleanup_sketch->merge(*other.cleanup_sketch);
        }
    }

    size_t space_usage_bytes() const {
        size_t total = sizeof(IBLTCascade) + primary.space_usage_bytes() + secondary.space_usage_bytes();
        if (owns_cleanup_sketch && cleanup_sketch != nullptr) {
            total += sizeof(Sketch);
            total += cleanup_sketch->bucket_array_bytes();
        }
        return total;
    }

    inline uint64_t get_seed() const { return seed; }
    inline double get_secondary_ratio() const { return secondary_ratio; }
};
