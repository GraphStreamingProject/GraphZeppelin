#include "sketch.h"
#include "recovery.h"
#include "bucket.h"
#include <chrono>
#include <gtest/gtest.h>
#include <random>
#include "testing_vector.h"

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

static const int num_columns = 1;
template <typename RecoveryImpl>
class RecoveryTestSuite : public ::testing::Test {};

using RecoveryImplementations =
    ::testing::Types<SparseRecoveryCFRChain, SparseRecoveryIBLT, SparseRecoveryIBLTCascade>;

TYPED_TEST_SUITE(RecoveryTestSuite, RecoveryImplementations);

TYPED_TEST(RecoveryTestSuite, RecoveryZeroOrOne) {
    TypeParam recovery(1 << 20, 1 << 10, 1, get_seed());
    auto result = recovery.recover();
    ASSERT_EQ(result.recovered_indices.size(), 0);
    ASSERT_EQ(result.result, SUCCESS);
    recovery.update(5);
    ASSERT_EQ(recovery.recover().recovered_indices.size(), 1);  
    ASSERT_EQ(recovery.recover().recovered_indices[0], 5);
    recovery.update(5);
    ASSERT_EQ(result.recovered_indices.size(), 0);
    ASSERT_EQ(result.result, SUCCESS);
}

TYPED_TEST(RecoveryTestSuite, RecoveryExtremelySmall) {
    TypeParam recovery(1 << 13, 16, 1, get_seed());
    auto result = recovery.recover();
    ASSERT_EQ(result.recovered_indices.size(), 0);
    ASSERT_EQ(result.result, SUCCESS);
    recovery.update(5);
    ASSERT_EQ(recovery.recover().recovered_indices.size(), 1);  
    ASSERT_EQ(recovery.recover().recovered_indices[0], 5);
    std::unordered_set<vec_t> inserted;
    for (vec_t i = 0; i < 8; i++) {
        recovery.update(i);
        inserted.insert(i);
    }
    inserted.erase(5); // 5 was already inserted
    auto result2 = recovery.recover();
    ASSERT_EQ(result2.result, SUCCESS);
    ASSERT_EQ(result2.recovered_indices.size(), 7);
    std::unordered_set<vec_t> recovered2(result2.recovered_indices.begin(), result2.recovered_indices.end());
    ASSERT_EQ(recovered2, inserted);
}

TYPED_TEST(RecoveryTestSuite, RecoveryMediumSize) {
    TypeParam recovery(1 << 20, 1 << 10, 1, get_seed());
    auto result = recovery.recover();
    ASSERT_EQ(result.recovered_indices.size(), 0);
    ASSERT_EQ(result.result, SUCCESS);
    recovery.update(5);
    ASSERT_EQ(recovery.recover().recovered_indices.size(), 1);  
    ASSERT_EQ(recovery.recover().recovered_indices[0], 5);
    std::unordered_set<vec_t> inserted;
    recovery.update(5);
    for (vec_t i = 0; i < 1 << 10; i++) {
        recovery.update(i);
        inserted.insert(i);
    }
    auto result2 = recovery.recover();
    std::unordered_set<vec_t> recovered2(result2.recovered_indices.begin(), result2.recovered_indices.end());
    ASSERT_EQ(recovered2, inserted);
    auto result3 = recovery.recover();
    std::unordered_set<vec_t> recovered3(result3.recovered_indices.begin(), result3.recovered_indices.end());
    ASSERT_EQ(recovered3, inserted);
    
    // REPEAT TO MAKE SURE NON-DESTRUCTIVE
}

TYPED_TEST(RecoveryTestSuite, RecoveryFailureCondition) {
    TypeParam recovery(1 << 20, 1 << 10, 1, get_seed());
    std::unordered_set<vec_t> inserted;
    for (vec_t i = 0; i < 1 << 14; i++) {
        recovery.update(i);
        inserted.insert(i);
    }
    auto result = recovery.recover();
    ASSERT_EQ(result.result, FAILURE);
    std::cout << "size: " << result.recovered_indices.size() << std::endl;
    // make sure all returned things were in there:
    for (auto idx: result.recovered_indices) {
      ASSERT_TRUE(inserted.find(idx) != inserted.end());
    }
    // inserted.clear();
    // remove all but the final few elements
    // TODO - figure out the right place to put sketch clearing
    recovery.cleanup_sketch->reset_sample_state();
    for (vec_t i = 0; i < (1 << 14) - 1027; i++) {
        recovery.update(i);
        inserted.erase(i);
    }
    // TODO - WRITE A HELPER FUNCTION FOR TIHS STYLE OF TEST CASE
    auto result3 = recovery.recover();
    std::unordered_set<vec_t> recovered3(result3.recovered_indices.begin(), result3.recovered_indices.end());
    ASSERT_EQ(result3.result, SUCCESS);
    ASSERT_EQ(recovered3, inserted);
}

TYPED_TEST(RecoveryTestSuite, RecoveryForceSketchUse) {
  // TODO - IRON THIS OUT
    TypeParam recovery(1 << 20, 1 << 4, 1, get_seed());
    std::unordered_set<vec_t> inserted;
    for (vec_t i = 0; i < (1 << 4) * 2; i++) {
        recovery.update(i);
        inserted.insert(i);
    }
    auto result = recovery.recover();
    for (auto idx: result.recovered_indices) {
        ASSERT_TRUE(inserted.find(idx) != inserted.end());
    }
}

TYPED_TEST(RecoveryTestSuite, RecoveryMerge) {
  // TODO - IRON THIS OUT
    auto seed = get_seed();
    TypeParam recovery1(1 << 20, 1 << 10, 1, seed);
    TypeParam recovery2(1 << 20, 1 << 10, 1, seed);
    for (vec_t i = 0; i < (1 << 10) * 2; i++) {
        recovery1.update(i);
    }
    vec_t offset = 512;
    for (vec_t i = 0; i < (1 << 10) * 2; i++) {
        recovery1.update(i+512);
    }
    recovery1.merge(recovery2);
    auto result = recovery1.recover();
    ASSERT_EQ(result.result, SUCCESS);
    ASSERT_EQ(result.recovered_indices.size(), 1 << 10);
    for (auto idx: result.recovered_indices) {
        ASSERT_TRUE(idx < 512 || idx >= 1024);
    }
}

TYPED_TEST(RecoveryTestSuite, RecoveryManyFailureProbability) {
  // TODO - IRON THIS OUT
    auto vector_size = 1 << 20;
    auto recovery_size = 1 << 10;
    auto num_sketches = 1 << 15;
    double recovery_size_adjustment = 1;
    auto seed = get_seed();
        std::vector<TypeParam> recoveries;
    for (vec_t i = 0; i < num_sketches; i++) {
            recoveries.push_back(TypeParam(
          vector_size, ceill(recovery_size * recovery_size_adjustment), 1,
          seed));
    }
    for (size_t i = 0; i < num_sketches; i++) {
        for (vec_t j = recovery_size * i; j < recovery_size * (i+1); j++) {
            recoveries[i].update(j);
        }
    }
    size_t num_failures = 0;
    for (size_t i = 0; i < num_sketches; i++) {
        auto result = recoveries[i].recover();
        if (result.result == SUCCESS) {
            ASSERT_EQ(result.recovered_indices.size(), recovery_size);
            for (auto idx: result.recovered_indices) {
                ASSERT_TRUE(idx >= recovery_size * i && idx < recovery_size * (i+1));
            }
        } else {
            num_failures++;
            for (auto idx: result.recovered_indices) {
                ASSERT_TRUE(idx >= recovery_size * i && idx < recovery_size * (i+1));
            }
        }
    }
    // allow 0.1% failure rate for this test.
    ASSERT_LE(num_failures, num_sketches / 1024);
    
}

TYPED_TEST(RecoveryTestSuite, RecoveryWithoutCleanupSketch) {
    TypeParam recovery(1 << 20, 1 << 10, 1, get_seed(), false);
    auto empty_result = recovery.recover();
    ASSERT_EQ(empty_result.result, SUCCESS);
    ASSERT_EQ(empty_result.recovered_indices.size(), 0);

    recovery.update(5);
    auto result = recovery.recover();
    ASSERT_EQ(result.result, SUCCESS);
    ASSERT_EQ(result.recovered_indices.size(), 1);
    ASSERT_EQ(result.recovered_indices[0], 5);
}

TYPED_TEST(RecoveryTestSuite, PartialRecoveryAPIWithoutCleanupPhase) {
    TypeParam recovery(1 << 20, 2, 1, get_seed(), false);
    recovery.update(42);

    auto full_recover_result = recovery.recover();
    ASSERT_TRUE(full_recover_result.result == FAILURE || full_recover_result.result == SUCCESS);

    auto partial_result = recovery.recover(true);
    ASSERT_NE(partial_result.result, FAILURE);
    ASSERT_TRUE(partial_result.result == SUCCESS || partial_result.result == PARTIAL_RECOVERY);

    // Partial recovery is non-destructive, so repeated calls are consistent.
    auto partial_result_again = recovery.recover(true);
    ASSERT_NE(partial_result_again.result, FAILURE);
    ASSERT_TRUE(partial_result_again.result == SUCCESS || partial_result_again.result == PARTIAL_RECOVERY);
    ASSERT_EQ(partial_result_again.recovered_indices, partial_result.recovered_indices);
}