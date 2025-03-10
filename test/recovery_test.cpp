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
TEST(RecoveryTestSuite, RecoveryZeroOrOne) {
    SparseRecovery recovery(1 << 20, 1 << 10, 1, get_seed());
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

TEST(RecoveryTestSuite, RecoveryMediumSize) {
    SparseRecovery recovery(1 << 20, 1 << 10, 1, get_seed());
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

TEST(RecoveryTestSuite, RecoveryFailureCondition) {
    SparseRecovery recovery(1 << 20, 1 << 10, 1, get_seed());
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
    recovery.cleanup_sketch.reset_sample_state();
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