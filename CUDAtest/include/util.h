#pragma once
#include "./graph_zeppelin_common.h"

unsigned long long int double_to_ull(double d, double epsilon = 0.00000001);

/*
 * Inverts the concat pairing function.
 * @param idx
 * @return the pair, with left and right ordered lexicographically.
 */
std::pair<uint32_t , uint32_t> inv_concat_pairing_fn(uint64_t idx);