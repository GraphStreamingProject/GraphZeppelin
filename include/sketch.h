#pragma once
#include "dense_sketch.h"
#include "sparse_sketch.h"

#ifdef L0_FULLY_DENSE
typedef DenseSketch Sketch;
#else
typedef SparseSketch Sketch;
#endif
