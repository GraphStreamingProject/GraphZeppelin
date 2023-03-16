#pragma once
#include "../l0_sampling/sketch.h"

/*
 * Static functions for creating sketches without a provided memory location
 * used in unit testing.
 */
using SketchUniquePtr = std::unique_ptr<Sketch,std::function<void(Sketch*)>>;
SketchUniquePtr makeSketch(long seed) {
  void* loc = malloc(Sketch::sketchSizeof());
  return {
    Sketch::makeSketch(loc, seed),
    [](Sketch* s){ s->~Sketch(); free(s); }
  };
}

SketchUniquePtr makeSketch(long seed, std::fstream &binary_in, bool sparse=false) {
  void* loc = malloc(Sketch::sketchSizeof());
  return {
    Sketch::makeSketch(loc, seed, binary_in, sparse),
    [](Sketch* s){ free(s); }
  };
}
