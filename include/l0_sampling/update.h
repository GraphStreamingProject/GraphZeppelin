#pragma once
#include <iostream>
#include "../types.h"

/**
 * Representation of a generic vector point update.
 */
struct Update {
  // the position in the vector that is changed
  vec_t index;
  // the magnitude of the change
  long delta;

  friend std::ostream& operator<< (std::ostream &out, const Update &update);
  friend bool operator== (const Update &upd1, const Update &upd2);
};

