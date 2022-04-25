#include "../../include/l0_sampling/update.h"

std::ostream& operator<< (std::ostream &out, const Update &update){
  const uint64_t hi = update.index >> 64;
  const uint64_t lo = update.index;
  out << "Index: " << std::hex << hi << lo << " Value: " << std::dec << update.delta;
  return out;
}

bool operator== (const Update &upd1, const Update &upd2) {
  return upd1.index == upd2.index && upd1.delta == upd2.delta;
}
