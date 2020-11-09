#include <iostream>
#include "../include/update.h"

std::ostream& operator<< (std::ostream &out, const Update &update){
    out << "Index: " << update.index << " Value: " << update.delta;
    return out;
}

bool operator== (const Update &upd1, const Update &upd2) {
  return upd1.index == upd2.index && upd1.delta == upd2.delta;
}
