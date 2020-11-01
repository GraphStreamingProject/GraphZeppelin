#include <iostream>
#include "update.h"

std::ostream& operator<< (std::ostream &out, const Update &update){
    out << "Index: " << update.index << " Value: " << update.delta;
    return out;
}