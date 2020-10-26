#pragma once
#include <iostream>

/**
 * Representation of a generic vector point update.
 */
struct Update{
    // the position in the vector that is changed
    long index;
    // the magnitude of the change
    long delta;
    friend std::ostream& operator<< (std::ostream &out, const Update &update);
};

std::ostream& operator<< (std::ostream &out, const Update &update){
    out << "Index: " << update.index << " Value: " << update.delta;
    return out;
}
