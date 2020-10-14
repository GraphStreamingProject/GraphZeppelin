#pragma once
#include <iostream>

struct Update{
    long index,delta;
    friend std::ostream& operator<< (std::ostream &out, const Update &update);
};

std::ostream& operator<< (std::ostream &out, const Update &update){
    out << "Index: " << update.index << " Value: " << update.delta;
    return out;
}
