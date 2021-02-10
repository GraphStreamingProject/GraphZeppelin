#pragma once
#include <vector>
#include "../../include/update.h"
/* This class takes as input N and P and
generates a random stream of P updates to a vector of length N. It then processes
those updates to the vector and makes both the updates and the processed vector
visible.
*/

using namespace std;

class Testing_Vector{
  Update* stream;
  vector<long>* vect;
  const unsigned long vector_length,num_updates;

public:
  //n is size of vector and m is number of updates
  Testing_Vector(unsigned long vector_length, unsigned long num_updates): vector_length(vector_length), num_updates(num_updates) {
    stream = new Update[num_updates];
    vect = new vector<long>(vector_length,0);
    //Initialize the stream, and finalize the input vector.
    for (unsigned int i = 0; i < num_updates; i++){
      stream[i] = {static_cast<vec_t>(rand() % vector_length),rand()%10 - 5};
      (*vect)[stream[i].index] += stream[i].delta;
      //cout << "Index: " << stream[i].index << " Delta: " << stream[i].delta << endl;
    }
  }

  ~Testing_Vector(){
    delete [] stream;
    delete vect;
  }

  //get the ith update (starting from index 0)
  Update get_update(unsigned long i){
    return stream[i];
  }

  //get the ith entry of the processed Testing_Vector
  long get_entry(unsigned long i){
    return (*vect)[i];
  }

  friend std::ostream& operator<< (std::ostream &os, const Testing_Vector &vec);
};

ostream& operator<<(ostream& os, const Testing_Vector& vec) {
  for (unsigned long i = 0; i < vec.num_updates; i++) {
    os << vec.stream[i] << std::endl;
  }
  vector<long> vect = *(vec.vect);
  for (auto val : vect) {
    os << val  << ',';
  }
  os << std::endl;
  return os;
}
