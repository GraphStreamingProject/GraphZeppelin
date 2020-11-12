#pragma once
#include <vector>
#include <exception>
#include "update.h"
#include "bucket.h"
using namespace std;

/**
 * An implementation of a "sketch" as defined in the L0 algorithm.
 * Note a sketch may only be queried once. Attempting to query multiple times will
 * raise an error.
 */
class Sketch {
  const long seed;
  const unsigned long long int n;
  std::vector<Bucket> buckets;
  const unsigned long long int random_prime;
  bool already_quered = false;

  //Initialize a sketch of a vector of size n
public:
  Sketch(unsigned long long int n, long seed);

  /**
   * Update a sketch based on information about one of its indices.
   * @param update the point update.
   */
  void update(Update update);

  /**
   * Function to query a sketch.
   * @return                        an index in the form of an Update.
   * @throws MultipleQueryException if the sketch has already been queried.
   * @throws NoGoodBucketException  if there are no good buckets to choose an
   *                                index from.
   */
  Update query();

  friend Sketch operator+ (const Sketch &sketch1, const Sketch &sketch2);
  friend Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2);
  friend Sketch operator* (const Sketch &sketch1, long scaling_factor );
};

class MultipleQueryException : public exception {
  virtual const char* what() const throw() {
    return "This sketch has already been sampled!";
  }
};

class NoGoodBucketException : public exception {
  virtual const char* what() const throw() {
    return "Found no good bucket!";
  }
};
