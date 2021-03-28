%module sketch
%{
#include <stdint.h>
#include "../include/types.h"
#include "../include/sketch.h"
#include "../include/update.h"
%}

%include <std_string.i>
%include <stdint.i>

%ignore log2;
%include "../include/types.h"

%rename(plus) operator+;
%rename(plus_equals) operator+=;
%rename(equals) operator==;
%rename(times) operator*;
%rename(to_stream) operator<<;

class Sketch {
  const long seed;
  const vec_t n;
  std::vector<Bucket_Boruvka> buckets;
  const ubucket_t large_prime;
  bool already_quered = false;

  FRIEND_TEST(SketchTestSuite, TestExceptions);
  FRIEND_TEST(SketchTestSuite, TestBatchUpdate);

  //Initialize a sketch of a vector of size n
 public:
  Sketch(vec_t n, long seed);

  /**
   * Update a sketch based on information about one of its indices.
   * @param update the point update.
   */
  void update(Update update);

  /**
   * Update a sketch given a batch of updates
   * @param begin a ForwardIterator to the first update
   * @param end a ForwardIterator to after the last update
   */
  void batch_update(const std::vector<Update> &updates);

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
    friend std::ostream& operator<< (std::ostream &os, const Sketch &sketch);
};

%extend Sketch {
  std::string __str__() const {
    std::ostringstream out;
    out << *$self;
    return out.str();
  }
}

%extend Sketch {
%pythoncode {
    def __reduce__(self):
        sketch.Sketch.__init__(size, seed)
        args = self.size, self.seed
        return self.__class__, args
}
}

struct Update {
  // the position in the vector that is changed
  vec_t index;
  // the magnitude of the change
  long delta;
  friend std::ostream& operator<< (std::ostream &out, const Update &update);
  friend bool operator== (const Update &upd1, const Update &upd2);
};

%extend Update {
  std::string __str__() const {
    std::ostringstream out;
    out << *$self;
    return out.str();
  }
}
