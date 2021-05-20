%module sketch
%begin %{
#define SWIG_PYTHON_STRICT_BYTE_CHAR
%}
%{
#include <stdint.h>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <sstream>
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

}

%define %boost_picklable(cls...)
    %extend cls {
        std::string __getstate__()
        {
            std::stringstream ss;
            boost::archive::binary_oarchive ar(ss);
            ar << *($self);
            return ss.str();
        }

        void __setstate_internal(std::string const& sState)
        {
            std::stringstream ss(sState);
            boost::archive::binary_iarchive ar(ss);
            ar >> *($self);
        }

        std::string __tostr__() const {
            std::ostringstream out;
	    out << *$self;
	    return out.str();
	}

        %pythoncode %{
	    def __str__(self):
                return self.__tostr__().decode("utf-8")

	    def __setstate__(self, sState):
                self.__init__(0, 0)
                self.__setstate_internal(sState)
        %}
    }
%enddef

%boost_picklable(Sketch)

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
