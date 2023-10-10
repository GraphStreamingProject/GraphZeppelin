
/**
 * SketchDriver class:
 * Driver for sketching algorithms on a single machine
 * templatized by the algorithm.
 */
template <class Alg>
class SketchDriver {
 private:
  GutteringSystem *gts;
  Alg *sketching_alg;

 public:
  SketchDriver();
  update(GraphUpdate e, int thr_id = 0);
  prep_query();
}
