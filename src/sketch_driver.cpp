#include "system_driver.h"

template <class Alg>
SketchDriver::SketchDriver(Alg* alg, GraphConfiguration* config) : sketching_alg(alg) {
  // set the leaf size of the guttering system appropriately
  if (config._gutter_conf.get_gutter_bytes() == GutteringConfiguration::uninit_param) {
    config._gutter_conf.gutter_bytes(Supernode::get_size() * config._batch_factor);
  }

  backup_file = config._disk_dir + "supernode_backup.data";
  // Create the guttering system
  if (config._gutter_sys == GUTTERTREE)
    gts = new GutterTree(config._disk_dir, num_nodes, config._num_graph_workers,
                         config._gutter_conf, true);
  else if (config._gutter_sys == STANDALONE)
    gts = new StandAloneGutters(num_nodes, config._num_graph_workers, num_inserters,
                                config._gutter_conf);
  else
    gts = new CacheGuttering(num_nodes, config._num_graph_workers, num_inserters,
                             config._gutter_conf);

  GraphWorker::set_config(config._num_graph_workers);
  GraphWorker::start_workers(this, gts, Supernode::get_size());
  std::cout << config << std::endl;  // print the graph configuration
}

template <class Alg>
SketchDriver::update(GraphUpdate upd, int thr_id) {
  alg->pre_insert(upd, thr_id);
  Edge &edge = upd.edge;
  gts->insert({edge.src, edge.dst}, thr_id);
  gts->insert({edge.dst, edge.src}, thr_id);
}

template <class Alg>
SketchDriver::prep_query() {
  if (alg->is_query_cached())
    return;

  gts->force_flush();            // flush everything in guttering system to make final updates
  GraphWorker::flush_workers();  // wait for the workers to finish applying the updates
}
