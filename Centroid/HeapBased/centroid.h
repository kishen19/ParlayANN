// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>

#include "../utils/NSGDist.h"
#include "../utils/beamSearch.h"
#include "../utils/check_nn_recall.h"
#include "../utils/parse_results.h"
#include "../utils/stats.h"
#include "../utils/types.h"
#include "../utils/graph.h"
#include "../utils/aspen_graph.h"
#include "../utils/aspen_flat_graph.h"
#include "../../algorithms/vamana/index.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"


template<typename Point, typename PointRange, typename indexType, typename GraphType>
void Centroid(GraphType &Graph, long k, BuildParams &BP,
         char *res_file,
         bool graph_built, PointRange &Points) {
  std::cout << "Size of dataset: " << Points.size() << std::endl;
  using findex = knn_index<Point, PointRange, indexType, GraphType>;
  using distanceType = typename Point::distanceType;
  
  // Build Index
  parlay::internal::timer t("Centroid");
  size_t n = Points.size();
  findex I(BP);
  double idx_time;
  stats<unsigned int> BuildStats(Points.size());
  if(graph_built){
    idx_time = 0;
  } else{
    I.build_index(Graph, Points, BuildStats);
    idx_time = t.next_time();
  }
  indexType start_point = I.get_start();

  // Init Priority Queue
  using kv = std::tuple<distanceType,indexType,indexType>;
  auto G = Graph.Get_Graph_Read_Only();
  QueryParams QP((long) 2, BP.L,  (double) 0.0, (long) Points.size(), (long) G.max_degree());
  auto h = parlay::sequence<kv>::from_function(n,[&](size_t i){
    auto out = beam_search(Points[i], G, Points, start_point, QP);
    if (i<10){
      std::cout << i << " " << out.first.first.size() << std::endl;
    }
    if (out.first.first[0].first!=i){
      return std::make_tuple(out.first.first[0].second,i,out.first.first[0].first);
    } else {
      return std::make_tuple(out.first.first[1].second,i,out.first.first[1].first);
    }
  });
  Graph.Release_Graph(std::move(G));
  for (int i=0; i<10; i++){
    std::cout << std::get<0>(h[i]) << " " << std::get<1>(h[i]) << " " << std::get<2>(h[i]) << std::endl;
  }
  std::priority_queue<kv, vector<kv>, std::greater<kv>> H(h.begin(), h.end());
  std::cout << "Heap Init Done" << std::endl;

  // Centroid Process
  std::cout << "Starting Centroid Process" << std::endl;
  kv cand = H.top();
  H.pop();
  auto u = std::get<1>(cand);
  auto v = std::get<2>(cand);
  std::cout << "u: " << u << ", v: " << v << ", dist: " << std::get<0>(cand) << std::endl;
  auto p1 = Points[u];
  auto p2 = Points[v];
  auto c = p1.centroid(p2,n);
  std::cout << "Computed Centroid" << std::endl;
  Points.add_point(c);
  parlay::sequence<indexType> deletes = {u,v};
  std::cout << "Starting Deletion" << std::endl;
  I.lazy_delete(deletes);
  I.start_delete_epoch();
  I.consolidate(Graph, Points);
  I.end_delete_epoch(Graph);
  std::cout << "Finished deleting" << std::endl;
  parlay::sequence<indexType> inserts = {(indexType) n};
  I.insert(Graph, Points, BuildStats, inserts);
  std::cout << "Inserted Centroid" << std::endl;
  n++;

  // size_t DEL_THRESH = n/20;
  // size_t cur_dels = 0;
  // for(int i=0; i<n-1; i++){
  //   parlay::sequence<indexType> indices = parlay::tabulate(s, [&] (size_t j){return static_cast<indexType>(i*s+j);});
  //   I.lazy_delete(indices);
  //   I.start_delete_epoch();
  //   I.consolidate(Graph, Points);
  //   I.end_delete_epoch(Graph);
  //   std::cout << "Finished deleting" << std::endl;
  //   I.insert(Graph, Points, BuildStats, indices);
  //   std::cout << "Finished re-inserting" << std::endl;
  // }

  // indexType start_point = I.get_start();
  // std::string name = "Vamana";
  // std::string params =
  //     "R = " + std::to_string(BP.R) + ", L = " + std::to_string(BP.L);

  // auto G = Graph.Get_Graph_Read_Only();
  // auto [avg_deg, max_deg] = graph_stats_(G);
  // size_t G_size = G.size();
  // Graph.Release_Graph(std::move(G));
  // auto vv = BuildStats.visited_stats();
  // std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1]
  //           << std::endl;
  // Graph_ G_(name, params, G_size, avg_deg, max_deg, 0.0);
  // G_.print();
  // if(Query_Points.size() != 0) search_and_parse<Point, PointRange, indexType>(G_, Graph, Points, Query_Points, GT, res_file, k, false, start_point);

}