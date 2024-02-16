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

#pragma once

#include <algorithm>
#include <iostream>

#include "NSGDist.h"
#include "parlay/internal/file_map.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "types.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cpam/cpam.h>
#include <pam/pam.h>

#include <mutex>

#include "aspen/aspen.h"

template <typename indexType>
struct Aspen_Graph {
  using GraphT = aspen::symmetric_graph<indexType>;
  using vertex_tree = typename GraphT::vertex_tree;
  using vertex_entry = typename GraphT::vertex_entry;
  using vtx_entry = typename GraphT::vtx_entry;
  using edge_array = typename GraphT::edge_array;
  using version = typename aspen::versioned_graph<GraphT>::version;

  struct Aspen_Vertex {

    size_t size() { return edge_data.size(); }
    indexType id() { return id_; }

    Aspen_Vertex() {}
    Aspen_Vertex(indexType id, edge_array edges, long maxDeg, GraphT& G)
        : id_(id), edge_data(std::move(edges)), maxDeg(maxDeg), G(G) {}

    template <typename rangeType>
    void append_neighbors(const rangeType& r) {
      parlay::sequence<indexType> neighbors;
      auto edges = edge_data.get_edges();
      for (size_t i = 0; i < edges.size(); ++i) {
        neighbors.push_back(edges[i]);
      }
      for (indexType i : r)
        neighbors.push_back(i);
      return update_neighbors(neighbors);
    }

    template <typename rangeType>
    void update_neighbors(const rangeType& r) {
      if (r.size() > maxDeg) {
        std::cout << "Error in update: tried to exceed max degree" << std::endl;
        abort();
      }
      auto edges = edge_array(r);
      auto seq = {std::make_tuple(id_, std::move(edges))};
      G.insert_vertices_batch(seq.size(), seq.begin());
    }

    parlay::slice<indexType*, indexType*> neighbors() {
      return edge_data.get_edges();
    }

    // TODO is reordering vertices possible here? probably not right?

    void prefetch() {
      // TODO
    }

   private:
    indexType id_;
    edge_array edge_data;
    long maxDeg;
    GraphT& G;
  };

  struct Graph {

    long max_degree() const { return maxDeg; }
    size_t size() const { return V.graph.num_vertices(); }

    Graph() {}

    Graph(long maxDeg) : maxDeg(maxDeg) {}

    Graph(version V, long maxDeg, bool read_only = true)
        : V(V), maxDeg(maxDeg) {
      if (read_only)
        G = V.graph;
      else
        G = (V.graph).functional_copy();
    }

    void batch_update(
       parlay::sequence<std::pair<indexType, parlay::sequence<indexType>>>&
          edges) {
      // std::cout << "processing updates to " << edges.size() << " vertices" <<
      // std::endl;
      auto vals = parlay::tabulate(edges.size(), [&](size_t i) {
        indexType index = edges[i].first;
        size_t ngh_size = edges[i].second.size();
        if (ngh_size > maxDeg) {
          std::cout << "ERROR in batch_update: ngh too large" << std::endl;
          abort();
        }
        auto my_edges = edge_array(edges[i].second);
        return std::make_tuple(index, std::move(my_edges));
      });
      G.insert_vertices_batch(vals.size(), vals.begin());
    }

    void batch_delete(parlay::sequence<indexType>& deletes) {
      G.delete_vertices_batch(deletes.size(), deletes.begin());
    }

    GraphT move_graph() { return std::move(G); }

    version move_version() { return std::move(V); }

    Aspen_Vertex operator[](indexType i) {
      return Aspen_Vertex(i, G.get_vertex(i), maxDeg, G);
    }

   private:
    long maxDeg;
    version V;
    GraphT G;
  };

  Aspen_Graph() {}

  Aspen_Graph(long md, size_t n) : maxDeg(md) {
    GraphT GG;
    VG = aspen::versioned_graph<GraphT>(std::move(GG));
  }

  Aspen_Graph(char* gFile) {}

  Graph Get_Graph() {
    auto S = VG.acquire_version();
    std::cout << "Acquired writable version with timestamp " << S.timestamp
              << std::endl;
    return Graph(std::move(S), maxDeg, false);
  }

  // TODO add safeguard to avoid updating graph in read-only mode
  Graph Get_Graph_Read_Only() {
    auto S = VG.acquire_version();
    std::cout << "Acquired read-only version with timestamp " << S.timestamp
              << std::endl;
    return Graph(std::move(S), maxDeg, true);
  }

  // TODO do we need to do anything with the copy of graph that's stored in the
  // graph wrapper?
  void Release_Graph(Graph G) {
    auto S = G.move_version();
    VG.release_version(std::move(S));
  }

  void Update_Graph(Graph G) {
    auto S = G.move_version();
    GraphT new_G = G.move_graph();
    VG.add_version_from_graph(std::move(new_G));
    VG.release_version(std::move(S));
  }

  void save(char* oFile) {}

 private:
  aspen::versioned_graph<GraphT> VG;
  size_t maxDeg;
};