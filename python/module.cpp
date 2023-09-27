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

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/numpy.h"

#include "builder.cpp"
#include "vamana_index.cpp"
#include "../algorithms/IVF/IVF.h"
#include "../algorithms/IVF/posting_list.h"
#include "../algorithms/utils/filters.h"

PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

namespace py = pybind11;
using namespace pybind11::literals;

// using NeighborsAndDistances = std::pair<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>, py::array_t<float, py::array::c_style | py::array::forcecast>>;

struct Variant
{
    std::string builder_name;
    std::string index_name;
    std::string ivf_name;
};

const Variant FloatEuclidianVariant{"build_vamana_float_euclidian_index", "VamanaFloatEuclidianIndex", "IVFFloatEuclidianIndex"};
const Variant FloatMipsVariant{"build_vamana_float_mips_index", "VamanaFloatMipsIndex", "IVFFloatMipsIndex"};

const Variant UInt8EuclidianVariant{"build_vamana_uint8_euclidian_index", "VamanaUInt8EuclidianIndex", "IVFUInt8EuclidianIndex"};
const Variant UInt8MipsVariant{"build_vamana_uint8_mips_index", "VamanaUInt8MipsIndex", "IVFUInt8MipsIndex"};

const Variant Int8EuclidianVariant{"build_vamana_int8_euclidian_index", "VamanaInt8EuclidianIndex", "IVFInt8EuclidianIndex"};
const Variant Int8MipsVariant{"build_vamana_int8_mips_index", "VamanaInt8MipsIndex", "IVFInt8MipsIndex"};

template <typename T, typename Point> inline void add_variant(py::module_ &m, const Variant &variant)
{

    m.def(variant.builder_name.c_str(), build_vamana_index<T, Point>, "distance_metric"_a,
          "data_file_path"_a, "index_output_path"_a, "graph_degree"_a, "beam_width"_a, "alpha"_a);

    py::class_<VamanaIndex<T, Point>>(m, variant.index_name.c_str())
        .def(py::init<std::string &, std::string &, size_t, size_t>(),
             "index_path"_a, "data_path"_a, "num_points"_a, "dimensions"_a) //maybe these last two are unnecessary?
        //do we want to add options like visited limit, or leave those as defaults?
        .def("batch_search", &VamanaIndex<T, Point>::batch_search, "queries"_a, "num_queries"_a, "knn"_a,
             "beam_width"_a)
        .def("batch_search_from_string", &VamanaIndex<T, Point>::batch_search_from_string, "queries"_a, "num_queries"_a, "knn"_a,
             "beam_width"_a)
        .def("check_recall", &VamanaIndex<T, Point>::check_recall, "gFile"_a, "neighbors"_a, "k"_a);

    py::class_<IVFIndex<T, Point, NaivePostingList<T, Point>>>(m, variant.ivf_name.c_str())
        .def(py::init())
        .def("fit", &IVFIndex<T, Point, NaivePostingList<T, Point>>::fit, "points"_a, "cluster_size"_a)
        .def("fit_from_filename", &IVFIndex<T, Point, NaivePostingList<T, Point>>::fit_from_filename, "filename"_a, "cluster_size"_a)
        .def("batch_search", &IVFIndex<T, Point, NaivePostingList<T, Point>>::batch_search, "queries"_a, "num_queries"_a, "knn"_a, "n_lists"_a)
        .def("print_stats", &IVFIndex<T, Point, NaivePostingList<T, Point>>::print_stats);
}

PYBIND11_MODULE(_ParlayANNpy, m)
{
    m.doc() = "ParlayANN Python Bindings";
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

    // let's re-export our defaults
    py::module_ default_values = m.def_submodule(
        "defaults");

    default_values.attr("METRIC") = "Euclidian";
    default_values.attr("ALPHA") = 1.2;
    default_values.attr("GRAPH_DEGREE") = 64;
    default_values.attr("BEAMWIDTH") = 128;

    add_variant<float, Euclidian_Point<float>>(m, FloatEuclidianVariant);
    add_variant<float, Mips_Point<float>>(m, FloatMipsVariant);
    add_variant<uint8_t, Euclidian_Point<uint8_t>>(m, UInt8EuclidianVariant);
    add_variant<uint8_t, Mips_Point<uint8_t>>(m, UInt8MipsVariant);
    add_variant<int8_t, Euclidian_Point<int8_t>>(m, Int8EuclidianVariant);
    add_variant<int8_t, Mips_Point<int8_t>>(m, Int8MipsVariant);

    py::class_<csr_filters>(m, "csr_filters")
        .def(py::init<std::string &>())
        .def("match", &csr_filters::match, "p"_a, "f"_a)
        .def("first_label", &csr_filters::first_label, "p"_a)
        .def("print_stats", &csr_filters::print_stats)
        .def("filter_count", &csr_filters::filter_count, "f"_a)
        .def("point_count", &csr_filters::point_count, "p"_a)
        .def("transpose", &csr_filters::transpose);

}