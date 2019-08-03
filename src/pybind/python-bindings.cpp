#include <mathtoolbox/classical-mds.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

PYBIND11_MODULE(pymathtoolbox, m)
{
    m.doc() = "mathtoolbox python bindings";

    m.def("compute_classical_mds",
          &mathtoolbox::ComputeClassicalMds,
          "A function which computes classical MDS",
          pybind11::arg("D"),
          pybind11::arg("dim"));
}
