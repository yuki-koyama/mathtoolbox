#include <cstdlib>
#include <mathtoolbox/classical-mds.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

void SetSeed(const unsigned seed)
{
    srand(seed);
}

PYBIND11_MODULE(pymathtoolbox, m)
{
    m.doc() = "mathtoolbox python bindings";

    m.def("set_seed", &SetSeed);

    m.def("compute_classical_mds",
          &mathtoolbox::ComputeClassicalMds,
          "A function which computes classical MDS",
          pybind11::arg("D"),
          pybind11::arg("dim"));
}
