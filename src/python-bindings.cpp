#include <mathtoolbox/classical-mds.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

int add(int i, int j)
{
    return i + j;
}

PYBIND11_MODULE(pymathtoolbox, m)
{
    m.doc() = "mathtoolbox python bindings";

    m.def("ComputeClassicalMds", &mathtoolbox::ComputeClassicalMds, "A function which computes classical MDS");
}
