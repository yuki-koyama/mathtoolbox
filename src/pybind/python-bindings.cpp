#include <cstdlib>
#include <functional>
#include <mathtoolbox/bayesian-optimization.hpp>
#include <mathtoolbox/classical-mds.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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
          py::arg("D"),
          py::arg("dim"));

    py::class_<mathtoolbox::optimization::BayesianOptimizer>(m, "BayesianOptimizer")
        .def(py::init<const std::function<double(const Eigen::VectorXd&)>&,
                      const Eigen::VectorXd&,
                      const Eigen::VectorXd&>())
        .def("step", &mathtoolbox::optimization::BayesianOptimizer::Step)
        .def("evaluate_point", &mathtoolbox::optimization::BayesianOptimizer::EvaluatePoint)
        .def("get_current_optimizer", &mathtoolbox::optimization::BayesianOptimizer::GetCurrentOptimizer)
        .def("predict_mean", &mathtoolbox::optimization::BayesianOptimizer::PredictMean)
        .def("predict_stdev", &mathtoolbox::optimization::BayesianOptimizer::PredictStdev)
        .def("calc_acquisition_value", &mathtoolbox::optimization::BayesianOptimizer::CalcAcquisitionValue)
        .def("get_data", &mathtoolbox::optimization::BayesianOptimizer::GetData);
}
