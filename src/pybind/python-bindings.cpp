#include <cstdlib>
#include <functional>
#include <mathtoolbox/bayesian-optimization.hpp>
#include <mathtoolbox/classical-mds.hpp>
#include <mathtoolbox/rbf-interpolation.hpp>
#include <mathtoolbox/som.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace mt = mathtoolbox;

void SetSeed(const unsigned seed)
{
    srand(seed);
}

PYBIND11_MODULE(pymathtoolbox, m)
{
    m.doc() = "mathtoolbox python bindings";

    m.def("set_seed", &SetSeed, py::arg("seed"));

    // bayesian-optimization

    py::class_<mt::optimization::BayesianOptimizer>(m, "BayesianOptimizer")
        .def(py::init<const std::function<double(const Eigen::VectorXd&)>&,
                      const Eigen::VectorXd&,
                      const Eigen::VectorXd&>(),
             py::arg("f"),
             py::arg("lower_bound"),
             py::arg("upper_bound"))
        .def("step", &mt::optimization::BayesianOptimizer::Step)
        .def("evaluate_point", &mt::optimization::BayesianOptimizer::EvaluatePoint)
        .def("get_current_optimizer", &mt::optimization::BayesianOptimizer::GetCurrentOptimizer)
        .def("predict_mean", &mt::optimization::BayesianOptimizer::PredictMean)
        .def("predict_stdev", &mt::optimization::BayesianOptimizer::PredictStdev)
        .def("calc_acquisition_value", &mt::optimization::BayesianOptimizer::CalcAcquisitionValue)
        .def("get_data", &mt::optimization::BayesianOptimizer::GetData);

    // classical-mds

    m.def("compute_classical_mds",
          &mt::ComputeClassicalMds,
          "Compute low-dimensional embedding by using classical multi-dimensional scaling (MDS)",
          py::arg("D"),
          py::arg("target_dim"));

    // rbf-interpolation

    py::class_<mt::GaussianRbfKernel>(m, "GaussianRbfKernel")
        .def(py::init<double>(), py::arg("theta"))
        .def("__call__", &mt::GaussianRbfKernel::operator(), py::arg("r"));

    py::class_<mt::LinearRbfKernel>(m, "LinearRbfKernel")
        .def(py::init<>())
        .def("__call__", &mt::LinearRbfKernel::operator(), py::arg("r"));

    py::class_<mt::ThinPlateSplineRbfKernel>(m, "ThinPlateSplineRbfKernel")
        .def(py::init<>())
        .def("__call__", &mt::ThinPlateSplineRbfKernel::operator(), py::arg("r"));

    py::class_<mt::CubicRbfKernel>(m, "CubicRbfKernel")
        .def(py::init<>())
        .def("__call__", &mt::CubicRbfKernel::operator(), py::arg("r"));

    py::class_<mt::RbfInterpolator>(m, "RbfInterpolator")
        .def(py::init<std::function<double(double)>, const bool>(),
             py::arg("rbf_kernel")          = mt::ThinPlateSplineRbfKernel(),
             py::arg("use_polynomial_term") = true)
        .def("set_data", &mt::RbfInterpolator::SetData, py::arg("X"), py::arg("y"))
        .def("calc_weights",
             &mt::RbfInterpolator::CalcWeights,
             py::arg("use_regularization") = false,
             py::arg("lambda")             = 0.001)
        .def("calc_value", &mt::RbfInterpolator::CalcValue, py::arg("x"));

    // som

    py::class_<mt::Som>(m, "Som")
        .def(py::init<const Eigen::MatrixXd&,
                      const int,
                      const int,
                      const double,
                      const double,
                      const double,
                      const bool>(),
             py::arg("data"),
             py::arg("latent_num_dims")      = 2,
             py::arg("resolution")           = 10,
             py::arg("init_var")             = 0.50,
             py::arg("min_var")              = 0.01,
             py::arg("var_decreasing_speed") = 50.0,
             py::arg("normalize_data")       = true)
        .def("get_latent_space_node_positions", &mt::Som::GetLatentSpaceNodePositions)
        .def("get_data_space_node_positions", &mt::Som::GetDataSpaceNodePositions)
        .def("get_latent_space_node_positions", &mt::Som::GetLatentSpaceDataPositions)
        .def("step", &mt::Som::Step);
}
