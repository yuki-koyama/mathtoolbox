#ifndef MATHTOOLBOX_ACQUISITION_FUNCTIONS_HPP
#define MATHTOOLBOX_ACQUISITION_FUNCTIONS_HPP

#include <Eigen/Core>
#include <functional>

namespace mathtoolbox
{
    double GetExpectedImprovement(const Eigen::VectorXd&                               x,
                                  const std::function<double(const Eigen::VectorXd&)>& mu,
                                  const std::function<double(const Eigen::VectorXd&)>& sigma,
                                  const Eigen::VectorXd&                               x_best);

    Eigen::VectorXd
    GetExpectedImprovementDerivative(const Eigen::VectorXd&                                        x,
                                     const std::function<double(const Eigen::VectorXd&)>&          mu,
                                     const std::function<double(const Eigen::VectorXd&)>&          sigma,
                                     const Eigen::VectorXd&                                        x_best,
                                     const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& mu_derivative,
                                     const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& sigma_derivative);

    /// \param hyperparam The hyperparameter that controls the trade-off of exploration and exploitation. Specifically,
    /// this hyperparameter corresponds to the square root of the beta in [Srinivas et al. ICML '10]. Setting this to
    /// zero means pure exploitation, and setting to a very large value means (almost) pure exploration. This value
    /// needs to be non-negative.
    double GetGaussianProcessUpperConfidenceBound(const Eigen::VectorXd&                               x,
                                                  const std::function<double(const Eigen::VectorXd&)>& mu,
                                                  const std::function<double(const Eigen::VectorXd&)>& sigma,
                                                  const double                                         hyperparam);

    /// \param hyperparam The hyperparameter that controls the trade-off of exploration and exploitation. Specifically,
    /// this hyperparameter corresponds to the square root of the beta in [Srinivas et al. ICML '10]. Setting this to
    /// zero means pure exploitation, and setting to a very large value means (almost) pure exploration. This value
    /// needs to be non-negative.
    Eigen::VectorXd GetGaussianProcessUpperConfidenceBoundDerivative(
        const Eigen::VectorXd&                                        x,
        const std::function<double(const Eigen::VectorXd&)>&          mu,
        const std::function<double(const Eigen::VectorXd&)>&          sigma,
        const double                                                  hyperparam,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& mu_derivative,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& sigma_derivative);
} // namespace mathtoolbox

#endif // MATHTOOLBOX_ACQUISITION_FUNCTIONS_HPP
