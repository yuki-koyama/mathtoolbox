#ifndef MATHTOOLBOX_GAUSSIAN_PROCESS_REGRESSION_HPP
#define MATHTOOLBOX_GAUSSIAN_PROCESS_REGRESSION_HPP

#include <Eigen/Core>
#include <mathtoolbox/kernel-functions.hpp>

namespace mathtoolbox
{
    class GaussianProcessRegression
    {
    public:
        enum class KernelType
        {
            ArdSquaredExp,
            ArdMatern52
        };

        // Construction with input data
        GaussianProcessRegression(const Eigen::MatrixXd& X,
                                  const Eigen::VectorXd& y,
                                  const KernelType       kernel_type = KernelType::ArdMatern52);

        // Estimation methods
        double PredictMean(const Eigen::VectorXd& x) const;
        double PredictVariance(const Eigen::VectorXd& x) const;

        // Hyperparameters setup methods
        void SetHyperparams(double sigma_squared_f, double sigma_squared_n, const Eigen::VectorXd& length_scales);
        void PerformMaximumLikelihood(double                 sigma_squared_f_initial,
                                      double                 sigma_squared_n_initial,
                                      const Eigen::VectorXd& length_scales_initial);

        // Getter methods
        const Eigen::MatrixXd& GetLargeX() const { return m_X; }
        const Eigen::VectorXd& GetY() const { return m_y; }

    private:
        // Data points
        Eigen::MatrixXd m_X;
        Eigen::VectorXd m_y;

        // Derivative data
        Eigen::MatrixXd m_K_y;
        Eigen::MatrixXd m_K_y_inv;

        // Hyperparameters
        Eigen::VectorXd m_kernel_hyperparams;
        double          m_sigma_squared_n;

        // Kernel functions
        Kernel                 m_kernel;
        KernelThetaIDerivative m_kernel_deriv_theta_i;
    };
} // namespace mathtoolbox

#endif // MATHTOOLBOX_GAUSSIAN_PROCESS_REGRESSION_HPP
