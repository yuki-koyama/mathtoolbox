#ifndef gaussian_process_regression_hpp
#define gaussian_process_regression_hpp

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
        double PredictVar(const Eigen::VectorXd& x) const;
        double PredictStdev(const Eigen::VectorXd& x) const;

        // Derivative estimation methods
        Eigen::VectorXd PredictMeanDeriv(const Eigen::VectorXd& x) const;
        Eigen::VectorXd PredictVarDeriv(const Eigen::VectorXd& x) const;
        Eigen::VectorXd PredictStdevDeriv(const Eigen::VectorXd& x) const;

        // Hyperparameters setup methods
        void SetHyperparameters(double sigma_squared_f, double sigma_squared_n, const Eigen::VectorXd& length_scales);
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
        Eigen::VectorXd m_kernel_hyperparameters;
        double          m_sigma_squared_n;

        // Kernel functions
        Kernel                   m_kernel;
        KernelThetaIDerivative   m_kernel_theta_i_derivative;
        KernelFirstArgDerivative m_kernel_deriv_first_arg;
    };
} // namespace mathtoolbox

#endif /* gaussian_process_regression_hpp */
