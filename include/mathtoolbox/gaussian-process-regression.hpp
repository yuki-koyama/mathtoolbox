#ifndef MATHTOOLBOX_GAUSSIAN_PROCESS_REGRESSION_HPP
#define MATHTOOLBOX_GAUSSIAN_PROCESS_REGRESSION_HPP

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <mathtoolbox/kernel-functions.hpp>

namespace mathtoolbox
{
    class GaussianProcessRegressor
    {
    public:
        enum class KernelType
        {
            ArdSquaredExp,
            ArdMatern52
        };

        /// \brief Construct an instance with input data
        GaussianProcessRegressor(const Eigen::MatrixXd& X,
                                 const Eigen::VectorXd& y,
                                 const KernelType       kernel_type            = KernelType::ArdMatern52,
                                 const bool             use_data_normalization = true);

        /// \brief Calculate the mean of the predicted distribution
        double PredictMean(const Eigen::VectorXd& x) const;

        /// \brief Calculate the standard deviation of the predicted distribution
        double PredictStdev(const Eigen::VectorXd& x) const;

        /// \brief Calculate the derivative of the mean of the predicted distribution
        Eigen::VectorXd PredictMeanDeriv(const Eigen::VectorXd& x) const;

        /// \brief Calculate the derivative of the standard deviation of the predicted distribution
        Eigen::VectorXd PredictStdevDeriv(const Eigen::VectorXd& x) const;

        /// \brief Set hyperparameters directly
        ///
        /// \details Covariance matrix calculation will run within this method.
        void SetHyperparams(const Eigen::VectorXd& kernel_hyperparams, const double noise_hyperparam);

        /// \brief Perform maximum likelihood estimation of the hyperparameters
        void PerformMaximumLikelihood(const Eigen::VectorXd& kernel_hyperparams_initial,
                                      const double           noise_hyperparam_initial);

        /// \brief Get the input data points
        const Eigen::MatrixXd& GetDataPoints() const { return m_X; }

        /// \brief Get the input data values
        const Eigen::VectorXd& GetDataValues() const { return m_y; }

    private:
        // Data points
        Eigen::MatrixXd m_X;
        Eigen::VectorXd m_y;

        // Derivative data
        Eigen::MatrixXd             m_K_y;
        Eigen::LLT<Eigen::MatrixXd> m_K_y_llt;
        Eigen::VectorXd             m_K_y_inv_y;

        // Normalization parameters
        double m_data_mu;
        double m_data_sigma;
        double m_data_scale;

        // Hyperparameters
        Eigen::VectorXd m_kernel_hyperparams;
        double          m_noise_hyperparam;

        // Kernel functions
        Kernel                   m_kernel;
        KernelThetaIDerivative   m_kernel_deriv_theta_i;
        KernelFirstArgDerivative m_kernel_deriv_first_arg;
    };
} // namespace mathtoolbox

#endif // MATHTOOLBOX_GAUSSIAN_PROCESS_REGRESSION_HPP
