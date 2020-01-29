#include <iostream>
#include <mathtoolbox/constants.hpp>
#include <mathtoolbox/gaussian-process-regression.hpp>
#include <mathtoolbox/kernel-functions.hpp>
#include <mathtoolbox/log-determinant.hpp>
#include <mathtoolbox/numerical-optimization.hpp>
#include <tuple>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace
{
    inline VectorXd Concat(const VectorXd& a, const double b) { return (VectorXd(a.size() + 1) << a, b).finished(); }
    inline VectorXd Concat(const double a, const VectorXd& b) { return (VectorXd(1 + b.size()) << a, b).finished(); }

    MatrixXd CalcLargeKF(const MatrixXd& X, const VectorXd& kernel_hyperparams, const mathtoolbox::Kernel& kernel)
    {
        const int N = X.cols();

        MatrixXd K_f(N, N);
        for (int i = 0; i < N; ++i)
        {
            for (int j = i; j < N; ++j)
            {
                const double value = kernel(X.col(i), X.col(j), kernel_hyperparams);

                K_f(i, j) = value;
                K_f(j, i) = value;
            }
        }
        return K_f;
    }

    MatrixXd CalcLargeKY(const MatrixXd&            X,
                         const double               noise_hyperparam,
                         const VectorXd&            kernel_hyperparams,
                         const mathtoolbox::Kernel& kernel)
    {
        const int      N   = X.cols();
        const MatrixXd K_f = CalcLargeKF(X, kernel_hyperparams, kernel);

        return K_f + noise_hyperparam * MatrixXd::Identity(N, N);
    }

    MatrixXd CalcLargeKYDerivKernelHyperparamsI(const MatrixXd&                            X,
                                                const VectorXd&                            kernel_hyperparams,
                                                const int                                  index,
                                                const mathtoolbox::KernelThetaIDerivative& kernel_deriv_theta_i)
    {
        const int N = X.cols();

        MatrixXd K_y_deriv_kernel_hyperparams_i(N, N);
        for (int i = 0; i < N; ++i)
        {
            for (int j = i; j < N; ++j)
            {
                const double kernel_sigma_squared_f_derivative =
                    kernel_deriv_theta_i(X.col(i), X.col(j), kernel_hyperparams, index);

                K_y_deriv_kernel_hyperparams_i(i, j) = kernel_sigma_squared_f_derivative;
                K_y_deriv_kernel_hyperparams_i(j, i) = kernel_sigma_squared_f_derivative;
            }
        }
        return K_y_deriv_kernel_hyperparams_i;
    }

    MatrixXd CalcLargeKDerivNoiseHyperparam(const MatrixXd& X, const double noise_hyperparam)
    {
        const int N = X.cols();
        return MatrixXd::Identity(N, N);
    }

    VectorXd CalcSmallK(const VectorXd&            x,
                        const MatrixXd&            X,
                        const VectorXd&            kernel_hyperparams,
                        const mathtoolbox::Kernel& kernel)
    {
        const int N = X.cols();
        return [&]() {
            VectorXd k(N);
            for (unsigned i = 0; i < N; ++i)
            {
                k(i) = kernel(x, X.col(i), kernel_hyperparams);
            }
            return k;
        }();
    }

    MatrixXd CalcSmallKDerivSmallX(const VectorXd&                              x,
                                   const MatrixXd&                              X,
                                   const VectorXd&                              kernel_hyperparams,
                                   const mathtoolbox::KernelFirstArgDerivative& kernel_deriv_first_arg)
    {
        const int num_points = X.cols();
        const int num_dims   = X.rows();

        assert(num_dims > 0);
        assert(num_points > 0);
        assert(x.size() == num_dims);

        MatrixXd k_deriv_x(num_dims, num_points);
        for (int i = 0; i < num_points; ++i)
        {
            k_deriv_x.col(i) = kernel_deriv_first_arg(x, X.col(i), kernel_hyperparams);
        }

        return k_deriv_x;
    }

    double CalcLogLikelihood(const MatrixXd&            X,
                             const VectorXd&            y,
                             const VectorXd&            kernel_hyperparams,
                             const double               noise_hyperparam,
                             const mathtoolbox::Kernel& kernel)
    {
        const int      N   = X.cols();
        const MatrixXd K_y = CalcLargeKY(X, noise_hyperparam, kernel_hyperparams, kernel);

        const Eigen::LLT<MatrixXd> K_y_llt(K_y);

        const VectorXd K_y_inv_y   = K_y_llt.solve(y);
        const double   log_K_y_det = mathtoolbox::CalcLogDetOfSymmetricPositiveDefiniteMatrix(K_y_llt);

        // Equation 5.8 [Rasmussen and Williams 2006]
        const double term1 = -0.5 * y.transpose() * K_y_inv_y;
        const double term2 = -0.5 * log_K_y_det;
        const double term3 = -0.5 * N * std::log(2.0 * mathtoolbox::constants::pi);

        if (std::isinf(term2))
        {
            throw std::runtime_error("Inf is detected.");
        }

        return term1 + term2 + term3;
    }

    VectorXd CalcLogLikelihoodDeriv(const MatrixXd&                            X,
                                    const VectorXd&                            y,
                                    const VectorXd&                            kernel_hyperparams,
                                    const double                               noise_hyperparam,
                                    const mathtoolbox::Kernel&                 kernel,
                                    const mathtoolbox::KernelThetaIDerivative& kernel_deriv_theta_i)
    {
        const int num_kernel_hyperparams = kernel_hyperparams.size();

        const MatrixXd             K_y       = CalcLargeKY(X, noise_hyperparam, kernel_hyperparams, kernel);
        const Eigen::LLT<MatrixXd> K_y_llt   = Eigen::LLT<MatrixXd>(K_y);
        const VectorXd             K_y_inv_y = K_y_llt.solve(y);

        const VectorXd log_likeliehood_deriv_kernel_hyperparams = [&]() {
            // Equation 5.9 [Rasmussen and Williams 2006]
            VectorXd log_likelihood_deriv_kernel_hyperparams(num_kernel_hyperparams);
            for (int i = 0; i < num_kernel_hyperparams; ++i)
            {
                const MatrixXd K_deriv_kernel_hyperparams_i =
                    CalcLargeKYDerivKernelHyperparamsI(X, kernel_hyperparams, i, kernel_deriv_theta_i);
                const double term1 = +0.5 * K_y_inv_y.transpose() * K_deriv_kernel_hyperparams_i * K_y_inv_y;
                const double term2 = -0.5 * K_y_llt.solve(K_deriv_kernel_hyperparams_i).trace();
                log_likelihood_deriv_kernel_hyperparams(i) = term1 + term2;
            }
            return log_likelihood_deriv_kernel_hyperparams;
        }();

        const double log_likeliehood_deriv_noise_hyperparam = [&]() {
            // Equation 5.9 [Rasmussen and Williams 2006]
            const MatrixXd K_deriv_sigma_squared_n = CalcLargeKDerivNoiseHyperparam(X, noise_hyperparam);
            const double   term1                   = +0.5 * K_y_inv_y.transpose() * K_deriv_sigma_squared_n * K_y_inv_y;
            const double   term2                   = -0.5 * K_y_llt.solve(K_deriv_sigma_squared_n).trace();
            return term1 + term2;
        }();

        return Concat(log_likeliehood_deriv_kernel_hyperparams, log_likeliehood_deriv_noise_hyperparam);
    }
} // namespace

mathtoolbox::GaussianProcessRegressor::GaussianProcessRegressor(const MatrixXd&  X,
                                                                const VectorXd&  y,
                                                                const KernelType kernel_type,
                                                                const bool       use_data_normalization)
    : m_X(X)
{
    assert(X.cols() == y.size());
    assert(X.cols() != 0);
    assert(y.rows() != 0);

    switch (kernel_type)
    {
        case KernelType::ArdSquaredExp:
        {
            m_kernel                 = GetArdSquaredExpKernel;
            m_kernel_deriv_theta_i   = GetArdSquaredExpKernelThetaIDerivative;
            m_kernel_deriv_first_arg = GetArdSquaredExpKernelFirstArgDerivative;
            break;
        }
        case KernelType::ArdMatern52:
        {
            m_kernel                 = GetArdMatern52Kernel;
            m_kernel_deriv_theta_i   = GetArdMatern52KernelThetaIDerivative;
            m_kernel_deriv_first_arg = GetArdMatern52KernelFirstArgDerivative;
            break;
        }
    }

    // Calculate parameters for data standardization
    if (use_data_normalization)
    {
        m_data_mu = y.mean();

        if (y.size() == 1)
        {
            m_data_sigma = 1.0;
            m_data_scale = 1.0;
        }
        else
        {
            const double inv_size = (1.0 / static_cast<double>(y.size()));
            const double var      = inv_size * (y - VectorXd::Constant(y.size(), m_data_mu)).squaredNorm();

            m_data_sigma = std::max(std::sqrt(var), 1e-32);
            m_data_scale = 0.20; // This value is empirically set
        }
    }
    else
    {
        m_data_mu    = 0.0;
        m_data_sigma = 1.0;
        m_data_scale = 1.0;
    }

    // Store a normalization data values
    m_y = (m_data_scale / m_data_sigma) * (y - VectorXd::Constant(y.size(), m_data_mu));

    // Set default hyperparameters
    const int        num_dims                   = X.rows();
    const VectorXd   default_kernel_hyperparams = Concat(0.10, VectorXd::Constant(num_dims, 0.10));
    constexpr double default_noise_hyperparam   = 1e-05;

    SetHyperparams(default_kernel_hyperparams, default_noise_hyperparam);
}

void mathtoolbox::GaussianProcessRegressor::SetHyperparams(const Eigen::VectorXd& kernel_hyperparams,
                                                           const double           noise_hyperparam)
{
    m_kernel_hyperparams = kernel_hyperparams;
    m_noise_hyperparam   = noise_hyperparam;

    m_K_y       = CalcLargeKY(m_X, m_noise_hyperparam, m_kernel_hyperparams, m_kernel);
    m_K_y_llt   = Eigen::LLT<MatrixXd>(m_K_y);
    m_K_y_inv_y = m_K_y_llt.solve(m_y);
}

void mathtoolbox::GaussianProcessRegressor::PerformMaximumLikelihood(const Eigen::VectorXd& kernel_hyperparams_initial,
                                                                     const double           noise_hyperparam_initial)
{
    const int num_dims               = m_X.rows();
    const int num_kernel_hyperparams = kernel_hyperparams_initial.size();

    assert(m_kernel_hyperparams.size() == num_kernel_hyperparams);

    const VectorXd x_initial = Concat(kernel_hyperparams_initial, noise_hyperparam_initial);
    const VectorXd upper     = VectorXd::Constant(num_kernel_hyperparams + 1, 1e+03);
    const VectorXd lower     = VectorXd::Constant(num_kernel_hyperparams + 1, 1e-06);

    using Data = std::tuple<const MatrixXd&, const VectorXd&>;
    Data data(m_X, m_y);

    // Currently, the mathtoolbox does not have efficient numerical optimization algorithms that support lower- and
    // upper-bound conditions. The hyperparameters here should always be positive for evaluating the objective
    // function. Also, they should not be very large values because the covariance matrix becomes difficult to
    // inverse. To resolve these issues, here, the search variables are encoded using a variant of the logit
    // function. While this approach does not handle bound conditions in an exact sense, it works in this case.

    const auto sigmoid = [](const double x) { return 1.0 / (1.0 + std::exp(-x)); };

    const auto logit = [](const double x) { return std::log(x / (1.0 - x)); };

    const auto encode_value = [&logit](const double x, const double l, const double u) {
        return logit((x - l) / (u - l));
    };

    const auto decode_value = [&sigmoid](const double x, const double l, const double u) {
        return (u - l) * sigmoid(x) + l;
    };

    const auto calc_decode_value_deriv = [&sigmoid](const double x, const double l, const double u) {
        const double s = sigmoid(x);
        return (u - l) * s * (1.0 - s);
    };

    const auto encode_vector = [&encode_value, &lower, &upper](const VectorXd& x) {
        auto encoded_x = VectorXd(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            encoded_x[i] = encode_value(x[i], lower[i], upper[i]);
        }
        return encoded_x;
    };

    const auto decode_vector = [&decode_value, &lower, &upper](const VectorXd& x) {
        auto decoded_x = VectorXd(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            decoded_x[i] = decode_value(x[i], lower[i], upper[i]);
        }
        return decoded_x;
    };

    const auto calc_decode_vector_deriv = [&calc_decode_value_deriv, &lower, &upper](const VectorXd& x) {
        auto grad = VectorXd(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            grad[i] = calc_decode_value_deriv(x[i], lower[i], upper[i]);
        }
        return grad;
    };

    const auto f = [&](const VectorXd& x) -> double {
        const auto decoded_x = decode_vector(x);

        const VectorXd kernel_hyperparams = decoded_x.segment(0, x.size() - 1);
        const double   sigma_squared_n    = decoded_x(x.size() - 1);

        const MatrixXd& X = std::get<0>(data);
        const VectorXd& y = std::get<1>(data);

        const double log_likelihood = CalcLogLikelihood(X, y, kernel_hyperparams, sigma_squared_n, m_kernel);

        return log_likelihood;
    };

    const auto g = [&](const VectorXd& x) -> VectorXd {
        const auto decoded_x = decode_vector(x);

        const VectorXd kernel_hyperparams = decoded_x.segment(0, x.size() - 1);
        const double   sigma_squared_n    = decoded_x(x.size() - 1);

        const MatrixXd& X = std::get<0>(data);
        const VectorXd& y = std::get<1>(data);

        const VectorXd log_likelihood_deriv =
            CalcLogLikelihoodDeriv(X, y, kernel_hyperparams, sigma_squared_n, m_kernel, m_kernel_deriv_theta_i);

        return (log_likelihood_deriv.array() * calc_decode_vector_deriv(x).array()).matrix();
    };

    optimization::Setting input;
    input.algorithm          = optimization::Algorithm::LBfgs;
    input.x_init             = encode_vector(x_initial);
    input.f                  = f;
    input.g                  = g;
    input.epsilon            = 1e-06;
    input.max_num_iterations = 1000;
    input.type               = optimization::Type::Max;

    const auto     result    = optimization::RunOptimization(input);
    const VectorXd x_optimal = decode_vector(result.x_star);

    m_kernel_hyperparams = x_optimal.segment(0, num_kernel_hyperparams);
    m_noise_hyperparam   = x_optimal(num_kernel_hyperparams);

    assert(m_kernel_hyperparams.size() == num_dims + 1);
    std::cout << "sigma_squared_f: " << m_kernel_hyperparams[0] << std::endl;
    std::cout << "length_scales  : " << m_kernel_hyperparams.segment(1, num_dims).transpose() << std::endl;
    std::cout << "sigma_squared_n: " << m_noise_hyperparam << std::endl;

    m_K_y       = CalcLargeKY(m_X, m_noise_hyperparam, m_kernel_hyperparams, m_kernel);
    m_K_y_llt   = Eigen::LLT<MatrixXd>(m_K_y);
    m_K_y_inv_y = m_K_y_llt.solve(m_y);
}

double mathtoolbox::GaussianProcessRegressor::PredictMean(const VectorXd& x) const
{
    const VectorXd k               = CalcSmallK(x, m_X, m_kernel_hyperparams, m_kernel);
    const double   normalized_mean = k.transpose() * m_K_y_inv_y;

    return (m_data_sigma / m_data_scale) * normalized_mean + m_data_mu;
}

double mathtoolbox::GaussianProcessRegressor::PredictStdev(const VectorXd& x) const
{
    const VectorXd k                = CalcSmallK(x, m_X, m_kernel_hyperparams, m_kernel);
    const double   k_xx             = m_kernel_hyperparams[0];
    const double   normalized_stdev = std::sqrt(k_xx - k.transpose() * m_K_y_llt.solve(k));

    return (m_data_sigma / m_data_scale) * normalized_stdev;
}

VectorXd mathtoolbox::GaussianProcessRegressor::PredictMeanDeriv(const VectorXd& x) const
{
    const MatrixXd k_deriv_x = CalcSmallKDerivSmallX(x, m_X, m_kernel_hyperparams, m_kernel_deriv_first_arg);
    const VectorXd normalized_mean_deriv = k_deriv_x * m_K_y_inv_y;

    return (m_data_sigma / m_data_scale) * normalized_mean_deriv;
}

VectorXd mathtoolbox::GaussianProcessRegressor::PredictStdevDeriv(const VectorXd& x) const
{
    const MatrixXd k_deriv_x        = CalcSmallKDerivSmallX(x, m_X, m_kernel_hyperparams, m_kernel_deriv_first_arg);
    const VectorXd k                = CalcSmallK(x, m_X, m_kernel_hyperparams, m_kernel);
    const double   k_xx             = m_kernel_hyperparams[0];
    const VectorXd K_y_inv_k        = m_K_y_llt.solve(k);
    const double   normalized_stdev = std::sqrt(k_xx - k.transpose() * K_y_inv_k);
    const VectorXd normalized_stdev_deriv = -(1.0 / normalized_stdev) * k_deriv_x * K_y_inv_k;

    return (m_data_sigma / m_data_scale) * normalized_stdev_deriv;
}
