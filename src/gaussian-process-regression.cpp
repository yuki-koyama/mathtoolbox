#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <iostream>
#include <mathtoolbox/constants.hpp>
#include <mathtoolbox/gaussian-process-regression.hpp>
#include <mathtoolbox/kernel-functions.hpp>
#include <mathtoolbox/numerical-optimization.hpp>
#include <tuple>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace
{
    inline VectorXd Concat(const double a, const VectorXd& b)
    {
        VectorXd result(1 + b.size());

        result(0)                   = a;
        result.segment(1, b.size()) = b;

        return result;
    }

    inline VectorXd Concat(const double a, const double b, const VectorXd& c)
    {
        VectorXd result(2 + c.size());

        result(0)                   = a;
        result(1)                   = b;
        result.segment(2, c.size()) = c;

        return result;
    }

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
                         const double               sigma_squared_n,
                         const VectorXd&            kernel_hyperparams,
                         const mathtoolbox::Kernel& kernel)
    {
        const int      N   = X.cols();
        const MatrixXd K_f = CalcLargeKF(X, kernel_hyperparams, kernel);

        return K_f + sigma_squared_n * MatrixXd::Identity(N, N);
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

    MatrixXd CalcLargeKDerivSigmaSquaredN(const MatrixXd& X, const double sigma_squared_n)
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

        Eigen::MatrixXd k_deriv_x(num_dims, num_points);
        for (int i = 0; i < num_points; ++i)
        {
            k_deriv_x.col(i) = kernel_deriv_first_arg(x, X.col(i), kernel_hyperparams);
        }

        return k_deriv_x;
    }

    double CalcLogLikelihood(const MatrixXd&            X,
                             const VectorXd&            y,
                             const double               sigma_squared_n,
                             const VectorXd&            kernel_hyperparams,
                             const mathtoolbox::Kernel& kernel)
    {
        const int      N   = X.cols();
        const MatrixXd K_y = CalcLargeKY(X, sigma_squared_n, kernel_hyperparams, kernel);

        const Eigen::LLT<MatrixXd> K_y_llt(K_y);

        const VectorXd K_y_inv_y   = K_y_llt.solve(y);
        const double   log_K_y_det = 2.0 * K_y_llt.matrixL().toDenseMatrix().diagonal().array().log().sum();

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
                                    const double                               sigma_squared_n,
                                    const VectorXd&                            kernel_hyperparams,
                                    const mathtoolbox::Kernel&                 kernel,
                                    const mathtoolbox::KernelThetaIDerivative& kernel_deriv_theta_i)
    {
        const int D = X.rows();

        const MatrixXd K_y       = CalcLargeKY(X, sigma_squared_n, kernel_hyperparams, kernel);
        const MatrixXd K_y_inv   = K_y.inverse();
        const VectorXd K_y_inv_y = K_y_inv * y;

        const double log_likeliehood_deriv_sigma_squared_f = [&]() {
            // Equation 5.9 [Rasmussen and Williams 2006]
            const MatrixXd K_deriv_sigma_squared_f =
                CalcLargeKYDerivKernelHyperparamsI(X, kernel_hyperparams, 0, kernel_deriv_theta_i);
            const double term1 = +0.5 * K_y_inv_y.transpose() * K_deriv_sigma_squared_f * K_y_inv_y;
            const double term2 = -0.5 * (K_y_inv * K_deriv_sigma_squared_f).trace();
            return term1 + term2;
        }();

        const double log_likeliehood_deriv_sigma_squared_n = [&]() {
            // Equation 5.9 [Rasmussen and Williams 2006]
            const MatrixXd K_deriv_sigma_squared_n = CalcLargeKDerivSigmaSquaredN(X, sigma_squared_n);
            const double   term1                   = +0.5 * K_y_inv_y.transpose() * K_deriv_sigma_squared_n * K_y_inv_y;
            const double   term2                   = -0.5 * (K_y_inv * K_deriv_sigma_squared_n).trace();
            return term1 + term2;
        }();

        const VectorXd log_likelihood_deriv_length_scales = [&]() {
            // Equation 5.9 [Rasmussen and Williams 2006]
            VectorXd log_likelihood_deriv_length_scales(D);
            for (int i = 0; i < D; ++i)
            {
                const MatrixXd K_gradient_length_scale_i =
                    CalcLargeKYDerivKernelHyperparamsI(X, kernel_hyperparams, i + 1, kernel_deriv_theta_i);
                const double term1 = +0.5 * K_y_inv_y.transpose() * K_gradient_length_scale_i * K_y_inv_y;
                const double term2 = -0.5 * (K_y_inv * K_gradient_length_scale_i).trace();
                log_likelihood_deriv_length_scales(i) = term1 + term2;
            }
            return log_likelihood_deriv_length_scales;
        }();

        return Concat(log_likeliehood_deriv_sigma_squared_f,
                      log_likeliehood_deriv_sigma_squared_n,
                      log_likelihood_deriv_length_scales);
    }
} // namespace

mathtoolbox::GaussianProcessRegression::GaussianProcessRegression(const MatrixXd&  X,
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
        m_data_mu    = y.mean();
        m_data_sigma = std::max(std::sqrt((1.0 / static_cast<double>(y.size())) *
                                          (y - VectorXd::Constant(y.size(), m_data_mu)).squaredNorm()),
                                1e-32);
        m_data_scale = 0.20; // This value is empirically set
    }
    else
    {
        m_data_mu    = 0.0;
        m_data_sigma = 1.0;
        m_data_scale = 1.0;
    }

    // Store a normalization data values
    m_y = (m_data_scale / m_data_sigma) * (y - VectorXd::Constant(y.size(), m_data_mu));

    const int D = X.rows();

    SetHyperparams(0.10, 1e-05, VectorXd::Constant(D, 0.10));
}

void mathtoolbox::GaussianProcessRegression::SetHyperparams(double          sigma_squared_f,
                                                            double          sigma_squared_n,
                                                            const VectorXd& length_scales)
{
    this->m_kernel_hyperparams = Concat(sigma_squared_f, length_scales);
    this->m_sigma_squared_n    = sigma_squared_n;

    m_K_y     = CalcLargeKY(m_X, m_sigma_squared_n, m_kernel_hyperparams, m_kernel);
    m_K_y_inv = m_K_y.inverse();
}

void mathtoolbox::GaussianProcessRegression::PerformMaximumLikelihood(double          sigma_squared_f_initial,
                                                                      double          sigma_squared_n_initial,
                                                                      const VectorXd& length_scales_initial)
{
    const int D = m_X.rows();

    assert(m_kernel_hyperparams.size() == D + 1);
    assert(length_scales_initial.rows() == D);

    const VectorXd x_initial = [&]() {
        VectorXd x(D + 2);
        x(0)            = sigma_squared_f_initial;
        x(1)            = sigma_squared_n_initial;
        x.segment(2, D) = length_scales_initial;
        return x;
    }();
    const VectorXd upper = VectorXd::Constant(D + 2, 1e+04);
    const VectorXd lower = VectorXd::Constant(D + 2, 1e-05);

    using Data = std::tuple<const MatrixXd&, const VectorXd&>;
    Data data(m_X, m_y);

    // Currently, the mathtoolbox does not have any numerical optimization algorithms that support lower- and
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

    std::function<double(const VectorXd&)> f = [&](const VectorXd& x) -> double {
        const auto decoded_x = decode_vector(x);

        const double   sigma_squared_f = decoded_x[0];
        const double   sigma_squared_n = decoded_x[1];
        const VectorXd length_scales   = decoded_x.segment(2, x.size() - 2);

        const MatrixXd& X = std::get<0>(data);
        const VectorXd& y = std::get<1>(data);

        const double log_likelihood =
            CalcLogLikelihood(X, y, sigma_squared_n, Concat(sigma_squared_f, length_scales), m_kernel);

        return log_likelihood;
    };

    std::function<VectorXd(const VectorXd&)> g = [&](const VectorXd& x) -> VectorXd {
        const auto decoded_x = decode_vector(x);

        const double   sigma_squared_f = decoded_x[0];
        const double   sigma_squared_n = decoded_x[1];
        const VectorXd length_scales   = decoded_x.segment(2, x.size() - 2);

        const MatrixXd& X = std::get<0>(data);
        const VectorXd& y = std::get<1>(data);

        const VectorXd log_likelihood_deriv = CalcLogLikelihoodDeriv(
            X, y, sigma_squared_n, Concat(sigma_squared_f, length_scales), m_kernel, m_kernel_deriv_theta_i);

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

    m_kernel_hyperparams[0]            = x_optimal[0];
    m_kernel_hyperparams.segment(1, D) = x_optimal.segment(2, D);
    m_sigma_squared_n                  = x_optimal[1];

    std::cout << "sigma_squared_f: " << m_kernel_hyperparams[0] << std::endl;
    std::cout << "sigma_squared_n: " << m_sigma_squared_n << std::endl;
    std::cout << "length_scales  : " << m_kernel_hyperparams.segment(1, D).transpose() << std::endl;

    m_K_y     = CalcLargeKY(m_X, m_sigma_squared_n, m_kernel_hyperparams, m_kernel);
    m_K_y_inv = m_K_y.inverse();
}

double mathtoolbox::GaussianProcessRegression::PredictMean(const VectorXd& x) const
{
    const VectorXd k               = CalcSmallK(x, m_X, m_kernel_hyperparams, m_kernel);
    const double   normalized_mean = k.transpose() * (m_K_y_inv * m_y);

    return (m_data_sigma / m_data_scale) * normalized_mean + m_data_mu;
}

double mathtoolbox::GaussianProcessRegression::PredictStdev(const VectorXd& x) const
{
    const VectorXd k                = CalcSmallK(x, m_X, m_kernel_hyperparams, m_kernel);
    const double   k_xx             = m_kernel_hyperparams[0];
    const double   normalized_stdev = std::sqrt(k_xx - k.transpose() * (m_K_y_inv * k));

    return (m_data_sigma / m_data_scale) * normalized_stdev;
}

VectorXd mathtoolbox::GaussianProcessRegression::PredictMeanDeriv(const VectorXd& x) const
{
    const MatrixXd k_deriv_x = CalcSmallKDerivSmallX(x, m_X, m_kernel_hyperparams, m_kernel_deriv_first_arg);
    const VectorXd normalized_mean_deriv = k_deriv_x * (m_K_y_inv * m_y);

    return (m_data_sigma / m_data_scale) * normalized_mean_deriv;
}

VectorXd mathtoolbox::GaussianProcessRegression::PredictStdevDeriv(const VectorXd& x) const
{
    const MatrixXd k_deriv_x        = CalcSmallKDerivSmallX(x, m_X, m_kernel_hyperparams, m_kernel_deriv_first_arg);
    const VectorXd k                = CalcSmallK(x, m_X, m_kernel_hyperparams, m_kernel);
    const double   k_xx             = m_kernel_hyperparams[0];
    const VectorXd K_y_inv_k        = m_K_y_inv * k;
    const double   normalized_stdev = std::sqrt(k_xx - k.transpose() * K_y_inv_k);
    const VectorXd normalized_stdev_deriv = -(1.0 / normalized_stdev) * k_deriv_x * K_y_inv_k;

    return (m_data_sigma / m_data_scale) * normalized_stdev_deriv;
}
