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

    MatrixXd
    CalculateLargeKF(const MatrixXd& X, const VectorXd& kernel_hyperparameters, const mathtoolbox::Kernel& kernel)
    {
        const int N = X.cols();

        MatrixXd K_f(N, N);
        for (int i = 0; i < N; ++i)
        {
            for (int j = i; j < N; ++j)
            {
                const double value = kernel(X.col(i), X.col(j), kernel_hyperparameters);

                K_f(i, j) = value;
                K_f(j, i) = value;
            }
        }
        return K_f;
    }

    MatrixXd CalculateLargeKY(const MatrixXd&            X,
                              const double               sigma_squared_n,
                              const VectorXd&            kernel_hyperparameters,
                              const mathtoolbox::Kernel& kernel)
    {
        const int      N   = X.cols();
        const MatrixXd K_f = CalculateLargeKF(X, kernel_hyperparameters, kernel);

        return K_f + sigma_squared_n * MatrixXd::Identity(N, N);
    }

    MatrixXd
    CalculateLargeKYGradientKernelHyperparametersI(const MatrixXd&                            X,
                                                   const VectorXd&                            kernel_hyperparameters,
                                                   const int                                  index,
                                                   const mathtoolbox::KernelThetaIDerivative& kernel_theta_i_derivative)
    {
        const int N = X.cols();

        MatrixXd K_y_gradient_kernel_hyperparameters_i(N, N);
        for (int i = 0; i < N; ++i)
        {
            for (int j = i; j < N; ++j)
            {
                const double kernel_sigma_squared_f_derivative =
                    kernel_theta_i_derivative(X.col(i), X.col(j), kernel_hyperparameters, index);

                K_y_gradient_kernel_hyperparameters_i(i, j) = kernel_sigma_squared_f_derivative;
                K_y_gradient_kernel_hyperparameters_i(j, i) = kernel_sigma_squared_f_derivative;
            }
        }
        return K_y_gradient_kernel_hyperparameters_i;
    }

    MatrixXd CalculateLargeKGradientSigmaSquaredN(const MatrixXd& X, const double sigma_squared_n)
    {
        const int N = X.cols();
        return MatrixXd::Identity(N, N);
    }

    VectorXd CalculateSmallK(const VectorXd&            x,
                             const MatrixXd&            X,
                             const VectorXd&            kernel_hyperparameters,
                             const mathtoolbox::Kernel& kernel)
    {
        const int N = X.cols();
        return [&]() {
            VectorXd k(N);
            for (unsigned i = 0; i < N; ++i)
            {
                k(i) = kernel(x, X.col(i), kernel_hyperparameters);
            }
            return k;
        }();
    }

    double CalculateLogLikelihood(const MatrixXd&            X,
                                  const VectorXd&            y,
                                  const double               sigma_squared_n,
                                  const VectorXd&            kernel_hyperparameters,
                                  const mathtoolbox::Kernel& kernel)
    {
        const int      N   = X.cols();
        const MatrixXd K_y = CalculateLargeKY(X, sigma_squared_n, kernel_hyperparameters, kernel);

        const Eigen::FullPivLU<MatrixXd> lu(K_y);

        if (!lu.isInvertible()) { throw std::runtime_error("Non-invertible K_y is detected."); }

        const MatrixXd K_y_inv = lu.inverse();
        const double   K_y_det = lu.determinant();

        // Equation 5.8 [Rasmuss and Williams 2006]
        const double term1 = -0.5 * y.transpose() * K_y_inv * y;
        const double term2 = -0.5 * std::log(K_y_det);
        const double term3 = -0.5 * N * std::log(2.0 * mathtoolbox::constants::pi);

        if (std::isinf(term2)) { throw std::runtime_error("Inf is detected."); }

        return term1 + term2 + term3;
    }

    VectorXd CalculateLogLikelihoodGradient(const MatrixXd&                            X,
                                            const VectorXd&                            y,
                                            const double                               sigma_squared_n,
                                            const VectorXd&                            kernel_hyperparameters,
                                            const mathtoolbox::Kernel&                 kernel,
                                            const mathtoolbox::KernelThetaIDerivative& kernel_theta_i_derivative)
    {
        const int D = X.rows();

        const MatrixXd K_y     = CalculateLargeKY(X, sigma_squared_n, kernel_hyperparameters, kernel);
        const MatrixXd K_y_inv = K_y.inverse();

        const double log_likeliehood_gradient_sigma_squared_f = [&]() {
            // Equation 5.9 [Rasmuss and Williams 2006]
            const MatrixXd K_gradient_sigma_squared_f =
                CalculateLargeKYGradientKernelHyperparametersI(X, kernel_hyperparameters, 0, kernel_theta_i_derivative);
            const double term1 = +0.5 * y.transpose() * K_y_inv * K_gradient_sigma_squared_f * K_y_inv * y;
            const double term2 = -0.5 * (K_y_inv * K_gradient_sigma_squared_f).trace();
            return term1 + term2;
        }();

        const double log_likeliehood_gradient_sigma_squared_n = [&]() {
            // Equation 5.9 [Rasmuss and Williams 2006]
            const MatrixXd K_gradient_sigma_squared_n = CalculateLargeKGradientSigmaSquaredN(X, sigma_squared_n);
            const double   term1 = +0.5 * y.transpose() * K_y_inv * K_gradient_sigma_squared_n * K_y_inv * y;
            const double   term2 = -0.5 * (K_y_inv * K_gradient_sigma_squared_n).trace();
            return term1 + term2;
        }();

        const VectorXd log_likelihood_gradient_length_scales = [&]() {
            // Equation 5.9 [Rasmuss and Williams 2006]
            VectorXd log_likelihood_gradient_length_scales(D);
            for (int i = 0; i < D; ++i)
            {
                const MatrixXd K_gradient_length_scale_i = CalculateLargeKYGradientKernelHyperparametersI(
                    X, kernel_hyperparameters, i + 1, kernel_theta_i_derivative);
                const double term1 = +0.5 * y.transpose() * K_y_inv * K_gradient_length_scale_i * K_y_inv * y;
                const double term2 = -0.5 * (K_y_inv * K_gradient_length_scale_i).trace();
                log_likelihood_gradient_length_scales(i) = term1 + term2;
            }
            return log_likelihood_gradient_length_scales;
        }();

        return Concat(log_likeliehood_gradient_sigma_squared_f,
                      log_likeliehood_gradient_sigma_squared_n,
                      log_likelihood_gradient_length_scales);
    }
} // namespace

namespace mathtoolbox
{
    GaussianProcessRegression::GaussianProcessRegression(const MatrixXd&  X,
                                                         const VectorXd&  y,
                                                         const KernelType kernel_type)
        : X(X), y(y)
    {
        switch (kernel_type)
        {
        case KernelType::ArdSquaredExp:
            kernel                    = GetArdSquaredExpKernel;
            kernel_theta_i_derivative = GetArdSquaredExpKernelThetaIDerivative;
            break;
        case KernelType::Matern52:
            kernel                    = GetArdMatern52Kernel;
            kernel_theta_i_derivative = GetArdMatern52KernelThetaIDerivative;
            break;
        }

        const int D = X.rows();

        SetHyperparameters(0.10, 1e-05, VectorXd::Constant(D, 0.10));
    }

    void GaussianProcessRegression::SetHyperparameters(double                 sigma_squared_f,
                                                       double                 sigma_squared_n,
                                                       const Eigen::VectorXd& length_scales)
    {
        this->sigma_squared_f = sigma_squared_f;
        this->sigma_squared_n = sigma_squared_n;
        this->length_scales   = length_scales;

        K     = CalculateLargeKY(X, sigma_squared_n, Concat(sigma_squared_f, length_scales), kernel);
        K_inv = K.inverse();
    }

    void GaussianProcessRegression::PerformMaximumLikelihood(double                 sigma_squared_f_initial,
                                                             double                 sigma_squared_n_initial,
                                                             const Eigen::VectorXd& length_scales_initial)
    {
        const int D = X.rows();

        assert(length_scales.rows() == D);
        assert(length_scales_initial.rows() == D);

        const VectorXd x_initial = [&]() {
            VectorXd x(D + 2);
            x(0)            = sigma_squared_f_initial;
            x(1)            = sigma_squared_n_initial;
            x.segment(2, D) = length_scales_initial;
            return x;
        }();
        const VectorXd upper = VectorXd::Constant(D + 2, 1e+03);
        const VectorXd lower = VectorXd::Constant(D + 2, 1e-05);

        using Data = std::tuple<const MatrixXd&, const VectorXd&>;
        Data data(X, y);

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

        const auto calc_decode_value_derivative = [&sigmoid](const double x, const double l, const double u) {
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

        const auto calc_decode_vector_derivative = [&calc_decode_value_derivative, &lower, &upper](const VectorXd& x) {
            auto grad = VectorXd(x.size());
            for (int i = 0; i < x.size(); ++i)
            {
                grad[i] = calc_decode_value_derivative(x[i], lower[i], upper[i]);
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
                CalculateLogLikelihood(X, y, sigma_squared_n, Concat(sigma_squared_f, length_scales), kernel);

            return log_likelihood;
        };

        std::function<VectorXd(const VectorXd&)> g = [&](const VectorXd& x) -> VectorXd {
            const auto decoded_x = decode_vector(x);

            const double   sigma_squared_f = decoded_x[0];
            const double   sigma_squared_n = decoded_x[1];
            const VectorXd length_scales   = decoded_x.segment(2, x.size() - 2);

            const MatrixXd& X = std::get<0>(data);
            const VectorXd& y = std::get<1>(data);

            const VectorXd log_likelihood_gradient = CalculateLogLikelihoodGradient(
                X, y, sigma_squared_n, Concat(sigma_squared_f, length_scales), kernel, kernel_theta_i_derivative);

            return (log_likelihood_gradient.array() * calc_decode_vector_derivative(x).array()).matrix();
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

        sigma_squared_f = x_optimal[0];
        sigma_squared_n = x_optimal[1];
        length_scales   = x_optimal.segment(2, x_optimal.size() - 2);

        std::cout << "sigma_squared_f: " << sigma_squared_f << std::endl;
        std::cout << "sigma_squared_n: " << sigma_squared_n << std::endl;
        std::cout << "length_scales  : " << length_scales.transpose() << std::endl;

        K     = CalculateLargeKY(X, sigma_squared_n, Concat(sigma_squared_f, length_scales), kernel);
        K_inv = K.inverse();
    }

    double GaussianProcessRegression::EstimateY(const VectorXd& x) const
    {
        const VectorXd k = CalculateSmallK(x, X, Concat(sigma_squared_f, length_scales), kernel);
        return k.transpose() * K_inv * y;
    }

    double GaussianProcessRegression::EstimateVariance(const VectorXd& x) const
    {
        const VectorXd k = CalculateSmallK(x, X, Concat(sigma_squared_f, length_scales), kernel);
        return sigma_squared_f - k.transpose() * K_inv * k;
    }
} // namespace mathtoolbox
