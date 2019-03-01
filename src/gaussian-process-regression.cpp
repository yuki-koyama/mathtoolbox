#include <mathtoolbox/gaussian-process-regression.hpp>
#include <tuple>
#include <Eigen/LU>
#include <nlopt-util.hpp>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace
{
    // Equation 5.1 [Rasmuss and Williams 2006]
    double CalculateArdSquaredExponentialKernel(const VectorXd& x_i,
                                                const VectorXd& x_j,
                                                const double    sigma_squared_f,
                                                const VectorXd& length_scales)
    {
        const int    D   = x_i.rows();
        const double sum = [&]()
        {
            double sum = 0.0;
            for (int i = 0; i < D; ++ i)
            {
                sum += (x_i(i) - x_j(i)) * (x_i(i) - x_j(i)) / (length_scales(i) * length_scales(i));
            }
            return sum;
        }();
        return sigma_squared_f * std::exp(- 0.5 * sum);
    }

    double CalculateArdSquaredExponentialKernelGradientSigmaSquaredF(const VectorXd& x_i,
                                                                     const VectorXd& x_j,
                                                                     const double    sigma_squared_f,
                                                                     const VectorXd& length_scales)
    {
        const int D = length_scales.rows();
        return [&]()
        {
            double sum = 0.0;
            for (int i = 0; i < D; ++ i)
            {
                sum += (x_i(i) - x_j(i)) * (x_i(i) - x_j(i)) / (length_scales(i) * length_scales(i));
            }
            return std::exp(- 0.5 * sum);
        }();
    }

    VectorXd CalculateArdSquaredExponentialKernelGradientLengthScales(const VectorXd& x_i,
                                                                      const VectorXd& x_j,
                                                                      const double    sigma_squared_f,
                                                                      const VectorXd& length_scales)
    {
        const int D = length_scales.rows();
        return [&]() -> VectorXd
        {
            VectorXd partial_derivative(D);
            for (int i = 0; i < D; ++ i)
            {
                partial_derivative(i) = (x_i(i) - x_j(i)) * (x_i(i) - x_j(i)) / (length_scales(i) * length_scales(i) * length_scales(i));
            }
            return CalculateArdSquaredExponentialKernel(x_i, x_j, sigma_squared_f, length_scales) * partial_derivative;
        }();
    }

    MatrixXd CalculateLargeK(const MatrixXd& X,
                             const double    sigma_squared_f,
                             const double    sigma_squared_n,
                             const VectorXd& length_scales)
    {
        const int      N = X.cols();
        const MatrixXd K = [&]()
        {
            MatrixXd K(N, N);
            for (int i = 0; i < N; ++ i)
            {
                for (int j = i; j < N; ++ j)
                {
                    const double value = CalculateArdSquaredExponentialKernel(X.col(i), X.col(j), sigma_squared_f, length_scales);
                    K(i, j) = value;
                    K(j, i) = value;
                }
            }
            return K;
        }();
        return K + sigma_squared_n * MatrixXd::Identity(N, N);
    }

    MatrixXd CalculateLargeKGradientSigmaSquaredF(const MatrixXd& X,
                                                  const double    sigma_squared_f,
                                                  const double    sigma_squared_n,
                                                  const VectorXd& length_scales)
    {
        const int N = X.cols();
        return [&]()
        {
            MatrixXd K_gradient_sigma_squared_f(N, N);
            for (int i = 0; i < N; ++ i)
            {
                for (int j = i; j < N; ++ j)
                {
                    const double ard_squared_exponential_kernel_gradient_sigma_squared_f = CalculateArdSquaredExponentialKernelGradientSigmaSquaredF(X.col(i), X.col(j), sigma_squared_f, length_scales);
                    K_gradient_sigma_squared_f(i, j) = ard_squared_exponential_kernel_gradient_sigma_squared_f;
                    K_gradient_sigma_squared_f(j, i) = ard_squared_exponential_kernel_gradient_sigma_squared_f;
                }
            }
            return K_gradient_sigma_squared_f;
        }();
    }

    MatrixXd CalculateLargeKGradientSigmaSquaredN(const MatrixXd& X,
                                                  const double    sigma_squared_f,
                                                  const double    sigma_squared_n,
                                                  const VectorXd& length_scales)
    {
        const int N = X.cols();
        return MatrixXd::Identity(N, N);
    }

    MatrixXd CalculateLargeKGradientLengthScaleI(const MatrixXd& X,
                                                 const double    sigma_squared_f,
                                                 const double    sigma_squared_n,
                                                 const VectorXd& length_scales,
                                                 const int       index)
    {
        const int N = X.cols();
        return [&]()
        {
            MatrixXd K_gradient_length_scale_i(N, N);
            for (int i = 0; i < N; ++ i)
            {
                for (int j = i; j < N; ++ j)
                {
                    const VectorXd ard_squared_exponential_kernel_gradient_length_scales = CalculateArdSquaredExponentialKernelGradientLengthScales(X.col(i), X.col(j), sigma_squared_f, length_scales);
                    K_gradient_length_scale_i(i, j) = ard_squared_exponential_kernel_gradient_length_scales(index);
                    K_gradient_length_scale_i(j, i) = ard_squared_exponential_kernel_gradient_length_scales(index);
                }
            }
            return K_gradient_length_scale_i;
        }();
    }

    VectorXd CalculateSmallK(const VectorXd& x,
                             const MatrixXd& X,
                             const double    sigma_squared_f,
                             const VectorXd& length_scales)
    {
        const int N = X.cols();
        return [&]()
        {
            VectorXd k(N);
            for (unsigned i = 0; i < N; ++ i)
            {
                k(i) = CalculateArdSquaredExponentialKernel(x, X.col(i), sigma_squared_f, length_scales);
            }
            return k;
        }();
    }

    double CalculateLogLikelihood(const MatrixXd& X,
                                  const VectorXd& y,
                                  const double    sigma_squared_f,
                                  const double    sigma_squared_n,
                                  const VectorXd& length_scales)
    {
        const int      N = X.cols();
        const MatrixXd K = CalculateLargeK(X, sigma_squared_f, sigma_squared_n, length_scales);

        const Eigen::FullPivLU<MatrixXd> lu(K);

        assert(lu.isInvertible());

        const MatrixXd K_inv = lu.inverse();
        const double   K_det = lu.determinant();

        // Equation 5.8 [Rasmuss and Williams 2006]
        const double term1 = - 0.5 * y.transpose() * K_inv * y;
        const double term2 = - 0.5 * std::log(K_det);
        const double term3 = - 0.5 * N * std::log(2.0 * M_PI);

        return term1 + term2 + term3;
    }

    VectorXd CalculateLogLikelihoodGradient(const MatrixXd& X,
                                            const VectorXd& y,
                                            const double    sigma_squared_f,
                                            const double    sigma_squared_n,
                                            const VectorXd& length_scales)
    {
        const int D = X.rows();

        const MatrixXd K     = CalculateLargeK(X, sigma_squared_f, sigma_squared_n, length_scales);
        const MatrixXd K_inv = K.inverse();

        const double log_likeliehood_gradient_sigma_squared_f = [&]()
        {
            // Equation 5.9 [Rasmuss and Williams 2006]
            const MatrixXd K_gradient_sigma_squared_f = CalculateLargeKGradientSigmaSquaredF(X, sigma_squared_f, sigma_squared_n, length_scales);
            const double term1 = + 0.5 * y.transpose() * K_inv * K_gradient_sigma_squared_f * K_inv * y;
            const double term2 = - 0.5 * (K_inv * K_gradient_sigma_squared_f).trace();
            return term1 + term2;
        }();

        const double log_likeliehood_gradient_sigma_squared_n = [&]()
        {
            // Equation 5.9 [Rasmuss and Williams 2006]
            const MatrixXd K_gradient_sigma_squared_n = CalculateLargeKGradientSigmaSquaredN(X, sigma_squared_f, sigma_squared_n, length_scales);
            const double term1 = + 0.5 * y.transpose() * K_inv * K_gradient_sigma_squared_n * K_inv * y;
            const double term2 = - 0.5 * (K_inv * K_gradient_sigma_squared_n).trace();
            return term1 + term2;
        }();

        const VectorXd log_likelihood_gradient_length_scales = [&]()
        {
            // Equation 5.9 [Rasmuss and Williams 2006]
            VectorXd log_likelihood_gradient_length_scales(D);
            for (int i = 0; i < D; ++ i)
            {
                const MatrixXd K_gradient_length_scale_i = CalculateLargeKGradientLengthScaleI(X, sigma_squared_f, sigma_squared_n, length_scales, i);
                const double term1 = + 0.5 * y.transpose() * K_inv * K_gradient_length_scale_i * K_inv * y;
                const double term2 = - 0.5 * (K_inv * K_gradient_length_scale_i).trace();
                log_likelihood_gradient_length_scales(i) = term1 + term2;
            }
            return log_likelihood_gradient_length_scales;
        }();

        return [&]()
        {
            VectorXd concat(D + 2);
            concat(0) = log_likeliehood_gradient_sigma_squared_f;
            concat(1) = log_likeliehood_gradient_sigma_squared_n;
            concat.segment(2, D) = log_likelihood_gradient_length_scales;
            return concat;
        }();
    }
}

namespace mathtoolbox
{
    GaussianProcessRegression::GaussianProcessRegression(const MatrixXd& X, const VectorXd& y) : X(X), y(y)
    {
        const int D = X.rows();

        SetHyperparameters(0.10, 1e-05, VectorXd::Constant(D, 0.10));
    }

    void GaussianProcessRegression::SetHyperparameters(double sigma_squared_f,
                                                       double sigma_squared_n,
                                                       const Eigen::VectorXd& length_scales)
    {
        this->sigma_squared_f = sigma_squared_f;
        this->sigma_squared_n = sigma_squared_n;
        this->length_scales   = length_scales;

        K     = CalculateLargeK(X, sigma_squared_f, sigma_squared_n, length_scales);
        K_inv = K.inverse();
    }

    void GaussianProcessRegression::PerformMaximumLikelihood(double sigma_squared_f_initial,
                                                             double sigma_squared_n_initial,
                                                             const Eigen::VectorXd& length_scales_initial)
    {
        const int D = X.rows();

        assert(length_scales.rows() == D);
        assert(length_scales_initial.rows() == D);

        const VectorXd x_initial = [&]()
        {
            VectorXd x(D + 2);
            x(0) = sigma_squared_f_initial;
            x(1) = sigma_squared_n_initial;
            x.segment(2, D) = length_scales_initial;
            return x;
        }();
        const VectorXd upper = VectorXd::Constant(D + 2, 1e+05);
        const VectorXd lower = VectorXd::Constant(D + 2, 1e-08);

        using Data = std::tuple<const MatrixXd&, const VectorXd&>;
        Data data(X, y);

        auto objective = [](const std::vector<double>& x, std::vector<double>& grad, void* data)
        {
            const double   sigma_squared_f = x[0];
            const double   sigma_squared_n = x[1];
            const VectorXd length_scales   = Eigen::Map<const VectorXd>(&x[2], x.size() - 2);

            const MatrixXd& X = std::get<0>(*static_cast<Data*>(data));
            const VectorXd& y = std::get<1>(*static_cast<Data*>(data));

            const double   log_likelihood          = CalculateLogLikelihood(X, y, sigma_squared_f, sigma_squared_n, length_scales);
            const VectorXd log_likelihood_gradient = CalculateLogLikelihoodGradient(X, y, sigma_squared_f, sigma_squared_n, length_scales);

            assert(!grad.empty());

            grad = std::vector<double>(log_likelihood_gradient.data(), log_likelihood_gradient.data() + log_likelihood_gradient.rows());

            return log_likelihood;
        };

        const VectorXd x_optimal = nloptutil::solve(x_initial,
                                                    upper,
                                                    lower,
                                                    objective,
                                                    nlopt::LD_LBFGS,
                                                    &data,
                                                    true,
                                                    1000,
                                                    1e-06,
                                                    1e-06,
                                                    true);

        sigma_squared_f = x_optimal[0];
        sigma_squared_n = x_optimal[1];
        length_scales           = Eigen::Map<const VectorXd>(&x_optimal[2], x_optimal.size() - 2);

        std::cout << "sigma_squared_f: " << sigma_squared_f << std::endl;
        std::cout << "sigma_squared_n: " << sigma_squared_n << std::endl;
        std::cout << "length_scales  : " << length_scales.transpose() << std::endl;

        K     = CalculateLargeK(X, sigma_squared_f, sigma_squared_n, length_scales);
        K_inv = K.inverse();
    }

    double GaussianProcessRegression::EstimateY(const VectorXd& x) const
    {
        const VectorXd k = CalculateSmallK(x, X, sigma_squared_f, length_scales);
        return k.transpose() * K_inv * y;
    }

    double GaussianProcessRegression::EstimateVariance(const VectorXd& x) const
    {
        const VectorXd k = CalculateSmallK(x, X, sigma_squared_f, length_scales);
        return sigma_squared_f + sigma_squared_n - k.transpose() * K_inv * k;
    }
}
