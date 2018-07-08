#include "gaussian-process-regression.hpp"
#include <tuple>
#include <Eigen/LU>
#include <nlopt-util.hpp>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace
{
    // Equation 5.1 [Rasmuss and Williams 2006]
    double CalculateArdSquaredExponentialKernel(const VectorXd& x_i, const VectorXd& x_j, const double s_f_squared, const VectorXd& l)
    {
        const int    D   = x_i.rows();
        const double sum = [&]()
        {
            double sum = 0.0;
            for (int i = 0; i < D; ++ i)
            {
                sum += (x_i(i) - x_j(i)) * (x_i(i) - x_j(i)) / (l(i) * l(i));
            }
            return sum;
        }();
        return s_f_squared * std::exp(- 0.5 * sum);
    }
    
    double CalculateArdSquaredExponentialKernelGradientSFSquared(const VectorXd& x_i, const VectorXd& x_j, const double s_f_squared, const VectorXd& l)
    {
        const int D = l.rows();
        return [&]()
        {
            double sum = 0.0;
            for (int i = 0; i < D; ++ i)
            {
                sum += (x_i(i) - x_j(i)) * (x_i(i) - x_j(i)) / (l(i) * l(i));
            }
            return std::exp(- 0.5 * sum);
        }();
    }
    
    VectorXd CalculateArdSquaredExponentialKernelGradientL(const VectorXd& x_i, const VectorXd& x_j, const double s_f_squared, const VectorXd& l)
    {
        const int D = l.rows();
        return [&]() -> VectorXd
        {
            VectorXd partial_derivative(D);
            for (int i = 0; i < D; ++ i)
            {
                partial_derivative(i) = (x_i(i) - x_j(i)) * (x_i(i) - x_j(i)) / (l(i) * l(i) * l(i));
            }
            return CalculateArdSquaredExponentialKernel(x_i, x_j, s_f_squared, l) * partial_derivative;
        }();
    }
    
    MatrixXd CalculateLargeK(const MatrixXd& X, const double s_f_squared, const double s_n_squared, const VectorXd& l)
    {
        const int      N = X.cols();
        const MatrixXd K = [&]()
        {
            MatrixXd K(N, N);
            for (int i = 0; i < N; ++ i)
            {
                for (int j = i; j < N; ++ j)
                {
                    const double value = CalculateArdSquaredExponentialKernel(X.col(i), X.col(j), s_f_squared, l);
                    K(i, j) = value;
                    K(j, i) = value;
                }
            }
            return K;
        }();
        return K + s_n_squared * MatrixXd::Identity(N, N);
    }
    
    MatrixXd CalculateLargeKGradientSFSquared(const MatrixXd& X, const double s_f_squared, const double s_n_squared, const VectorXd& l)
    {
        const int N = X.cols();
        return [&]()
        {
            MatrixXd K_gradient_s_f_squared(N, N);
            for (int i = 0; i < N; ++ i)
            {
                for (int j = i; j < N; ++ j)
                {
                    const double ard_squared_exponential_kernel_gradient_s_f_squared = CalculateArdSquaredExponentialKernelGradientSFSquared(X.col(i), X.col(j), s_f_squared, l);
                    K_gradient_s_f_squared(i, j) = ard_squared_exponential_kernel_gradient_s_f_squared;
                    K_gradient_s_f_squared(j, i) = ard_squared_exponential_kernel_gradient_s_f_squared;
                }
            }
            return K_gradient_s_f_squared;
        }();
    }
    
    MatrixXd CalculateLargeKGradientSNSquared(const MatrixXd& X, const double s_f_squared, const double s_n_squared, const VectorXd& l)
    {
        const int N = X.cols();
        return MatrixXd::Identity(N, N);
    }
    
    MatrixXd CalculateLargeKGradientLI(const MatrixXd& X, const double s_f_squared, const double s_n_squared, const VectorXd& l, const int index)
    {
        const int N = X.cols();
        return [&]()
        {
            MatrixXd K_gradient_l_i(N, N);
            for (int i = 0; i < N; ++ i)
            {
                for (int j = i; j < N; ++ j)
                {
                    const VectorXd ard_squared_exponential_kernel_gradient_l = CalculateArdSquaredExponentialKernelGradientL(X.col(i), X.col(j), s_f_squared, l);
                    K_gradient_l_i(i, j) = ard_squared_exponential_kernel_gradient_l(index);
                    K_gradient_l_i(j, i) = ard_squared_exponential_kernel_gradient_l(index);
                }
            }
            return K_gradient_l_i;
        }();
    }
    
    VectorXd CalculateSmallK(const VectorXd& x, const MatrixXd& X, const double s_f_squared, const VectorXd& l)
    {
        const int N = X.cols();
        return [&]()
        {
            VectorXd k(N);
            for (unsigned i = 0; i < N; ++ i)
            {
                k(i) = CalculateArdSquaredExponentialKernel(x, X.col(i), s_f_squared, l);
            }
            return k;
        }();
    }
    
    double CalculateLogLikelihood(const MatrixXd& X, const VectorXd& y, const double s_f_squared, const double s_n_squared, const VectorXd& l)
    {
        const int      N = X.cols();
        const MatrixXd K = CalculateLargeK(X, s_f_squared, s_n_squared, l);
        
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
    
    VectorXd CalculateLogLikelihoodGradient(const MatrixXd& X, const VectorXd& y, const double s_f_squared, const double s_n_squared, const VectorXd& l)
    {
        const int D = X.rows();
        
        const MatrixXd K     = CalculateLargeK(X, s_f_squared, s_n_squared, l);
        const MatrixXd K_inv = K.inverse();
        
        const double log_likeliehood_gradient_s_f_squared = [&]()
        {
            // Equation 5.9 [Rasmuss and Williams 2006]
            const MatrixXd K_gradient_s_f_squared = CalculateLargeKGradientSFSquared(X, s_f_squared, s_n_squared, l);
            const double term1 = + 0.5 * y.transpose() * K_inv * K_gradient_s_f_squared * K_inv * y;
            const double term2 = - 0.5 * (K_inv * K_gradient_s_f_squared).trace();
            return term1 + term2;
        }();
        
        const double log_likeliehood_gradient_s_n_squared = [&]()
        {
            // Equation 5.9 [Rasmuss and Williams 2006]
            const MatrixXd K_gradient_s_n_squared = CalculateLargeKGradientSNSquared(X, s_f_squared, s_n_squared, l);
            const double term1 = + 0.5 * y.transpose() * K_inv * K_gradient_s_n_squared * K_inv * y;
            const double term2 = - 0.5 * (K_inv * K_gradient_s_n_squared).trace();
            return term1 + term2;
        }();
        
        const VectorXd log_likelihood_gradient_l = [&]()
        {
            // Equation 5.9 [Rasmuss and Williams 2006]
            VectorXd log_likelihood_gradient_l(D);
            for (int i = 0; i < D; ++ i)
            {
                const MatrixXd K_gradient_l_i = CalculateLargeKGradientLI(X, s_f_squared, s_n_squared, l, i);
                const double term1 = + 0.5 * y.transpose() * K_inv * K_gradient_l_i * K_inv * y;
                const double term2 = - 0.5 * (K_inv * K_gradient_l_i).trace();
                log_likelihood_gradient_l(i) = term1 + term2;
            }
            return log_likelihood_gradient_l;
        }();
        
        return [&]()
        {
            VectorXd concat(D + 2);
            concat(0) = log_likeliehood_gradient_s_f_squared;
            concat(1) = log_likeliehood_gradient_s_n_squared;
            concat.segment(2, D) = log_likelihood_gradient_l;
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
    
    void GaussianProcessRegression::SetHyperparameters(double s_f_squared, double s_n_squared, const Eigen::VectorXd& l)
    {
        this->s_f_squared = s_f_squared;
        this->s_n_squared = s_n_squared;
        this->l           = l;
        
        K     = CalculateLargeK(X, s_f_squared, s_n_squared, l);
        K_inv = K.inverse();
    }
    
    void GaussianProcessRegression::PerformMaximumLikelihood(double s_f_squared_initial, double s_n_squared_initial, const Eigen::VectorXd& l_initial)
    {
        const int D = l.rows();
        
        const VectorXd x_initial = [&]()
        {
            VectorXd x(D + 2);
            x(0) = s_f_squared_initial;
            x(1) = s_n_squared_initial;
            x.segment(2, D) = l_initial;
            return x;
        }();
        const VectorXd upper = VectorXd::Constant(D + 2, 1e+05);
        const VectorXd lower = VectorXd::Constant(D + 2, 1e-08);
        
        using Data = std::tuple<const MatrixXd&, const VectorXd&>;
        Data data(X, y);
        
        auto objective = [](const std::vector<double>& x, std::vector<double>& grad, void* data)
        {
            const double   s_f_squared = x[0];
            const double   s_n_squared = x[1];
            const VectorXd l           = Eigen::Map<const VectorXd>(&x[2], x.size() - 2);
            
            const MatrixXd& X = std::get<0>(*static_cast<Data*>(data));
            const VectorXd& y = std::get<1>(*static_cast<Data*>(data));
            
            const double   log_likelihood          = CalculateLogLikelihood(X, y, s_f_squared, s_n_squared, l);
            const VectorXd log_likelihood_gradient = CalculateLogLikelihoodGradient(X, y, s_f_squared, s_n_squared, l);
            
            assert(!grad.empty());
            
            grad = std::vector<double>(log_likelihood_gradient.data(), log_likelihood_gradient.data() + log_likelihood_gradient.rows());
            
            return log_likelihood;
        };
        
        const VectorXd x_optimal = nloptutil::compute(x_initial, upper, lower, objective, &data, nlopt::LD_LBFGS, 1000, 1e-06, 1e-06, true, true);
        
        s_f_squared = x_optimal[0];
        s_n_squared = x_optimal[1];
        l           = Eigen::Map<const VectorXd>(&x_optimal[2], x_optimal.size() - 2);
        
        std::cout << "s_f_squared: " << s_f_squared << std::endl;
        std::cout << "s_n_squared: " << s_n_squared << std::endl;
        std::cout << "l:           " << l.transpose() << std::endl;
        
        K     = CalculateLargeK(X, s_f_squared, s_n_squared, l);
        K_inv = K.inverse();
    }
    
    double GaussianProcessRegression::EstimateY(const VectorXd& x) const
    {
        const VectorXd k = CalculateSmallK(x, X, s_f_squared, l);
        return k.transpose() * K_inv * y;
    }
    
    double GaussianProcessRegression::EstimateVariance(const VectorXd& x) const
    {
        const VectorXd k = CalculateSmallK(x, X, s_f_squared, l);
        return s_f_squared + s_n_squared - k.transpose() * K_inv * k;
    }
}
