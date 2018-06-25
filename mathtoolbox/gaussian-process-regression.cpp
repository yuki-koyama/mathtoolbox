#include "gaussian-process-regression.hpp"
#include <tuple>
#include <Eigen/LU>
#include <nlopt-util.hpp>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace
{
    // Equation 5.1
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
    
    MatrixXd CalculateLargeK(const MatrixXd& X, const double s_f_squared, const double s_n_squared, const VectorXd& l)
    {
        const int      N = X.cols();
        const MatrixXd K = [&]()
        {
            MatrixXd K(N, N);
            for (unsigned i = 0; i < N; ++ i)
            {
                for (unsigned j = i; j < N; ++ j)
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
    
    VectorXd CalculateSmallK(const VectorXd& x, const MatrixXd& X, const double s_f_squared, const VectorXd& l)
    {
        const int      N = X.cols();
        const VectorXd k = [&]()
        {
            VectorXd k(N);
            for (unsigned i = 0; i < N; ++ i)
            {
                k(i) = CalculateArdSquaredExponentialKernel(x, X.col(i), s_f_squared, l);
            }
            return k;
        }();
        
        return k;
    }
    
    // Equation 5.8
    double CalculateLogLikelihood(const MatrixXd& X, const VectorXd& y, const double s_f_squared, const double s_n_squared, const VectorXd& l)
    {
        const int N = X.cols();
        
        const MatrixXd K     = CalculateLargeK(X, s_f_squared, s_n_squared, l);
        const MatrixXd K_inv = K.inverse();
        
        const double term1 = - 0.5 * y.transpose() * K_inv * y;
        const double term2 = - 0.5 * std::log(K.determinant());
        const double term3 = - 0.5 * N * std::log(2.0 * M_PI);
        
        return term1 + term2 + term3;
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
        const VectorXd upper = [&D]()
        {
            VectorXd x(D + 2);
            x(0) = 1e+05;
            x(1) = 1e+05;
            x.segment(2, D) = VectorXd::Constant(D, 1e+05);
            return x;
        }();
        const VectorXd lower = [&D]()
        {
            VectorXd x(D + 2);
            x(0) = 1e-02;
            x(1) = 1e-08;
            x.segment(2, D) = VectorXd::Constant(D, 1e-02);
            return x;
        }();
        
        typedef std::tuple<const MatrixXd&, const VectorXd&> Data;
        Data data(X, y);
        
        auto objective = [](const std::vector<double>& x, std::vector<double>& grad, void* data)
        {
            const double   s_f_squared = x[0];
            const double   s_n_squared = x[1];
            const VectorXd l           = Eigen::Map<const VectorXd>(&x[2], x.size() - 2);
            
            const MatrixXd& X = std::get<0>(*static_cast<Data*>(data));
            const VectorXd& y = std::get<1>(*static_cast<Data*>(data));
            
            const double log_likelihood = CalculateLogLikelihood(X, y, s_f_squared, s_n_squared, l);

            return log_likelihood;
        };
        
        const VectorXd x_optimal = nloptutil::compute(x_initial, upper, lower, objective, &data, nlopt::LN_BOBYQA, 1000, 1e-06, 1e-06, true);
        
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
    
    double GaussianProcessRegression::EstimateS(const VectorXd& x) const
    {
        const VectorXd k = CalculateSmallK(x, X, s_f_squared, l);
        return std::sqrt(s_f_squared + s_n_squared - k.transpose() * K_inv * k);
    }
}
