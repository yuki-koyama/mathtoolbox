#include "rbf-interpolation.hpp"
#include <cmath>
#include <Eigen/LU>

using std::vector;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::FullPivLU;
using Eigen::Map;

namespace mathtoolbox
{
    extern inline VectorXd SolveLinearSystem(const MatrixXd& A, const VectorXd& y);
    
    RbfInterpolation::RbfInterpolation(RbfType rbf_type, double epsilon) :
    rbf_type(rbf_type),
    epsilon(epsilon)
    {
    }
    
    void RbfInterpolation::SetData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
    {
        assert(y.rows() == X.cols());
        this->X = X;
        this->y = y;
    }
    
    void RbfInterpolation::ComputeWeights(bool use_regularization, double lambda)
    {
        const int dim = y.rows();
        
        MatrixXd O = MatrixXd::Zero(dim, dim);
        
        for (int i = 0; i < dim; ++ i)
        {
            for (int j = 0; j < dim; ++ j)
            {
                O(i, j) = GetRbfValue(X.col(i), X.col(j));
            }
        }
        
        MatrixXd A;
        VectorXd b;
        if (use_regularization)
        {
            MatrixXd O2 = MatrixXd::Zero(dim * 2, dim);
            O2.block(0, 0, dim, dim) = O;
            const double coef = 0.5 * lambda;
            for (int i = 0; i < dim; ++ i)
            {
                O2(i + dim, i) = coef;
            }
            
            VectorXd y2 = VectorXd::Zero(dim * 2);
            y2.segment(0, dim) = y;

            A = O2.transpose() * O2;
            b = O2.transpose() * y2;
        }
        else
        {
            A = O;
            b = y;
        }
        
        w = SolveLinearSystem(A, b);
    }
    
    double RbfInterpolation::GetValue(const VectorXd& x) const
    {
        const int dim = w.rows();
        
        double result = 0.0;
        for (int i = 0; i < dim; ++ i)
        {
            result += w(i) * GetRbfValue(x, X.col(i));
        }
        
        return result;
    }
    
    double RbfInterpolation::GetRbfValue(double r) const
    {
        switch (rbf_type)
        {
            case RbfType::Gaussian:
            {
                return std::exp(- std::pow((epsilon * r), 2.0));
            }
            case RbfType::ThinPlateSpline:
            {
                const double result = r * r * std::log(r);
                if (isnan(result))
                {
                    return 0.0;
                }
                return result;
            }
            case RbfType::InverseQuadratic:
            {
                return 1.0 / (1.0 + std::pow((epsilon * r), 2.0));
            }
            case RbfType::BiharmonicSpline:
            {
                return r;
            }
        }
    }
    
    double RbfInterpolation::GetRbfValue(const VectorXd& xi, const VectorXd& xj) const
    {
        assert(xi.rows() == xj.rows());
        return GetRbfValue((xj - xi).norm());
    }
    
    inline VectorXd SolveLinearSystem(const MatrixXd& A, const VectorXd& y)
    {
        FullPivLU<MatrixXd> lu(A);
        return lu.solve(y);
    }
}
