#include "rbf-interpolation.hpp"
#include <cmath>
#include <Eigen/Core>
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
    
    void RbfInterpolation::Reset()
    {
        ys.clear();
        xs.clear();
        w.clear();
    }
    
    void RbfInterpolation::AddCenterPoint(double y, const vector<double>& x)
    {
        ys.push_back(y);
        xs.push_back(x);
    }
    
    void RbfInterpolation::ComputeWeights(bool use_regularization, double lambda)
    {
        assert(ys.size() == xs.size());
        
        const int dim = ys.size();
        
        MatrixXd O = MatrixXd::Zero(dim, dim);
        VectorXd y = Map<VectorXd>(&ys[0], dim);
        
        for (int i = 0; i < dim; ++ i)
        {
            for (int j = 0; j < dim; ++ j)
            {
                O(i, j) = GetRbfValue(xs[i], xs[j]);
            }
        }
        
        MatrixXd A;
        VectorXd b;
        if (use_regularization)
        {
            MatrixXd O2 = MatrixXd::Zero(dim * 2, dim);
            for (int i = 0; i < dim; ++ i)
            {
                for (int j = 0; j < dim; ++ j)
                {
                    O2(i, j) = O(i, j);
                }
            }
            const double coef = 0.5 * lambda;
            for (int i = 0; i < dim; ++ i)
            {
                O2(i + dim, i) = coef;
            }
            
            VectorXd y2 = VectorXd::Zero(dim * 2);
            for (int i = 0; i < dim; ++ i)
            {
                y2(i) = y(i);
            }
            
            A = O2.transpose() * O2;
            b = O2.transpose() * y2;
        }
        else
        {
            A = O;
            b = y;
        }
        
        const VectorXd x = SolveLinearSystem(A, b);
        assert(x.rows() == dim);
        
        w.resize(dim);
        for (int i = 0; i < dim; ++ i)
        {
            w[i] = x(i);
        }
    }
    
    double RbfInterpolation::GetValue(const vector<double>& x) const
    {
        assert(w.size() == xs.size());
        
        const int dim = w.size();
        
        double result = 0.0;
        for (int i = 0; i < dim; ++ i)
        {
            result += w[i] * GetRbfValue(x, xs[i]);
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
    
    double RbfInterpolation::GetRbfValue(const vector<double>& xi, const vector<double>& xj) const
    {
        assert(xi.size() == xj.size());
        
        const VectorXd xiVec = Map<const VectorXd>(&xi[0], xi.size());
        const VectorXd xjVec = Map<const VectorXd>(&xj[0], xj.size());
        
        return GetRbfValue((xjVec - xiVec).norm());
    }
    
    inline VectorXd SolveLinearSystem(const MatrixXd& A, const VectorXd& y)
    {
        FullPivLU<MatrixXd> lu(A);
        return lu.solve(y);
    }
}
