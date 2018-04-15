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
    extern VectorXd solveLinearSystem(const MatrixXd& A, const VectorXd& y);
    
    Interpolator::Interpolator(FunctionType functionType, const double epsilon) :
    functionType(functionType),
    epsilon(epsilon)
    {
    }
    
    void Interpolator::reset()
    {
        ys.clear();
        xs.clear();
        w.clear();
    }
    
    void Interpolator::addCenterPoint(const double y, const vector<double>& x)
    {
        ys.push_back(y);
        xs.push_back(x);
    }
    
    void Interpolator::computeWeights(const bool useRegularization, const double lambda)
    {
        assert(ys.size() == xs.size());
        
        const int dim = ys.size();
        
        MatrixXd O = MatrixXd::Zero(dim, dim);
        VectorXd y = Map<VectorXd>(&ys[0], dim);
        
        for (int i = 0; i < dim; ++ i)
        {
            for (int j = 0; j < dim; ++ j)
            {
                O(i, j) = getRbfValue(xs[i], xs[j]);
            }
        }
        
        MatrixXd A;
        VectorXd b;
        if (useRegularization)
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
        
        const VectorXd x = solveLinearSystem(A, b);
        assert(x.rows() == dim);
        
        w.resize(dim);
        for (int i = 0; i < dim; ++ i)
        {
            w[i] = x(i);
        }
    }
    
    double Interpolator::getInterpolatedValue(const vector<double>& x) const
    {
        assert(w.size() == xs.size());
        
        const int dim = w.size();
        
        double result = 0.0;
        for (int i = 0; i < dim; ++ i)
        {
            result += w[i] * getRbfValue(x, xs[i]);
        }
        
        return result;
    }
    
    double Interpolator::getRbfValue(const double r) const
    {
        double result;
        switch (functionType)
        {
            case FunctionType::Gaussian:
                result = exp(- pow((epsilon * r), 2.0));
                break;
            case FunctionType::ThinPlateSpline:
                result = r * r * log(r);
                if (isnan(result))
                {
                    result = 0.0;
                }
                break;
            case FunctionType::InverseQuadratic:
                result = 1.0 / (1.0 + pow((epsilon * r), 2.0));
                break;
            case FunctionType::BiharmonicSpline:
                result = r;
                break;
            default:
                break;
        }
        return result;
    }
    
    double Interpolator::getRbfValue(const vector<double>& xi, const vector<double>& xj) const
    {
        assert (xi.size() == xj.size());
        
        const VectorXd xiVec = Map<const VectorXd>(&xi[0], xi.size());
        const VectorXd xjVec = Map<const VectorXd>(&xj[0], xj.size());
        
        return getRbfValue((xjVec - xiVec).norm());
    }
    
    VectorXd solveLinearSystem(const MatrixXd& A, const VectorXd& y)
    {
        FullPivLU<MatrixXd> lu(A);
        return lu.solve(y);
    }
}
