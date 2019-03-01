#ifndef RBF_INTERPOLATION_HPP
#define RBF_INTERPOLATION_HPP

#include <vector>
#include <Eigen/Core>

namespace mathtoolbox
{
    enum class RbfType
    {
        Gaussian,         // f(r) = exp(-(epsilon * r)^2)
        ThinPlateSpline,  // f(r) = (r^2) * log(r)
        InverseQuadratic, // f(r) = (1 + (epsilon * r)^2)^(-1)
        Linear,           // f(r) = r
    };

    class RbfInterpolation
    {
    public:
        RbfInterpolation(RbfType rbf_type = RbfType::ThinPlateSpline, double epsilon = 2.0);

        // API
        void   SetData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
        void   ComputeWeights(bool use_regularization = false, double lambda = 0.001);
        double GetValue(const Eigen::VectorXd& x) const;

        // Getter methods
        const Eigen::VectorXd& GetY() const { return y; }
        const Eigen::MatrixXd& GetX() const { return X; }
        const Eigen::VectorXd& GetW() const { return w; }

    private:

        // Function type
        RbfType rbf_type;

        // A control parameter used in some kernel functions
        double epsilon;

        // Data points
        Eigen::MatrixXd X;
        Eigen::VectorXd y;

        // Weights
        Eigen::VectorXd w;

        // Returns f(r)
        double GetRbfValue(double r) const;

        // Returns f(||xj - xi||)
        double GetRbfValue(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj) const;
    };
}

#endif // RBF_INTERPOLATION_HPP
