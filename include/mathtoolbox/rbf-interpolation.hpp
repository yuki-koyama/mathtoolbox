#ifndef MATHTOOLBOX_RBF_INTERPOLATION_HPP
#define MATHTOOLBOX_RBF_INTERPOLATION_HPP

#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <vector>

namespace mathtoolbox
{
    class AbstractRbfKernel
    {
    public:
        AbstractRbfKernel() {}

        virtual double EvaluateValue(const double r) const = 0;
    };

    class GaussianRbfKernel final : public AbstractRbfKernel
    {
    public:
        GaussianRbfKernel(const double theta = 1.0) : m_theta(theta) {}

        double EvaluateValue(const double r) const override
        {
            assert(r >= 0.0);
            return std::exp(-m_theta * r * r);
        }

        const double m_theta;
    };

    class ThinPlateSplineRbfKernel final : public AbstractRbfKernel
    {
    public:
        ThinPlateSplineRbfKernel() {}

        double EvaluateValue(const double r) const override
        {
            assert(r >= 0.0);
            const double value = r * r * std::log(r);
            return std::isnan(value) ? 0.0 : value;
        }
    };

    class LinearRbfKernel final : AbstractRbfKernel
    {
    public:
        LinearRbfKernel() {}

        double EvaluateValue(const double r) const override { return std::abs(r); }
    };

    class InverseQuadraticRbfKernel final : public AbstractRbfKernel
    {
    public:
        InverseQuadraticRbfKernel(const double theta = 1.0) : m_theta(theta) {}

        double EvaluateValue(const double r) const override { return 1.0 / std::sqrt(r * r + m_theta * m_theta); }

        const double m_theta;
    };

    class RbfInterpolator
    {
    public:
        RbfInterpolator(
            const std::shared_ptr<AbstractRbfKernel> rbf_kernel = std::make_shared<ThinPlateSplineRbfKernel>());

        // API
        void   SetData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
        void   ComputeWeights(bool use_regularization = false, double lambda = 0.001);
        double GetValue(const Eigen::VectorXd& x) const;

        // Getter methods
        const Eigen::VectorXd& GetY() const { return m_y; }
        const Eigen::MatrixXd& GetX() const { return m_X; }
        const Eigen::VectorXd& GetW() const { return m_w; }

    private:
        // RBF kernel
        const std::shared_ptr<AbstractRbfKernel> m_rbf_kernel;

        // Data points
        Eigen::MatrixXd m_X;
        Eigen::VectorXd m_y;

        // Weights
        Eigen::VectorXd m_w;

        // Returns f(||xj - xi||)
        double GetRbfValue(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj) const;
    };
} // namespace mathtoolbox

#endif // MATHTOOLBOX_RBF_INTERPOLATION_HPP
