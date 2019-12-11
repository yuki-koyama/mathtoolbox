#ifndef MATHTOOLBOX_RBF_INTERPOLATION_HPP
#define MATHTOOLBOX_RBF_INTERPOLATION_HPP

#include <Eigen/Core>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

namespace mathtoolbox
{
    class AbstractRbfKernel
    {
    public:
        AbstractRbfKernel() {}
        virtual ~AbstractRbfKernel(){};

        virtual double operator()(const double r) const = 0;
    };

    class GaussianRbfKernel final : public AbstractRbfKernel
    {
    public:
        GaussianRbfKernel(const double theta = 1.0) : m_theta(theta) {}

        double operator()(const double r) const override
        {
            assert(r >= 0.0);
            return std::exp(-m_theta * r * r);
        }

    private:
        const double m_theta;
    };

    class ThinPlateSplineRbfKernel final : public AbstractRbfKernel
    {
    public:
        ThinPlateSplineRbfKernel() {}

        double operator()(const double r) const override
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

        double operator()(const double r) const override { return std::abs(r); }
    };

    class InverseQuadraticRbfKernel final : public AbstractRbfKernel
    {
    public:
        InverseQuadraticRbfKernel(const double theta = 1.0) : m_theta(theta) {}

        double operator()(const double r) const override { return 1.0 / std::sqrt(r * r + m_theta * m_theta); }

    private:
        const double m_theta;
    };

    class RbfInterpolator
    {
    public:
        RbfInterpolator(const std::function<double(const double)>& rbf_kernel = ThinPlateSplineRbfKernel());

        /// \brief Set data points and their values
        void SetData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

        /// \brief Calculate the interpolation weights
        ///
        /// \details This method should be called after setting the data
        void CalcWeights(const bool use_regularization = false, const double lambda = 0.001);

        /// \brief Calculate the interpolatetd value at the specified data point
        ///
        /// \details This method should be called after calculating the weights
        double CalcValue(const Eigen::VectorXd& x) const;

    private:
        // RBF kernel
        const std::function<double(double)> m_rbf_kernel;

        // Data points
        Eigen::MatrixXd m_X;
        Eigen::VectorXd m_y;

        // Weights
        Eigen::VectorXd m_w;

        // Returns f(||xj - xi||)
        double CalcRbfValue(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj) const;
    };
} // namespace mathtoolbox

#endif // MATHTOOLBOX_RBF_INTERPOLATION_HPP
