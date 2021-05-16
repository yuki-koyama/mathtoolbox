#ifndef MATHTOOLBOX_RBF_INTERPOLATION_HPP
#define MATHTOOLBOX_RBF_INTERPOLATION_HPP

#include <Eigen/Core>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

namespace mathtoolbox
{
    /// \brief An abstract class for helping define custom kernel functor classes
    class AbstractRbfKernel
    {
    public:
        AbstractRbfKernel() {}
        virtual ~AbstractRbfKernel(){};

        /// \brief Perform RBF evaluation
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

    /// \details This kernel is also known as the polyharmonic kernel with k = 1
    class LinearRbfKernel final : AbstractRbfKernel
    {
    public:
        LinearRbfKernel() {}

        double operator()(const double r) const override
        {
            assert(r >= 0.0);
            return r;
        }
    };

    /// \details This kernel is also known as the polyharmonic kernel with k = 2
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

    /// \details This kernel is also known as the polyharmonic kernel with k = 3
    class CubicRbfKernel final : public AbstractRbfKernel
    {
    public:
        CubicRbfKernel() {}

        double operator()(const double r) const override
        {
            assert(r >= 0.0);
            return r * r * r;
        }
    };

    class RbfInterpolator
    {
    public:
        RbfInterpolator(const std::function<double(const double)>& rbf_kernel          = ThinPlateSplineRbfKernel(),
                        const bool                                 use_polynomial_term = true);

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
        /// \brief The RBF kernel
        const std::function<double(double)> m_rbf_kernel;

        /// \brief The polynomial term setting
        const bool m_use_polynomial_term;

        /// \brief Data locations
        Eigen::MatrixXd m_X;

        /// \brief Data values
        Eigen::VectorXd m_y;

        /// \brief Weights for the RBF kernel values
        Eigen::VectorXd m_w;

        /// \brief Weights for the polynomial terms
        Eigen::VectorXd m_v;
    };
} // namespace mathtoolbox

#endif // MATHTOOLBOX_RBF_INTERPOLATION_HPP
