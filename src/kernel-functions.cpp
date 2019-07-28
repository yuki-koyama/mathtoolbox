#include <cmath>
#include <mathtoolbox/kernel-functions.hpp>

using Eigen::VectorXd;

double mathtoolbox::GetArdSquaredExpKernel(const VectorXd& x_a, const VectorXd& x_b, const VectorXd& theta)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == theta.size() - 1);

    const int       dim             = x_a.size();
    const double&   sigma_squared_f = theta[0];
    const VectorXd& length_scales   = theta.segment(1, dim);

    const double sum = [&]() {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            const double r = x_a(i) - x_b(i);
            sum += (r * r) / (length_scales(i) * length_scales(i));
        }
        return sum;
    }();

    return sigma_squared_f * std::exp(-0.5 * sum);
}

VectorXd
mathtoolbox::GetArdSquaredExpKernelThetaDerivative(const VectorXd& x_a, const VectorXd& x_b, const VectorXd& theta)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == theta.size() - 1);

    const int       dim           = x_a.size();
    const VectorXd& length_scales = theta.segment(1, dim);

    VectorXd derivative(theta.size());

    derivative(0) = [&]() {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            const double r = x_a(i) - x_b(i);
            sum += (r * r) / (length_scales(i) * length_scales(i));
        }
        return std::exp(-0.5 * sum);
    }();

    const double k = GetArdSquaredExpKernel(x_a, x_b, theta);

    for (int i = 0; i < dim; ++i)
    {
        const double r    = x_a(i) - x_b(i);
        derivative(i + 1) = k * (r * r) / (length_scales(i) * length_scales(i) * length_scales(i));
    }

    return derivative;
}

double mathtoolbox::GetArdSquaredExpKernelThetaIDerivative(const VectorXd& x_a,
                                                           const VectorXd& x_b,
                                                           const VectorXd& theta,
                                                           const int       index)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == theta.size() - 1);

    const int       dim           = x_a.size();
    const VectorXd& length_scales = theta.segment(1, dim);

    if (index == 0)
    {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            const double r = x_a(i) - x_b(i);
            sum += (r * r) / (length_scales(i) * length_scales(i));
        }
        return std::exp(-0.5 * sum);
    }
    else
    {
        const int    i = index - 1;
        const double k = GetArdSquaredExpKernel(x_a, x_b, theta);
        const double r = x_a(i) - x_b(i);

        return k * (r * r) / (length_scales(i) * length_scales(i) * length_scales(i));
    }
}

double mathtoolbox::GetArdMatern52Kernel(const VectorXd& x_a, const VectorXd& x_b, const VectorXd& theta)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == theta.size() - 1);

    const int       dim             = x_a.size();
    const double&   sigma_squared_f = theta[0];
    const VectorXd& length_scales   = theta.segment(1, dim);

    const double r_squared = [&]() {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            const double r = x_a(i) - x_b(i);
            sum += (r * r) / (length_scales(i) * length_scales(i));
        }
        return sum;
    }();
    const double sqrt_of_5_r_squared = std::sqrt(5.0 * r_squared);
    const double scale_term          = 1.0 + sqrt_of_5_r_squared + (5.0 / 3.0) * r_squared;
    const double exp_term            = std::exp(-sqrt_of_5_r_squared);

    return sigma_squared_f * scale_term * exp_term;
}

VectorXd
mathtoolbox::GetArdMatern52KernelThetaDerivative(const VectorXd& x_a, const VectorXd& x_b, const VectorXd& theta)
{
}

double mathtoolbox::GetArdMatern52KernelThetaIDerivative(const VectorXd& x_a,
                                                         const VectorXd& x_b,
                                                         const VectorXd& theta,
                                                         const int       index)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == theta.size() - 1);

    const int       dim             = x_a.size();
    const double&   sigma_squared_f = theta[0];
    const VectorXd& length_scales   = theta.segment(1, dim);

    const double r_squared = [&]() {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            const double r = x_a(i) - x_b(i);
            sum += (r * r) / (length_scales(i) * length_scales(i));
        }
        return sum;
    }();
    const double sqrt_of_5_r_squared = std::sqrt(5.0 * r_squared);
    const double scale_term          = 1.0 + sqrt_of_5_r_squared + (5.0 / 3.0) * r_squared;
    const double exp_term            = std::exp(-sqrt_of_5_r_squared);

    if (index == 0) { return scale_term * exp_term; }
    else
    {
        const int i = index - 1;

        const double diff          = x_a(i) - x_b(i);
        const double diff_squared  = diff * diff;
        const double theta_i_cubed = length_scales(i) * length_scales(i) * length_scales(i);

        return (5.0 / 3.0) * sigma_squared_f * diff_squared * exp_term * (1.0 + sqrt_of_5_r_squared) / theta_i_cubed;
    }
}
