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

    const VectorXd diff      = x_a - x_b;
    const double   r_squared = diff.transpose() * length_scales.array().square().inverse().matrix().asDiagonal() * diff;

    return sigma_squared_f * std::exp(-0.5 * r_squared);
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
        const VectorXd diff = x_a - x_b;
        const double   r_squared =
            diff.transpose() * length_scales.array().square().inverse().matrix().asDiagonal() * diff;

        return std::exp(-0.5 * r_squared);
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
        const VectorXd diff = x_a - x_b;
        const double   r_squared =
            diff.transpose() * length_scales.array().square().inverse().matrix().asDiagonal() * diff;

        return std::exp(-0.5 * r_squared);
    }
    else
    {
        const int    i = index - 1;
        const double k = GetArdSquaredExpKernel(x_a, x_b, theta);
        const double r = x_a(i) - x_b(i);

        return k * (r * r) / (length_scales(i) * length_scales(i) * length_scales(i));
    }
}

Eigen::VectorXd mathtoolbox::GetArdSquaredExpKernelFirstArgDerivative(const Eigen::VectorXd& x_a,
                                                                      const Eigen::VectorXd& x_b,
                                                                      const Eigen::VectorXd& theta)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == theta.size() - 1);

    const int       dim           = x_a.size();
    const VectorXd& length_scales = theta.segment(1, dim);
    const double    k             = GetArdSquaredExpKernel(x_a, x_b, theta);

    return -2.0 * k * length_scales.array().square().inverse().matrix().asDiagonal() * (x_a - x_b);
}

double mathtoolbox::GetArdMatern52Kernel(const VectorXd& x_a, const VectorXd& x_b, const VectorXd& theta)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == theta.size() - 1);

    const int       dim             = x_a.size();
    const double&   sigma_squared_f = theta[0];
    const VectorXd& length_scales   = theta.segment(1, dim);

    const VectorXd diff      = x_a - x_b;
    const double   r_squared = diff.transpose() * length_scales.array().square().inverse().matrix().asDiagonal() * diff;

    const double sqrt_of_5_r_squared = std::sqrt(5.0 * r_squared);
    const double scale_term          = 1.0 + sqrt_of_5_r_squared + (5.0 / 3.0) * r_squared;
    const double exp_term            = std::exp(-sqrt_of_5_r_squared);

    return sigma_squared_f * scale_term * exp_term;
}

VectorXd
mathtoolbox::GetArdMatern52KernelThetaDerivative(const VectorXd& x_a, const VectorXd& x_b, const VectorXd& theta)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == theta.size() - 1);

    const int       dim             = x_a.size();
    const double&   sigma_squared_f = theta[0];
    const VectorXd& length_scales   = theta.segment(1, dim);

    const VectorXd diff      = x_a - x_b;
    const double   r_squared = diff.transpose() * length_scales.array().square().inverse().matrix().asDiagonal() * diff;

    const double sqrt_of_5_r_squared = std::sqrt(5.0 * r_squared);
    const double scale_term          = 1.0 + sqrt_of_5_r_squared + (5.0 / 3.0) * r_squared;
    const double exp_term            = std::exp(-sqrt_of_5_r_squared);

    VectorXd derivative(theta.size());

    derivative(0) = scale_term * exp_term;

    const auto diff_squared        = diff.array().square();
    const auto length_scales_cubed = length_scales.array().cube();

    derivative.segment(1, dim) = (5.0 / 3.0) * sigma_squared_f * exp_term * (1.0 + sqrt_of_5_r_squared) * diff_squared *
                                 length_scales_cubed.inverse();

    return derivative;
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

    const VectorXd diff      = x_a - x_b;
    const double   r_squared = diff.transpose() * length_scales.array().square().inverse().matrix().asDiagonal() * diff;

    const double sqrt_of_5_r_squared = std::sqrt(5.0 * r_squared);
    const double scale_term          = 1.0 + sqrt_of_5_r_squared + (5.0 / 3.0) * r_squared;
    const double exp_term            = std::exp(-sqrt_of_5_r_squared);

    if (index == 0)
    {
        return scale_term * exp_term;
    }
    else
    {
        const int i = index - 1;

        const double diff          = x_a(i) - x_b(i);
        const double diff_squared  = diff * diff;
        const double theta_i_cubed = length_scales(i) * length_scales(i) * length_scales(i);

        return (5.0 / 3.0) * sigma_squared_f * diff_squared * exp_term * (1.0 + sqrt_of_5_r_squared) / theta_i_cubed;
    }
}

VectorXd
mathtoolbox::GetArdMatern52KernelFirstArgDerivative(const VectorXd& x_a, const VectorXd& x_b, const VectorXd& theta)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == theta.size() - 1);

    const int       dim             = x_a.size();
    const double&   sigma_squared_f = theta[0];
    const VectorXd& length_scales   = theta.segment(1, dim);

    const VectorXd diff      = x_a - x_b;
    const double   r_squared = diff.transpose() * length_scales.array().square().inverse().matrix().asDiagonal() * diff;

    const double sqrt_of_5_r_squared = std::sqrt(5.0 * r_squared);
    const double scale_term          = 1.0 + sqrt_of_5_r_squared + (5.0 / 3.0) * r_squared;
    const double exp_term            = std::exp(-sqrt_of_5_r_squared);

    // When x_a is very similar to x_b, the following calculation becomes numerically unstable. To avoid this, here the
    // derivative is simply approximated to a zero vector.
    if (sqrt_of_5_r_squared < 1e-30)
    {
        return Eigen::VectorXd::Zero(dim);
    }

    const VectorXd r_squared_first_arg_derivative =
        2.0 * length_scales.array().square().inverse().matrix().asDiagonal() * diff;
    const VectorXd sqrt_of_5_r_squared_first_arg_derivative =
        0.5 * std::sqrt(5.0 / r_squared) * r_squared_first_arg_derivative;
    const auto exp_term_first_arg_derivative = -sqrt_of_5_r_squared_first_arg_derivative * exp_term;
    const auto scale_term_first_arg_derivative =
        sqrt_of_5_r_squared_first_arg_derivative + (5.0 / 3.0) * r_squared_first_arg_derivative;

    return sigma_squared_f * (scale_term_first_arg_derivative * exp_term + scale_term * exp_term_first_arg_derivative);
}
