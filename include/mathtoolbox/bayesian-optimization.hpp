#ifndef MATHTOOLBOX_BAYESIAN_OPTIMIZATION_HPP
#define MATHTOOLBOX_BAYESIAN_OPTIMIZATION_HPP

#include <Eigen/Core>

namespace mathtoolbox
{
    namespace optimization
    {
        enum class KernelType
        {
            ArdSquaredExp,
            ArdMatern52
        };

        // Class-style API
        class BayesianOptimizer
        {
        public:
            BayesianOptimizer(const std::function<double(const Eigen::VectorXd&)>& f,
                              const Eigen::VectorXd&                               lower_bound,
                              const Eigen::VectorXd&                               upper_bound,
                              const KernelType& kernel_type = KernelType::ArdMatern52);

            /// \brief Perform a single step of the Bayesian optimization algorithm
            /// \return Newly observed data point
            Eigen::VectorXd Step();

            /// \brief Calculate the function value
            /// \return Evaluated function value
            double EvaluatePoint(const Eigen::VectorXd& x) const { return m_f(x); }

            /// \brief Retrieve the optimizer found so far
            /// \return Optimizer found so far
            Eigen::VectorXd GetCurrentOptimizer() const;

        private:
            const std::function<double(const Eigen::VectorXd&)> m_f;

            const Eigen::VectorXd m_lower_bound;
            const Eigen::VectorXd m_upper_bound;

            const KernelType m_kernel_type;

            Eigen::MatrixXd m_X;
            Eigen::VectorXd m_y;
        };

        // Function-style API
        void RunBayesianOptimization(const std::function<double(const Eigen::VectorXd&)>& f,
                                     const Eigen::VectorXd&                               lower_bound,
                                     const Eigen::VectorXd&                               upper_bound,
                                     const unsigned int                                   max_num_iterations,
                                     const KernelType&                                    kernel_type,
                                     Eigen::VectorXd&                                     x_star,
                                     unsigned int&                                        num_iterations);
    } // namespace optimization
} // namespace mathtoolbox

#endif // MATHTOOLBOX_BAYESIAN_OPTIMIZATION_HPP
