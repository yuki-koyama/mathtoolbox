#ifndef MATHTOOLBOX_BAYESIAN_OPTIMIZATION_HPP
#define MATHTOOLBOX_BAYESIAN_OPTIMIZATION_HPP

#include <Eigen/Core>
#include <memory>
#include <utility>

namespace mathtoolbox
{
    class GaussianProcessRegressor;

    namespace optimization
    {
        enum class KernelType
        {
            ArdSquaredExp,
            ArdMatern52
        };

        /// \brief An optimizer class for managing Bayesian optimization iterations
        ///
        /// \details This optimizer solves a maximization problem with a black-box function.
        class BayesianOptimizer
        {
        public:
            BayesianOptimizer(const std::function<double(const Eigen::VectorXd&)>& f,
                              const Eigen::VectorXd&                               lower_bound,
                              const Eigen::VectorXd&                               upper_bound,
                              const KernelType& kernel_type = KernelType::ArdMatern52);

            /// \brief Perform a single step of the Bayesian optimization algorithm
            /// \return Newly observed data point
            std::pair<Eigen::VectorXd, double> Step();

            /// \brief Calculate the function value
            /// \return Evaluated function value
            double EvaluatePoint(const Eigen::VectorXd& x) const;

            /// \brief Predict the mean value
            /// \return Predicted mean value
            double PredictMean(const Eigen::VectorXd& x) const;

            /// \brief Predict the standard deviation value
            /// \return Predicted standard deviation value
            double PredictStdev(const Eigen::VectorXd& x) const;

            /// \brief Calculate the acquisition function value at the point
            /// \return Acquisition function value at the point
            double CalcAcquisitionValue(const Eigen::VectorXd& x) const;

            /// \brief Retrieve the optimizer found so far
            /// \return Optimizer found so far
            Eigen::VectorXd GetCurrentOptimizer() const;

            /// \brief Get the observed data points and their values
            std::pair<Eigen::MatrixXd, Eigen::VectorXd> GetData() const;

        private:
            const std::function<double(const Eigen::VectorXd&)> m_f;

            const Eigen::VectorXd m_lower_bound;
            const Eigen::VectorXd m_upper_bound;

            const KernelType m_kernel_type;

            Eigen::MatrixXd m_X;
            Eigen::VectorXd m_y;

            std::shared_ptr<GaussianProcessRegressor> m_regressor;

            void AddDataEntry(const Eigen::VectorXd& x_new, const double y_new);
            void ConstructSurrogateFunction();
        };
    } // namespace optimization
} // namespace mathtoolbox

#endif // MATHTOOLBOX_BAYESIAN_OPTIMIZATION_HPP
