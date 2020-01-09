#ifndef MATHTOOLBOX_DATA_NORMALIZATION_HPP
#define MATHTOOLBOX_DATA_NORMALIZATION_HPP

#include <Eigen/Core>

namespace mathtoolbox
{
    class DataNormalizer
    {
    public:
        DataNormalizer(const Eigen::MatrixXd& data_points) : m_original_data_points(data_points)
        {
            const int num_dims = m_original_data_points.rows();

            // TODO
            m_mean = Eigen::VectorXd::Ones(num_dims);
            m_var  = Eigen::VectorXd::Ones(num_dims);

            // TODO
            m_normalized_data_points = m_original_data_points;
        }

        const Eigen::MatrixXd& GetNormalizedDataPoints() const { return m_normalized_data_points; }

        Eigen::MatrixXd Normalize(const Eigen::MatrixXd& data_points) const
        {
            // TODO
            return data_points;
        }

        Eigen::MatrixXd Denormalize(const Eigen::MatrixXd& normalized_data_points) const
        {
            // TODO
            return normalized_data_points;
        }

    private:
        const Eigen::MatrixXd m_original_data_points;

        Eigen::MatrixXd m_normalized_data_points;
        Eigen::VectorXd m_mean;
        Eigen::VectorXd m_var;
    };
} // namespace mathtoolbox

#endif // MATHTOOLBOX_DATA_NORMALIZATION_HPP
