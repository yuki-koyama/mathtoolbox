#ifndef MATHTOOLBOX_DATA_NORMALIZATION_HPP
#define MATHTOOLBOX_DATA_NORMALIZATION_HPP

#include <Eigen/Core>

namespace mathtoolbox
{
    class DataNormalizer
    {
    public:
        /// \param data_points An n-by-m matrix representing m data points lying in an n-dimensional space.
        DataNormalizer(const Eigen::MatrixXd& data_points) : m_original_data_points(data_points)
        {
            const int num_dims        = m_original_data_points.rows();
            const int num_data_points = m_original_data_points.cols();

            assert(num_dims > 0);
            assert(num_data_points > 0);

            m_mean  = Eigen::VectorXd(num_dims);
            m_stdev = Eigen::VectorXd(num_dims);

            for (int dim = 0; dim < num_dims; ++dim)
            {
                const Eigen::VectorXd one_dim_data = m_original_data_points.row(dim);

                m_mean(dim) = one_dim_data.mean();
                m_stdev(dim) =
                    std::sqrt((one_dim_data - Eigen::VectorXd::Constant(num_data_points, m_mean(dim))).squaredNorm() /
                              static_cast<double>(num_data_points));
            }

            m_normalized_data_points = Normalize(m_original_data_points);
        }

        /// \brief Get the normalized data points.
        const Eigen::MatrixXd& GetNormalizedDataPoints() const { return m_normalized_data_points; }

        /// \brief Normalize a new set of data points by the same transformation as the originally provided data points.
        Eigen::MatrixXd Normalize(const Eigen::MatrixXd& data_points) const
        {
            const int num_dims        = data_points.rows();
            const int num_data_points = data_points.cols();

            Eigen::MatrixXd normalized_data_points(num_dims, num_data_points);
            for (int dim = 0; dim < num_dims; ++dim)
            {
                constexpr double epsilon = 1e-32;
                const double     scale   = 1.0 / std::max(m_stdev(dim), epsilon);
                const auto       offset  = Eigen::VectorXd::Constant(num_data_points, m_mean(dim));

                normalized_data_points.row(dim) = scale * (data_points.row(dim) - offset.transpose());
            }

            return normalized_data_points;
        }

        /// \brief Denormalize a new set of data points that is represented as the normalized form.
        Eigen::MatrixXd Denormalize(const Eigen::MatrixXd& normalized_data_points) const
        {
            const int num_dims        = normalized_data_points.rows();
            const int num_data_points = normalized_data_points.cols();

            Eigen::MatrixXd data_points(num_dims, num_data_points);
            for (int dim = 0; dim < num_dims; ++dim)
            {
                constexpr double epsilon = 1e-32;
                const double     scale   = std::max(m_stdev(dim), epsilon);
                const auto       offset  = Eigen::VectorXd::Constant(num_data_points, m_mean(dim));

                data_points.row(dim) = scale * normalized_data_points.row(dim) + offset.transpose();
            }

            return data_points;
        }

        /// \brief Get internal parameters about the means.
        const Eigen::VectorXd& GetMean() const { return m_mean; }

        /// \brief Get internal parameters about the standard deviations.
        const Eigen::VectorXd& GetStdev() const { return m_stdev; }

    private:
        const Eigen::MatrixXd m_original_data_points;

        Eigen::MatrixXd m_normalized_data_points;
        Eigen::VectorXd m_mean;
        Eigen::VectorXd m_stdev;
    };
} // namespace mathtoolbox

#endif // MATHTOOLBOX_DATA_NORMALIZATION_HPP
