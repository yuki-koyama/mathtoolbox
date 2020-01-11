#ifndef MATHTOOLBOX_SOM_HPP
#define MATHTOOLBOX_SOM_HPP

#include <Eigen/Core>
#include <memory>

namespace mathtoolbox
{
    class DataNormalizer;

    class Som
    {
    public:
        /// \param data An n-by-m matrix representing m data points lying in an n-dimensional space.
        ///
        /// \param latent_num_dims The dimensionality of the latent (map) space. It should be either 1 or 2.
        Som(const Eigen::MatrixXd& data,
            const int              latent_num_dims      = 2,
            const int              resolution           = 10,
            const double           init_var             = 0.50,
            const double           min_var              = 0.01,
            const double           var_decreasing_speed = 20.0,
            const bool             normalize_data       = true);

        const Eigen::MatrixXd& GetLatentSpaceNodePositions() const { return m_latent_node_positions; }
        const Eigen::MatrixXd& GetLatentSpaceDataPositions() const { return m_Z; }

        Eigen::MatrixXd GetDataSpaceNodePositions() const;

        void Step();

    private:
        const int m_latent_num_dims;
        const int m_resolution;

        const double m_init_var;
        const double m_min_var;
        const double m_var_decreasing_speed;

        const bool m_normalize_data;

        /// \brief Grid node positions in the latent space.
        const Eigen::MatrixXd m_latent_node_positions;

        int m_iter_count;

        /// \brief Observed data points.
        Eigen::MatrixXd m_X;

        /// \brief Vector values on grid nodes.
        Eigen::MatrixXd m_Y;

        /// \brief Embeded data points.
        Eigen::MatrixXd m_Z;

        std::shared_ptr<const DataNormalizer> m_data_normalizer;

        /// \brief Perform normalization for the current data.
        void NormalizeData();

        /// \brief Perform initialization.
        ///
        /// \details Currently, only the random initialization strategy is implemented.
        void PerformInitialization();
    };
} // namespace mathtoolbox

#endif // MATHTOOLBOX_SOM_HPP
