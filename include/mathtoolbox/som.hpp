#ifndef MATHTOOLBOX_SOM_HPP
#define MATHTOOLBOX_SOM_HPP

#include <Eigen/Core>

namespace mathtoolbox
{
    class Som
    {
    public:
        Som(const Eigen::MatrixXd& data,
            const int              latent_num_dims = 2,
            const int              resolution      = 10,
            const bool             normalize_data  = true);

        const Eigen::MatrixXd& GetGrid() const { return m_Y; }
        const Eigen::MatrixXd& GetEmbedding() const { return m_Z; }

        void Step();

    private:
        const int m_latent_num_dims;
        const int m_resolution;

        int m_iter_count;

        /// \brief Observed data points.
        Eigen::MatrixXd m_X;

        /// \brief Vector values on grid nodes.
        Eigen::MatrixXd m_Y;

        /// \brief Embeded data points.
        Eigen::MatrixXd m_Z;

        /// \brief Perform normalization for the current data.
        void NormalizeData();

        /// \brief Perform initialization.
        ///
        /// \details Currently, only the random initialization strategy is implemented.
        void PerformInitialization();
    };
} // namespace mathtoolbox

#endif // MATHTOOLBOX_SOM_HPP
