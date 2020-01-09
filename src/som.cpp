#include <Eigen/SparseCore>
#include <mathtoolbox/data-normalization.hpp>
#include <mathtoolbox/som.hpp>

namespace
{
    int GetNumNodes(const int resolution, const int latent_num_dims)
    {
        assert(latent_num_dims == 1 || latent_num_dims == 2);

        if (latent_num_dims == 1)
        {
            return resolution;
        }
        else
        {
            return resolution * resolution;
        }
    }

    /// \details The computational complexity is O(#data * #nodes)
    Eigen::VectorXi FindBestMatchingUnits(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    {
        assert(X.rows() == Y.rows());

        const int num_data  = X.cols();
        const int num_nodes = Y.cols();

        Eigen::VectorXi node_indices(num_data);

        for (int i = 0; i < num_data; ++i)
        {
            Eigen::VectorXd squared_norms(num_nodes);

            for (int j = 0; j < num_nodes; ++j)
            {
                squared_norms(j) = (X.col(i) - Y.col(j)).squaredNorm();
            }

            squared_norms.minCoeff(&(node_indices(i)));
        }

        return node_indices;
    }

    Eigen::SparseMatrix<double> ConvertBestMatchingUnitsIntoMat(const Eigen::VectorXi& best_matching_units,
                                                                const int              num_nodes)
    {
        const int num_data = best_matching_units.size();

        Eigen::SparseMatrix<double> B(num_nodes, num_data);

        for (int data_index = 0; data_index < num_data; ++data_index)
        {
            B.insert(best_matching_units(data_index), data_index) = 1.0;
        }

        return B;
    }

    /// \return Posititons in the latent space [0, 1]^{#dims x #nodes}
    Eigen::MatrixXd GetLatentSpacePositions(const int resolution, const int latent_num_dims)
    {
        assert(latent_num_dims == 1 || latent_num_dims == 2);
        assert(resolution >= 2);

        const int num_nodes = GetNumNodes(resolution, latent_num_dims);

        Eigen::MatrixXd positions(latent_num_dims, num_nodes);

        if (latent_num_dims == 1)
        {
            for (int x_index = 0; x_index < num_nodes; ++x_index)
            {
                positions(0, x_index) = static_cast<double>(x_index) / static_cast<double>(resolution - 1);
            }
        }
        else
        {
            for (int y_index = 0; y_index < resolution; ++y_index)
            {
                for (int x_index = 0; x_index < resolution; ++x_index)
                {
                    const int index = y_index * resolution + x_index;

                    const double x_offset = static_cast<double>(x_index) / static_cast<double>(resolution - 1);
                    const double y_offset = static_cast<double>(y_index) / static_cast<double>(resolution - 1);

                    positions.col(index) = Eigen::Vector2d(x_offset, y_offset);
                }
            }
        }

        return positions;
    }

    Eigen::MatrixXd CalcNeighborhoodMat(const Eigen::MatrixXd& latent_node_positions,
                                        const int              iter_count,
                                        const double           init_var,
                                        const double           min_var,
                                        const double           var_decreasing_speed)
    {
        const int    num_nodes = latent_node_positions.cols();
        const double var       = std::max(init_var * std::exp(-iter_count / var_decreasing_speed), min_var);

        Eigen::MatrixXd H(num_nodes, num_nodes);

        for (int i = 0; i < num_nodes; ++i)
        {
            for (int j = i; j < num_nodes; ++j)
            {
                const double squared_dist = (latent_node_positions.col(i) - latent_node_positions.col(j)).squaredNorm();
                const double value        = std::exp(-(1.0 / (2.0 * var)) * squared_dist);

                H(i, j) = value;
                H(j, i) = value;
            }
        }

        return H;
    }
} // namespace

mathtoolbox::Som::Som(const Eigen::MatrixXd& data,
                      const int              latent_num_dims,
                      const int              resolution,
                      const double           init_var,
                      const double           min_var,
                      const double           var_decreasing_speed,
                      const bool             normalize_data)
    : m_latent_num_dims(latent_num_dims),
      m_resolution(resolution),
      m_init_var(init_var),
      m_min_var(min_var),
      m_var_decreasing_speed(var_decreasing_speed),
      m_normalize_data(normalize_data),
      m_latent_node_positions(GetLatentSpacePositions(resolution, latent_num_dims)),
      m_iter_count(0),
      m_X(data),
      m_data_normalizer(nullptr)
{
    if (m_normalize_data)
    {
        this->NormalizeData();
    }

    this->PerformInitialization();
}

Eigen::MatrixXd mathtoolbox::Som::GetDataSpaceNodePositions() const
{
    return m_normalize_data ? m_data_normalizer->Denormalize(m_Y) : m_Y;
}

void mathtoolbox::Som::Step()
{
    const int             num_nodes           = GetNumNodes(m_resolution, m_latent_num_dims);
    const int             num_data            = m_X.cols();
    const Eigen::VectorXi best_matching_units = FindBestMatchingUnits(m_X, m_Y);

    // #nodes * #data
    const Eigen::SparseMatrix<double> B = ConvertBestMatchingUnitsIntoMat(best_matching_units, num_nodes);

    // #nodes * #nodes
    const Eigen::MatrixXd H =
        CalcNeighborhoodMat(m_latent_node_positions, m_iter_count, m_init_var, m_min_var, m_var_decreasing_speed);

    // #nodes * #data
    const Eigen::MatrixXd R = H * B;

    // #nodes * #nodes
    Eigen::VectorXd G_inv_diag(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
    {
        G_inv_diag(i) = 1.0 / R.row(i).sum();
    }
    const auto G_inv = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(G_inv_diag);

    // Update Y (the positions of the grid nodes in the data space)
    m_Y = (G_inv * H * (B * m_X.transpose())).transpose();

    // Update Z (the positions of the data points in the latent space)
    for (int i = 0; i < num_data; ++i)
    {
        m_Z.col(i) = m_Y.col(best_matching_units(i));
    }

    // Update the iteration count
    ++m_iter_count;
}

void mathtoolbox::Som::NormalizeData()
{
    // Instantiate a data normalizer object
    m_data_normalizer = std::make_shared<const DataNormalizer>(m_X);

    // Replace X with its normalized version
    m_X = m_data_normalizer->GetNormalizedDataPoints();
}

void mathtoolbox::Som::PerformInitialization()
{
    const int num_data_dims = m_X.rows();
    const int num_data      = m_X.cols();
    const int num_nodes     = GetNumNodes(m_resolution, m_latent_num_dims);

    m_Y = Eigen::MatrixXd::Random(num_data_dims, num_nodes);
    m_Z = Eigen::MatrixXd::Random(num_data_dims, num_data);
}
