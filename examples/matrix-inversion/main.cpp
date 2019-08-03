#include <Eigen/LU>
#include <mathtoolbox/matrix-inversion.hpp>

using Eigen::MatrixXd;

int main(int argc, char** argv)
{
    constexpr int size = 500;

    const MatrixXd random_matrix = [](const int size) {
        constexpr int large_number = 1;

        for (int i = 0; i < large_number; ++i)
        {
            const MatrixXd candidate = MatrixXd::Random(size, size);

            if ((candidate * candidate.inverse() - MatrixXd::Identity(size, size)).cwiseAbs().maxCoeff() < 1e-10)
            { return candidate; }
        }

        throw std::runtime_error("Failed to find an invertible matrix.");
    }(size);

    const MatrixXd result = mathtoolbox::GetInverseUsingUpperLeftBlockInverse(
        random_matrix, random_matrix.block(0, 0, size / 2, size / 2).inverse());

    return 0;
}
