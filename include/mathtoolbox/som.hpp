#ifndef MATHTOOLBOX_SOM_HPP
#define MATHTOOLBOX_SOM_HPP

#include <Eigen/Core>

namespace mathtoolbox
{
    class Som
    {
    public:
        Som(const Eigen::MatrixXd X);

    private:
        Eigen::MatrixXd m_X;
    };
} // namespace mathtoolbox

#endif // MATHTOOLBOX_SOM_HPP
