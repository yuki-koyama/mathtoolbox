#ifndef gaussian_process_regression_hpp
#define gaussian_process_regression_hpp

#include <Eigen/Core>

namespace mathtoolbox
{
    class GaussianProcessRegression
    {
    public:
        
        GaussianProcessRegression(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
        
        double EstimateY(const Eigen::VectorXd& x) const;
        double EstimateS(const Eigen::VectorXd& x) const;
        
        // Getter methods
        const Eigen::MatrixXd& GetX() const { return X; }
        const Eigen::VectorXd& GetY() const { return y; }

    private:
        
        // Data points
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        
        // Derivative data
        Eigen::MatrixXd K;
        Eigen::MatrixXd K_inv;
        
        // Hyperparameters
        double          s_f_squared;
        double          s_n_squared;
        Eigen::VectorXd l;
    };
}

#endif /* gaussian_process_regression_hpp */
