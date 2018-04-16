#ifndef RBF_INTERPOLATION_HPP
#define RBF_INTERPOLATION_HPP

#include <vector>

namespace mathtoolbox
{
    enum class RbfType
    {
        Gaussian,         // f(r) = exp(-(epsilon * r)^2)
        ThinPlateSpline,  // f(r) = (r^2) * log(r)
        InverseQuadratic, // f(r) = (1 + (epsilon * r)^2)^(-1)
        BiharmonicSpline, // f(r) = r
    };
    
    class RbfInterpolation
    {
    public:
        RbfInterpolation(RbfType rbf_type = RbfType::BiharmonicSpline, double epsilon = 2.0);
        
        // API
        void   Reset();
        void   AddCenterPoint(double y, const std::vector<double>& x);
        void   ComputeWeights(bool use_regularization = false, double lambda = 0.1);
        double GetValue(const std::vector<double>& x) const;
        
        // Getter methods
        const std::vector<double>&              GetYs() const { return ys; }
        const std::vector<std::vector<double>>& GetXs() const { return xs; }
        const std::vector<double>&              GetW()  const { return w;  }
        
    private:
        
        // Function type
        RbfType rbf_type;
        
        // A control parameter used in some kernel functions
        double epsilon;
        
        // Data points
        std::vector<double>              ys;
        std::vector<std::vector<double>> xs;
        
        // Weights
        std::vector<double>              w;
        
        // Returns f(r)
        double GetRbfValue(double r) const;
        
        // Returns f(||xj - xi||)
        double GetRbfValue(const std::vector<double>& xi, const std::vector<double>& xj) const;
    };
}

#endif // RBF_INTERPOLATION_HPP
