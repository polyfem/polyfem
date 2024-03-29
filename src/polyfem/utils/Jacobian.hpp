#include <polyfem/basis/ElementBases.hpp>

namespace polyfem::utils
{
    bool isValid(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const Eigen::VectorXd &u);

    template<unsigned int nVar>
    bool isValid(const Eigen::Matrix<double, -1, nVar> &cp, int order);
}