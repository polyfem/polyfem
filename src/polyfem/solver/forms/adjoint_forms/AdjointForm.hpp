#include "../ParametrizationForm.hpp"

namespace polyfem::solver
{
    enum class ParameterType {
        Shape, Material, InitialCondition, DirichletBoundary
    };

    class AdjointForm : public ParametrizationForm 
    {
    public:
        virtual ~AdjointForm() {}

    protected:
        virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const;

        virtual Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value);
        static Eigen::VectorXd compute_adjoint_term(const State &state, const Eigen::MatrixXd &adjoints, const Parameter &param);

        std::vector<std::pair<std::shared_ptr<State>, std::vector<Parameter>>> states;
    };
}