#include "NLProblem.hpp"

namespace polyfem::solver
{
	class HomogenizationNLProblem : public NLProblem
    {
    public:
		HomogenizationNLProblem(const int full_size,
				  const std::string &formulation,
				  const std::vector<int> &boundary_nodes,
				  const std::vector<mesh::LocalBoundary> &local_boundary,
				  const int n_boundary_samples,
				  const assembler::RhsAssembler &rhs_assembler,
				  const State &state,
				  const double t, std::vector<std::shared_ptr<Form>> &forms): NLProblem(full_size, formulation, boundary_nodes, local_boundary, n_boundary_samples, rhs_assembler, state, t, forms) {}

		TVector full_to_reduced(const TVector &full) const override
        {
            return NLProblem::full_to_reduced(full);
        }
		TVector reduced_to_full(const TVector &reduced) const override
        {
			if (reduced.size() == full_size())
				return reduced;
			else
            	return NLProblem::reduced_to_full(reduced) + macro_field_;
        }

        void set_macro_field(const Eigen::VectorXd &macro_field) { macro_field_ = macro_field; }

	private:
        Eigen::VectorXd macro_field_;
    };
}