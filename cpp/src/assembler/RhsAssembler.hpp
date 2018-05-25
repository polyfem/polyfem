#ifndef RHS_ASSEMBLER_HPP
#define RHS_ASSEMBLER_HPP

#include "ElementAssemblyValues.hpp"
#include "Problem.hpp"
#include "LocalBoundary.hpp"
#include "Types.hpp"

#include <vector>


namespace poly_fem
{
	class RhsAssembler
	{
	public:
		RhsAssembler(const Mesh &mesh, const int n_basis, const int size, const std::vector< ElementBases > &bases, const std::vector< ElementBases > &gbases, const std::string &formulation, const Problem &problem);

		void assemble(Eigen::MatrixXd &rhs) const;
		void set_bc(const std::vector< LocalBoundary > &local_boundary, const std::vector<int> &bounday_nodes, const int resolution,  const std::vector< LocalBoundary > &local_neumann_boundary, Eigen::MatrixXd &rhs) const;

		double compute_energy(const Eigen::MatrixXd &displacement, const std::vector< LocalBoundary > &local_neumann_boundary, const double t) const;

		inline const std::string &formulation() const { return formulation_; }

	private:
		bool sample_boundary(const LocalBoundary &local_boundary, const int n_samples, const bool skip_computation, Eigen::MatrixXd &samples, Eigen::VectorXi &global_primitive_ids) const;
		bool boundary_quadrature(const LocalBoundary &local_boundary, const int order, const bool skip_computation, Eigen::MatrixXd &points, Eigen::VectorXd &weights, Eigen::VectorXi &global_primitive_ids) const;

		const Mesh &mesh_;
		const int n_basis_;
		const int size_;
		const std::vector< ElementBases > &bases_;
		const std::vector< ElementBases > &gbases_;
		const std::string formulation_;
		const Problem &problem_;
	};
}

#endif //RHS_ASSEMBLER_HPP