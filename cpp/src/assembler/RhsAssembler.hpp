#ifndef RHS_ASSEMBLER_HPP
#define RHS_ASSEMBLER_HPP

#include "ElementAssemblyValues.hpp"
#include "Problem.hpp"
#include "LocalBoundary.hpp"

#include <vector>


namespace poly_fem
{
	class RhsAssembler
	{
	public:
		RhsAssembler(const Mesh &mesh, const int n_basis, const int size, const std::vector< ElementBases > &bases, const std::vector< ElementBases > &gbases, const Problem &problem);

		void assemble(Eigen::MatrixXd &rhs) const;
		void set_bc(const std::vector< LocalBoundary > &local_boundary, const std::vector<int> &bounday_nodes, const int resolution,  Eigen::MatrixXd &rhs) const;

	private:
		bool sample_boundary(const LocalBoundary &local_boundary, const int n_samples, const bool skip_computation, Eigen::MatrixXd &samples, Eigen::VectorXi &global_primitive_ids) const;

		const Mesh &mesh_;
		const int n_basis_;
		const int size_;
		const std::vector< ElementBases > &bases_;
		const std::vector< ElementBases > &gbases_;
		bool is_volume_;
		const Problem &problem_;
	};
}

#endif //RHS_ASSEMBLER_HPP