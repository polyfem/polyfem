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
		void assemble(const int n_basis, const int size, const std::vector< ElementAssemblyValues > &values, const std::vector< ElementAssemblyValues > &geom_values, const Problem &problem, Eigen::MatrixXd &rhs) const;
		void set_bc(const int size, const std::vector< ElementBases > &bases, const std::vector< ElementBases > &geom_bases, const bool is_volume, const std::vector< LocalBoundary > &local_boundary, const std::vector<int> &bounday_nodes, const int resolution,  const Problem &problem, Eigen::MatrixXd &rhs) const;

	private:
		bool sample_boundary(const bool is_volume, const LocalBoundary &local_boundary, const int resolution_one_d, const bool skip_computation, Eigen::MatrixXd &samples) const;
	};
}

#endif //RHS_ASSEMBLER_HPP