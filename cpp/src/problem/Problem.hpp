#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include "ElementBases.hpp"
#include "LocalBoundary.hpp"
#include "Mesh.hpp"

#include <vector>
#include <Eigen/Dense>
#include <memory>

namespace poly_fem
{
	enum class ProblemType
	{
		Linear = 0,
		Quadratic,
		Franke,
		Elastic,
		Zero_BC,
		Franke3d
	};

	class Problem
	{
	public:
		static std::shared_ptr<Problem> get_problem(const ProblemType type);


		virtual void rhs(const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const = 0;
		virtual void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const = 0;

		virtual void exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const { };

		inline ProblemType problem_num() const { return problem_num_; }

		virtual bool has_exact_sol() const = 0;
		virtual bool is_scalar() const = 0;

		void remove_neumann_nodes(const Mesh &mesh, const std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &boundary_nodes);


		virtual ~Problem() { }
	protected:
		ProblemType problem_num_;

		std::vector<int> boundary_ids_;
	};
}

#endif //PROBLEM_HPP

