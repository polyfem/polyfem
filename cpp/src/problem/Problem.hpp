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
	enum class ProblemType : int
	{
		Linear = 0,
		Quadratic = 1,
		Franke = 2,
		Elastic = 3,
		Zero_BC = 4,
		Franke3d = 5,
		ElasticExact = 6,
	};

	class Problem
	{
	public:
		static std::shared_ptr<Problem> get_problem(const ProblemType type);


		virtual void rhs(const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const = 0;
		virtual void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const = 0;

		virtual void exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const { };

		inline ProblemType problem_num() const { return problem_num_; }
		inline ProblemType& problem_num() { return problem_num_; }

		virtual bool has_exact_sol() const = 0;
		virtual bool is_scalar() const = 0;

		void remove_neumann_nodes(const Mesh &mesh, const std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &boundary_nodes);

		std::vector<int> &boundary_ids() { return boundary_ids_; }
		const std::vector<int> &boundary_ids() const { return boundary_ids_; }

		virtual ~Problem() { }
	protected:
		ProblemType problem_num_;

		std::vector<int> boundary_ids_;
	};
}

#endif //PROBLEM_HPP

