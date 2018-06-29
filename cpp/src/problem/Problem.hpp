#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <polyfem/ElementBases.hpp>
#include <polyfem/LocalBoundary.hpp>
#include <polyfem/Mesh.hpp>

#include <polyfem/Common.hpp>

#include <vector>
#include <Eigen/Dense>
#include <memory>

namespace poly_fem
{
	class Problem
	{
	public:
		Problem(const std::string &name);

		virtual void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const = 0;
		virtual void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const = 0;
		virtual void neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const { }

		virtual void exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const { };
		virtual void exact_grad(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const { };

		inline const std::string &name() const { return name_; }

		virtual bool has_exact_sol() const = 0;
		virtual bool is_scalar() const = 0;
		virtual bool is_linear_in_time() const { return true; }

		virtual void set_parameters(const json &params) { }

		void setup_bc(const Mesh &mesh, const std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &boundary_nodes, std::vector< LocalBoundary > &local_neumann_boundary);

		std::vector<int> &boundary_ids() { return boundary_ids_; }
		const std::vector<int> &boundary_ids() const { return boundary_ids_; }

		virtual ~Problem() { }
	protected:
		std::vector<int> boundary_ids_;
		std::vector<int> neumann_boundary_ids_;

	private:
		std::string name_;
	};



	class ProblemFactory
	{
	public:
		static const ProblemFactory &factory();

		std::shared_ptr<Problem> get_problem(const std::string &problem) const;
		inline const std::vector<std::string> &get_problem_names() const { return problem_names_; }


	private:
		ProblemFactory();
		std::map<std::string, std::shared_ptr<Problem>> problems_;
		std::vector<std::string> problem_names_;
	};
}

#endif //PROBLEM_HPP

