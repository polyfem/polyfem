#pragma once

#include <polyfem/ElementBases.hpp>
#include <polyfem/AssemblerUtils.hpp>
#include <polyfem/LocalBoundary.hpp>
#include <polyfem/Mesh.hpp>

#include <polyfem/Common.hpp>

#include <vector>
#include <Eigen/Dense>
#include <memory>

namespace polyfem
{
	class Problem
	{
	public:
		Problem(const std::string &name);
		virtual void init(const Mesh &mesh) {}

		virtual void rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const = 0;
		virtual bool is_rhs_zero() const = 0;

		virtual void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const = 0;
		virtual void velocity_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const {}
		virtual void acceleration_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const {}

		virtual void neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const {}
		virtual void neumann_velocity_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const {}
		virtual void neumann_acceleration_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const {}

		virtual void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const {};
		virtual void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const {};

		inline const std::string &name() const { return name_; }

		virtual bool has_exact_sol() const = 0;
		virtual bool is_scalar() const = 0;
		// virtual bool is_mixed() const { return false; }
		virtual bool is_linear_in_time() const { return true; }

		virtual void clear() {}

		virtual bool is_time_dependent() const { return false; }
		virtual void initial_solution(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const {}
		virtual void initial_velocity(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const {}
		virtual void initial_acceleration(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const {}
		virtual void initial_density(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const { }

		virtual int n_incremental_load_steps(const double diag) const { return 1; }

		virtual void set_parameters(const json &params) {}

		void setup_bc(const Mesh &mesh, const std::vector<ElementBases> &bases, const std::vector<ElementBases> &pressure_bases, std::vector<LocalBoundary> &local_boundary, std::vector<int> &boundary_nodes, std::vector<LocalBoundary> &local_neumann_boundary, std::vector<int> &pressure_boundary_nodes);

		// std::vector<int> &boundary_ids() { return boundary_ids_; }
		// const std::vector<int> &boundary_ids() const { return boundary_ids_; }

		virtual bool is_dimention_dirichet(const int tag, const int dim) const { return true; }

		//here for efficiency reasons
		virtual bool all_dimentions_dirichelt() const { return true; }

		virtual ~Problem() {}

	protected:
		std::vector<int> boundary_ids_;
		std::vector<int> neumann_boundary_ids_;
		std::vector<int> pressure_boundary_ids_;
		std::vector<int> splitting_pressure_boundary_ids_;

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
} // namespace polyfem
