#ifndef RHS_ASSEMBLER_HPP
#define RHS_ASSEMBLER_HPP

#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/AssemblerUtils.hpp>

#include <polyfem/Problem.hpp>
#include <polyfem/LocalBoundary.hpp>
#include <polyfem/ElasticityUtils.hpp>
#include <polyfem/Types.hpp>

#include <functional>
#include <vector>

namespace polyfem
{
	//computes the rhs of a problem by \int \phi rho rhs
	class RhsAssembler
	{
	public:
		//initialization with assembler factory mesh
		//size of the problem, bases
		//formulation, problem
		//and solver used internally
		RhsAssembler(const AssemblerUtils &assembler, const Mesh &mesh,
					 const int n_basis, const int size,
					 const std::vector<ElementBases> &bases, const std::vector<ElementBases> &gbases,
					 const std::string &formulation, const Problem &problem,
					 const std::string &solver, const std::string &preconditioner, const json &solver_params);

		//computes the rhs of a problem by \int \phi rho rhs
		void assemble(const Density &density, Eigen::MatrixXd &rhs, const double t = 1) const;

		//computes the inital soltion for time dependent, calls time_bc
		void initial_solution(Eigen::MatrixXd &sol) const;
		//computes the inital velocity for time dependent, calls time_bc
		void initial_velocity(Eigen::MatrixXd &sol) const;
		//computes the inital acceleration for time dependent, calls time_bc
		void initial_acceleration(Eigen::MatrixXd &sol) const;

		//sets boundary conditions to rhs, the boundary conditions are projected (Dirichlet) integrated (Neumann) at resolution
		//local boundary stores the mapping from elemment to nodes for Dirichelt nodes
		//local local_neumann_boundary stores the mapping from elemment to nodes for Neumann nodes
		//calls set_bc
		void set_bc(const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector<LocalBoundary> &local_neumann_boundary, Eigen::MatrixXd &rhs, const double t = 1) const;
		//same for velocity
		void set_velocity_bc(const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector<LocalBoundary> &local_neumann_boundary, Eigen::MatrixXd &rhs, const double t = 1) const;
		//same for acceleration
		void set_acceleration_bc(const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector<LocalBoundary> &local_neumann_boundary, Eigen::MatrixXd &rhs, const double t = 1) const;

		//compute body energy
		double compute_energy(const Eigen::MatrixXd &displacement, const std::vector<LocalBoundary> &local_neumann_boundary, const Density &density, const int resolution, const double t) const;
		//compute body energy gradient, hessian is zero, rhs is a linear function
		void compute_energy_grad(const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const Density &density, const int resolution, const std::vector<LocalBoundary> &local_neumann_boundary, const Eigen::MatrixXd &final_rhs, const double t, Eigen::MatrixXd &rhs) const;

		//return the formulation
		inline const std::string &formulation() const { return formulation_; }

	private:
		//set boundary condition
		//the 2 lambdas are callback to dirichlet df and neumann nf
		//diriclet boundary condition are projected on the FEM bases, it inverts a linear system
		void set_bc(
			const std::function<void(const Eigen::MatrixXi &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, Eigen::MatrixXd &)> &df,
			const std::function<void(const Eigen::MatrixXi &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, Eigen::MatrixXd &)> &nf,
			const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector<LocalBoundary> &local_neumann_boundary, Eigen::MatrixXd &rhs) const;

		//sets the time (initial) boundary condition
		//the lambda depeneds if soltuion, velocity, or acceleration
		//they are projected on the FEM bases, it inverts a linear system
		void time_bc(const std::function<void(const Mesh &, const Eigen::MatrixXi &, const Eigen::MatrixXd &, Eigen::MatrixXd &)> &fun, Eigen::MatrixXd &sol) const;

		//sample boundary facet, uv are local (ref) values, samples are global coordinates, used for dirichlet
		bool sample_boundary(const LocalBoundary &local_boundary, const int n_samples, const bool skip_computation, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples, Eigen::VectorXi &global_primitive_ids) const;
		//compute quadrature for boundary facet, uv are local (ref) values, samples are global coordinates, used for neumann
		bool boundary_quadrature(const LocalBoundary &local_boundary, const int order, const bool skip_computation, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::MatrixXd &normals, Eigen::VectorXd &weights, Eigen::VectorXi &global_primitive_ids) const;

		const AssemblerUtils &assembler_;
		const Mesh &mesh_;
		const int n_basis_;
		const int size_;
		const std::vector<ElementBases> &bases_;
		const std::vector<ElementBases> &gbases_;
		const std::string formulation_;
		const Problem &problem_;
		const std::string solver_, preconditioner_;
		const json solver_params_;
	};
} // namespace polyfem

#endif //RHS_ASSEMBLER_HPP