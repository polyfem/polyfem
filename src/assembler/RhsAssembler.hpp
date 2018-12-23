#ifndef RHS_ASSEMBLER_HPP
#define RHS_ASSEMBLER_HPP

#include <functional>

#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/Problem.hpp>
#include <polyfem/LocalBoundary.hpp>
#include <polyfem/Types.hpp>

#include <vector>


namespace polyfem
{
	class RhsAssembler
	{
	public:
		RhsAssembler(const Mesh &mesh, const int n_basis, const int size, const std::vector< ElementBases > &bases, const std::vector< ElementBases > &gbases, const std::string &formulation, const Problem &problem);

		void assemble(Eigen::MatrixXd &rhs, const double t = 1) const;

		void initial_solution(Eigen::MatrixXd &sol) const;
		void initial_velocity(Eigen::MatrixXd &sol) const;
		void initial_acceleration(Eigen::MatrixXd &sol) const;

		void set_bc(const std::vector< LocalBoundary > &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector< LocalBoundary > &local_neumann_boundary, Eigen::MatrixXd &rhs, const double t = 1) const;
		void set_velocity_bc(const std::vector< LocalBoundary > &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector< LocalBoundary > &local_neumann_boundary, Eigen::MatrixXd &rhs, const double t = 1) const;
		void set_acceleration_bc(const std::vector< LocalBoundary > &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector< LocalBoundary > &local_neumann_boundary, Eigen::MatrixXd &rhs, const double t = 1) const;

		double compute_energy(const Eigen::MatrixXd &displacement, const std::vector< LocalBoundary > &local_neumann_boundary, const int resolution, const double t) const;
		void compute_energy_grad(const std::vector< LocalBoundary > &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector< LocalBoundary > &local_neumann_boundary, const Eigen::MatrixXd &final_rhs, const double t, Eigen::MatrixXd &rhs) const;

		inline const std::string &formulation() const { return formulation_; }

	private:
		void set_bc(
			const std::function<void(const Eigen::MatrixXi&, const Eigen::MatrixXd&, const Eigen::MatrixXd&, Eigen::MatrixXd &)> &df,
			const std::function<void(const Eigen::MatrixXi&, const Eigen::MatrixXd&, const Eigen::MatrixXd&, Eigen::MatrixXd &)> &nf,
			const std::vector< LocalBoundary > &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector< LocalBoundary > &local_neumann_boundary, Eigen::MatrixXd &rhs) const;

		void time_bc(const std::function<void(const Eigen::MatrixXd&, Eigen::MatrixXd&)> &fun,Eigen::MatrixXd &sol) const;

		bool sample_boundary(const LocalBoundary &local_boundary, const int n_samples, const bool skip_computation, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples, Eigen::VectorXi &global_primitive_ids) const;
		bool boundary_quadrature(const LocalBoundary &local_boundary, const int order, const bool skip_computation, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights, Eigen::VectorXi &global_primitive_ids) const;

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