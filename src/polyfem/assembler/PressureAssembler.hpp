#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/mesh/Obstacle.hpp>

#include <polyfem/assembler/Problem.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>

namespace polyfem
{
	namespace assembler
	{
		class ThermodynamicProcess
		{
		public:
			ThermodynamicProcess() {}
			virtual ~ThermodynamicProcess(){};
			virtual double pressure(const double start_pressure, const double v0, const double v1) const { return 0; }
			virtual double first_derivative(const double start_pressure, const double v0, const double v1) const { return 0; }
			virtual double second_derivative(const double start_pressure, const double v0, const double v1) const { return 0; }
			virtual double energy(const double start_pressure, const double v0, const double v1) const { return 0; }

		protected:
			const double atmospheric = 101.3e3;
		};

		class IsothermalProcess : public ThermodynamicProcess
		{
		public:
			virtual double pressure(const double start_pressure, const double v0, const double v1) const override
			{
				return ((start_pressure + atmospheric) * v0) / v1 - atmospheric;
			}

			virtual double first_derivative(const double start_pressure, const double v0, const double v1) const override
			{
				return -((start_pressure + atmospheric) * v0) / (v1 * v1);
			}

			virtual double second_derivative(const double start_pressure, const double v0, const double v1) const override
			{
				return 2 * ((start_pressure + atmospheric) * v0) / (v1 * v1 * v1);
			}

			virtual double energy(const double start_pressure, const double v0, const double v1) const override { return (start_pressure + atmospheric) * v0 * std::log(v1) - atmospheric * v1; }
		};

		class AdiabaticProcess : public ThermodynamicProcess
		{
		public:
			virtual double pressure(const double start_pressure, const double v0, const double v1) const override
			{
				return ((start_pressure + atmospheric) * std::pow(v0 / v1, gamma_)) - atmospheric;
			}

			virtual double first_derivative(const double start_pressure, const double v0, const double v1) const override
			{
				return -gamma_ * (start_pressure + atmospheric) * std::pow(v0 / v1, gamma_) / v1;
			}

			virtual double second_derivative(const double start_pressure, const double v0, const double v1) const override
			{
				return gamma_ * (1 + gamma_) * (start_pressure + atmospheric) * std::pow(v0 / v1, gamma_) / v1 / v1;
			}

			virtual double energy(const double start_pressure, const double v0, const double v1) const override { return (start_pressure + atmospheric) * std::pow(v0, gamma_) * (std::pow(v1, 1. - gamma_)) / (1. - gamma_) - atmospheric * v1; }

		private:
			const double gamma_ = 1.4;
		};

		// computes the rhs of a problem by \int \phi rho rhs
		class PressureAssembler
		{
		public:
			// initialization with assembler factory mesh
			// size of the problem, bases
			// and solver used internally
			PressureAssembler(const Assembler &assembler, const mesh::Mesh &mesh, const mesh::Obstacle &obstacle,
							  const std::vector<mesh::LocalBoundary> &local_pressure_boundary,
							  const std::unordered_map<int, std::vector<mesh::LocalBoundary>> &local_pressure_cavity,
							  const std::vector<int> &dirichlet_nodes,
							  const std::vector<int> &primitive_to_nodes, const std::vector<int> &node_to_primitives,
							  const int n_basis, const int size,
							  const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases, const Problem &problem);

			double compute_energy(
				const Eigen::MatrixXd &displacement,
				const std::vector<mesh::LocalBoundary> &local_pressure_boundary,
				const int resolution,
				const double t) const;
			void compute_energy_grad(
				const Eigen::MatrixXd &displacement,
				const std::vector<mesh::LocalBoundary> &local_pressure_boundary,
				const std::vector<int> &dirichlet_nodes,
				const int resolution,
				const double t,
				Eigen::VectorXd &grad) const;
			void compute_energy_hess(
				const Eigen::MatrixXd &displacement,
				const std::vector<mesh::LocalBoundary> &local_pressure_boundary,
				const std::vector<int> &dirichlet_nodes,
				const int resolution,
				const double t,
				const bool project_to_psd,
				StiffnessMatrix &hess) const;

			double compute_cavity_energy(
				const Eigen::MatrixXd &displacement,
				const std::unordered_map<int, std::vector<mesh::LocalBoundary>> &local_pressure_cavity,
				const int resolution,
				const double t) const;
			void compute_cavity_energy_grad(
				const Eigen::MatrixXd &displacement,
				const std::unordered_map<int, std::vector<mesh::LocalBoundary>> &local_pressure_cavity,
				const std::vector<int> &dirichlet_nodes,
				const int resolution,
				const double t,
				Eigen::VectorXd &grad) const;
			void compute_cavity_energy_hess(
				const Eigen::MatrixXd &displacement,
				const std::unordered_map<int, std::vector<mesh::LocalBoundary>> &local_pressure_cavity,
				const std::vector<int> &dirichlet_nodes,
				const int resolution,
				const double t,
				const bool project_to_psd,
				StiffnessMatrix &hess) const;

			void compute_force_jacobian(
				const Eigen::MatrixXd &displacement,
				const std::vector<mesh::LocalBoundary> &local_pressure_boundary,
				const std::vector<int> &dirichlet_nodes,
				const int resolution,
				const double t,
				const int n_vertices,
				StiffnessMatrix &hess) const;

			inline const Problem &problem() const { return problem_; }
			inline const mesh::Mesh &mesh() const { return mesh_; }
			inline const std::vector<basis::ElementBases> &bases() const { return bases_; }
			inline const std::vector<basis::ElementBases> &gbases() const { return gbases_; }
			inline const Assembler &assembler() const { return assembler_; }

			void compute_grad_volume_id(const Eigen::MatrixXd &displacement,
										const int boundary_id,
										const std::vector<mesh::LocalBoundary> &local_boundary,
										const std::vector<int> &dirichlet_nodes,
										const int resolution,
										Eigen::VectorXd &grad,
										const double t = 0,
										const bool multiply_pressure = false) const;

		private:
			double compute_volume(
				const Eigen::MatrixXd &displacement,
				const std::vector<mesh::LocalBoundary> &local_boundary,
				const int resolution,
				const double t = 0,
				const bool multiply_pressure = false) const;
			void compute_grad_volume(const Eigen::MatrixXd &displacement,
									 const std::vector<mesh::LocalBoundary> &local_boundary,
									 const std::vector<int> &dirichlet_nodes,
									 const int resolution,
									 Eigen::VectorXd &grad,
									 const double t = 0,
									 const bool multiply_pressure = false) const;
			void compute_hess_volume_3d(
				const Eigen::MatrixXd &displacement,
				const std::vector<mesh::LocalBoundary> &local_boundary,
				const std::vector<int> &dirichlet_nodes,
				const int resolution,
				StiffnessMatrix &hess,
				const double t = 0,
				const bool multiply_pressure = false) const;
			void compute_hess_volume_2d(
				const Eigen::MatrixXd &displacement,
				const std::vector<mesh::LocalBoundary> &local_boundary,
				const std::vector<int> &dirichlet_nodes,
				const int resolution,
				StiffnessMatrix &hess,
				const double t = 0,
				const bool multiply_pressure = false) const;

			bool is_closed_or_boundary_fixed(
				const std::vector<mesh::LocalBoundary> &local_boundary,
				const std::vector<int> &dirichlet_nodes) const;

		private:
			const Assembler &assembler_;
			const mesh::Mesh &mesh_;
			const mesh::Obstacle &obstacle_;
			const int n_basis_;
			const int size_;
			const std::vector<basis::ElementBases> &bases_;
			const std::vector<basis::ElementBases> &gbases_;
			const Problem &problem_;

			std::unordered_map<int, double> starting_volumes_;
			std::unique_ptr<ThermodynamicProcess> cavity_thermodynamics_;

			const std::vector<int> primitive_to_nodes_;
			const std::vector<int> node_to_primitives_;
			std::set<int> relevant_pressure_nodes_;
		};
	} // namespace assembler
} // namespace polyfem
