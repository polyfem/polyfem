#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>

#include <functional>

namespace polyfem::assembler
{
	class NavierStokesFSIAssemblerData : public MultiSpacesNLAssemblerData
	{
	public:
		enum Space : int
		{
			Velocity = 0,
			Pressure = 1,
			MeshDisplacement = 2,
			NumberOfSpaces = 3
		};

		using BodyForceEvaluator = std::function<void(
			int element_id,
			const Eigen::MatrixXd &physical_points,
			double time,
			Eigen::MatrixXd &values)>;

		NavierStokesFSIAssemblerData(
			Values vals,
			Coefficients x,
			Coefficients x_prev,
			const double t,
			const double dt,
			const QuadratureVector &da,
			const Eigen::VectorXd &velocity_tilde,
			const Eigen::VectorXd &mesh_velocity,
			const double dmesh_velocity_dmesh_displacement,
			const double spatial_weight,
			const bool include_inertia,
			const bool picard,
			BodyForceEvaluator body_force_evaluator = {})
			: MultiSpacesNLAssemblerData(std::move(vals), std::move(x), std::move(x_prev), t, dt, da),
			  velocity_tilde(velocity_tilde),
			  mesh_velocity(mesh_velocity),
			  dmesh_velocity_dmesh_displacement(dmesh_velocity_dmesh_displacement),
			  spatial_weight(spatial_weight),
			  include_inertia(include_inertia),
			  picard(picard),
			  body_force_evaluator(std::move(body_force_evaluator))
		{
			assert(n_spaces() == NumberOfSpaces);
		}

		const Eigen::VectorXd &velocity() const { return x(Velocity); }
		const Eigen::VectorXd &pressure() const { return x(Pressure); }
		const Eigen::VectorXd &mesh_displacement() const { return x(MeshDisplacement); }

		const Eigen::VectorXd &velocity_tilde;
		const Eigen::VectorXd &mesh_velocity;
		const double dmesh_velocity_dmesh_displacement;
		const double spatial_weight;
		const bool include_inertia;
		const bool picard;
		const BodyForceEvaluator body_force_evaluator;
	};

	class NavierStokesFSIVelocity : public MultiSpacesNLAssembler
	{
	public:
		NavierStokesFSIVelocity();
		std::string name() const override { return "NavierStokesFSIVelocity"; }
		std::map<std::string, ParamFunc> parameters() const override;
		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;
		double compute_energy(const MultiSpacesNLAssemblerData &) const override;
		Eigen::VectorXd assemble_gradient(const MultiSpacesNLAssemblerData &data) const override;
		Eigen::MatrixXd assemble_hessian(const MultiSpacesNLAssemblerData &data, int row_space, int col_space) const override;
		bool is_fluid() const override { return true; }
		bool is_tensor() const override { return true; }

	private:
		GenericMatParam viscosity_;
		Density density_;
	};

	class NavierStokesFSIMixed : public MultiSpacesNLAssembler
	{
	public:
		std::string name() const override { return "NavierStokesFSIMixed"; }
		std::map<std::string, ParamFunc> parameters() const override { return {}; }
		double compute_energy(const MultiSpacesNLAssemblerData &) const override;
		Eigen::VectorXd assemble_gradient(const MultiSpacesNLAssemblerData &data) const override;
		Eigen::MatrixXd assemble_hessian(const MultiSpacesNLAssemblerData &data, int row_space, int col_space) const override;
		void set_size(const int size) override { size_ = size; }
	};

	class NavierStokesFSIPressure : public MultiSpacesNLAssembler
	{
	public:
		std::string name() const override { return "NavierStokesFSIPressure"; }
		std::map<std::string, ParamFunc> parameters() const override { return {}; }
		double compute_energy(const MultiSpacesNLAssemblerData &) const override { return 0; }
		Eigen::VectorXd assemble_gradient(const MultiSpacesNLAssemblerData &data) const override;
		Eigen::MatrixXd assemble_hessian(const MultiSpacesNLAssemblerData &data, int row_space, int col_space) const override;
		void set_size(const int) override { size_ = 1; }
	};

	class NavierStokesFSIInertia : public MultiSpacesNLAssembler
	{
	public:
		std::string name() const override { return "NavierStokesFSIInertia"; }
		std::map<std::string, ParamFunc> parameters() const override;
		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;
		double compute_energy(const MultiSpacesNLAssemblerData &) const override;
		Eigen::VectorXd assemble_gradient(const MultiSpacesNLAssemblerData &data) const override;
		Eigen::MatrixXd assemble_hessian(const MultiSpacesNLAssemblerData &data, int row_space, int col_space) const override;
		bool is_fluid() const override { return true; }
		bool is_tensor() const override { return true; }

	private:
		Density density_;
	};
} // namespace polyfem::assembler
