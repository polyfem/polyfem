
#include "AssemblerUtils.hpp"

#include "Mass.hpp"

#include "Laplacian.hpp"
#include "Bilaplacian.hpp"
#include "Helmholtz.hpp"

#include "LinearElasticity.hpp"
#include "HookeLinearElasticity.hpp"
#include "SaintVenantElasticity.hpp"
#include "NeoHookeanElasticity.hpp"
#include "MooneyRivlinElasticity.hpp"
#include "OgdenElasticity.hpp"
#include "MultiModel.hpp"
#include "ViscousDamping.hpp"

#include "Stokes.hpp"
#include "NavierStokes.hpp"
#include "IncompressibleLinElast.hpp"

#include <polyfem/utils/Logger.hpp>

#include <unsupported/Eigen/SparseExtra>

namespace polyfem
{
	using namespace basis;
	using namespace utils;

	namespace assembler
	{
		struct FormulationProperties
		{
			bool is_scalar;
			bool is_fluid;
			bool is_mixed;
			bool is_solution_displacement;
			bool is_linear;
			bool is_tensor() const { return !is_scalar; }
			bool is_nonlinear() const { return !is_linear; }
		};

		// clang-format off
		static const std::unordered_map<std::string, FormulationProperties> formulation_properties = {
			{"Laplacian",                      {/*is_scalar=*/true,  /*is_fluid=*/false, /*is_mixed=*/false, /*is_solution_displacement=*/false, /*is_linear=*/true}},
			{"Helmholtz",                      {/*is_scalar=*/true,  /*is_fluid=*/false, /*is_mixed=*/false, /*is_solution_displacement=*/false, /*is_linear=*/true}},
			{"Bilaplacian",                    {/*is_scalar=*/true,  /*is_fluid=*/false, /*is_mixed=*/true,  /*is_solution_displacement=*/false, /*is_linear=*/true}},
			{"LinearElasticity",               {/*is_scalar=*/false, /*is_fluid=*/false, /*is_mixed=*/false, /*is_solution_displacement=*/true,  /*is_linear=*/true}},
			{"HookeLinearElasticity",          {/*is_scalar=*/false, /*is_fluid=*/false, /*is_mixed=*/false, /*is_solution_displacement=*/true,  /*is_linear=*/true}},
			{"Damping",                        {/*is_scalar=*/false, /*is_fluid=*/false, /*is_mixed=*/false, /*is_solution_displacement=*/true,  /*is_linear=*/false}},
			{"IncompressibleLinearElasticity", {/*is_scalar=*/false, /*is_fluid=*/false, /*is_mixed=*/true,  /*is_solution_displacement=*/true,  /*is_linear=*/true}},
			{"SaintVenant",                    {/*is_scalar=*/false, /*is_fluid=*/false, /*is_mixed=*/false, /*is_solution_displacement=*/true,  /*is_linear=*/false}},
			{"NeoHookean",                     {/*is_scalar=*/false, /*is_fluid=*/false, /*is_mixed=*/false, /*is_solution_displacement=*/true,  /*is_linear=*/false}},
			{"MooneyRivlin",                   {/*is_scalar=*/false, /*is_fluid=*/false, /*is_mixed=*/false, /*is_solution_displacement=*/true,  /*is_linear=*/false}},
			{"MultiModels",                    {/*is_scalar=*/false, /*is_fluid=*/false, /*is_mixed=*/false, /*is_solution_displacement=*/true,  /*is_linear=*/false}},
			{"UnconstrainedOgden",             {/*is_scalar=*/false, /*is_fluid=*/false, /*is_mixed=*/false, /*is_solution_displacement=*/true, /*is_linear=*/false}},
			{"IncompressibleOgden",            {/*is_scalar=*/false, /*is_fluid=*/false, /*is_mixed=*/false, /*is_solution_displacement=*/true, /*is_linear=*/false}},
			{"Stokes",                         {/*is_scalar=*/false, /*is_fluid=*/true,  /*is_mixed=*/true,  /*is_solution_displacement=*/false, /*is_linear=*/true}},
			{"NavierStokes",                   {/*is_scalar=*/false, /*is_fluid=*/true,  /*is_mixed=*/true,  /*is_solution_displacement=*/false, /*is_linear=*/false}},
			{"OperatorSplitting",              {/*is_scalar=*/false, /*is_fluid=*/true,  /*is_mixed=*/true,  /*is_solution_displacement=*/false, /*is_linear=*/false}},
		};
		// clang-format on

		AssemblerUtils::AssemblerUtils()
		{
			mass_mat_ = std::make_shared<Mass>();
			mass_mat_no_density_ = std::make_shared<Mass>();

			laplacian_ = std::make_shared<Laplacian>();
			helmholtz_ = std::make_shared<Helmholtz>();

			bilaplacian_main_ = std::make_shared<BilaplacianMain>();
			bilaplacian_mixed_ = std::make_shared<BilaplacianMixed>();
			bilaplacian_aux_ = std::make_shared<BilaplacianAux>();

			linear_elasticity_ = std::make_shared<LinearElasticity>();
			linear_elasticity_energy_ = std::make_shared<LinearElasticity>();
			hooke_linear_elasticity_ = std::make_shared<HookeLinearElasticity>();

			saint_venant_elasticity_ = std::make_shared<SaintVenantElasticity>();
			neo_hookean_elasticity_ = std::make_shared<NeoHookeanElasticity>();
			mooney_rivlin_elasticity_ = std::make_shared<MooneyRivlinElasticity>();
			multi_models_elasticity_ = std::make_shared<MultiModel>();
			unconstrained_ogden_elasticity_ = std::make_shared<UnconstrainedOgdenElasticity>();
			incompressible_ogden_elasticity_ = std::make_shared<IncompressibleOgdenElasticity>();

			damping_ = std::make_shared<ViscousDamping>();

			stokes_velocity_ = std::make_shared<StokesVelocity>();
			stokes_mixed_ = std::make_shared<StokesMixed>();
			stokes_pressure_ = std::make_shared<StokesPressure>();

			navier_stokes_velocity_picard_ = std::make_shared<NavierStokesVelocity<false>>();
			navier_stokes_velocity_ = std::make_shared<NavierStokesVelocity<true>>();

			incompressible_lin_elast_displacement_ = std::make_shared<IncompressibleLinearElasticityDispacement>();
			incompressible_lin_elast_mixed_ = std::make_shared<IncompressibleLinearElasticityMixed>();
			incompressible_lin_elast_pressure_ = std::make_shared<IncompressibleLinearElasticityPressure>();
		}

		std::vector<std::string> AssemblerUtils::scalar_assemblers()
		{
			std::vector<std::string> names;
			for (const auto &[name, props] : formulation_properties)
			{
				if (props.is_scalar)
					names.push_back(name);
			}
			return names;
		}

		std::vector<std::string> AssemblerUtils::tensor_assemblers()
		{
			std::vector<std::string> names;
			for (const auto &[name, props] : formulation_properties)
			{
				if (props.is_tensor())
					names.push_back(name);
			}
			return names;
		}

		bool AssemblerUtils::is_scalar(const std::string &assembler)
		{
			return formulation_properties.at(assembler).is_scalar;
		}

		bool AssemblerUtils::is_fluid(const std::string &assembler)
		{
			return formulation_properties.at(assembler).is_fluid;
		}

		bool AssemblerUtils::is_tensor(const std::string &assembler)
		{
			return formulation_properties.at(assembler).is_tensor();
		}
		bool AssemblerUtils::is_mixed(const std::string &assembler)
		{
			return formulation_properties.at(assembler).is_mixed;
		}

		bool AssemblerUtils::is_solution_displacement(const std::string &assembler)
		{
			return formulation_properties.at(assembler).is_solution_displacement;
		}

		bool AssemblerUtils::is_linear(const std::string &assembler)
		{
			return formulation_properties.at(assembler).is_linear;
		}

		bool AssemblerUtils::has_damping() const
		{
			return damping_->is_valid();
		}

		const Density &AssemblerUtils::density() const { return mass_mat_->density(); }

		void AssemblerUtils::assemble_problem(const std::string &assembler,
											  const bool is_volume,
											  const int n_basis,
											  const std::vector<ElementBases> &bases,
											  const std::vector<ElementBases> &gbases,
											  const AssemblyValsCache &cache,
											  StiffnessMatrix &stiffness) const
		{
			if (assembler == "Helmholtz")
				helmholtz_->assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			else if (assembler == "Laplacian")
				laplacian_->assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			else if (assembler == "Bilaplacian")
				bilaplacian_main_->assemble(is_volume, n_basis, bases, gbases, cache, stiffness);

			else if (assembler == "LinearElasticity")
				linear_elasticity_->assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			else if (assembler == "HookeLinearElasticity")
				hooke_linear_elasticity_->assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			else if (assembler == "Stokes" || assembler == "NavierStokes" || assembler == "OperatorSplitting")
				stokes_velocity_->assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			else if (assembler == "IncompressibleLinearElasticity")
				incompressible_lin_elast_displacement_->assemble(is_volume, n_basis, bases, gbases, cache, stiffness);

			else if (assembler == "SaintVenant")
				return;
			else if (assembler == "NeoHookean")
				return;
			else if (assembler == "MooneyRivlin")
				return;
			else if (assembler == "MultiModels")
				return;
			// else if (assembler == "NavierStokes")
			// return;
			else if (assembler == "UnconstrainedOgden")
				return;
			else if (assembler == "IncompressibleOgden")
				return;
			else
			{
				logger().warn("{} not found, fallback to default", assembler);
				assert(false);
				laplacian_->assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			}
		}

		void AssemblerUtils::assemble_mass_matrix(const std::string &assembler,
												  const bool is_volume,
												  const int n_basis,
												  const bool use_density,
												  const std::vector<ElementBases> &bases,
												  const std::vector<ElementBases> &gbases,
												  const AssemblyValsCache &cache,
												  StiffnessMatrix &mass) const
		{
			// TODO use cache
			if (use_density)
				mass_mat_->assemble(is_volume, n_basis, bases, gbases, cache, mass, true);
			else
				mass_mat_no_density_->assemble(is_volume, n_basis, bases, gbases, cache, mass, true);
		}

		void AssemblerUtils::assemble_mixed_problem(const std::string &assembler,
													const bool is_volume,
													const int n_psi_basis,
													const int n_phi_basis,
													const std::vector<ElementBases> &psi_bases,
													const std::vector<ElementBases> &phi_bases,
													const std::vector<ElementBases> &gbases,
													const AssemblyValsCache &psi_cache,
													const AssemblyValsCache &phi_cache,
													StiffnessMatrix &stiffness) const
		{
			// TODO add cache
			if (assembler == "Bilaplacian")
				bilaplacian_mixed_->assemble(is_volume, n_psi_basis, n_phi_basis, psi_bases, phi_bases, gbases, psi_cache, phi_cache, stiffness);

			else if (assembler == "Stokes" || assembler == "NavierStokes" || assembler == "OperatorSplitting")
				stokes_mixed_->assemble(is_volume, n_psi_basis, n_phi_basis, psi_bases, phi_bases, gbases, psi_cache, phi_cache, stiffness);
			else if (assembler == "IncompressibleLinearElasticity")
				incompressible_lin_elast_mixed_->assemble(is_volume, n_psi_basis, n_phi_basis, psi_bases, phi_bases, gbases, psi_cache, phi_cache, stiffness);

			else
			{
				logger().warn("{} not found, fallback to default", assembler);
				assert(false);
				stokes_mixed_->assemble(is_volume, n_psi_basis, n_phi_basis, psi_bases, phi_bases, gbases, psi_cache, phi_cache, stiffness);
			}
		}

		void AssemblerUtils::assemble_pressure_problem(const std::string &assembler,
													   const bool is_volume,
													   const int n_basis,
													   const std::vector<ElementBases> &bases,
													   const std::vector<ElementBases> &gbases,
													   const AssemblyValsCache &cache,
													   StiffnessMatrix &stiffness) const
		{
			if (assembler == "Bilaplacian")
				bilaplacian_aux_->assemble(is_volume, n_basis, bases, gbases, cache, stiffness);

			else if (assembler == "Stokes" || assembler == "NavierStokes" || assembler == "OperatorSplitting")
				stokes_pressure_->assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			else if (assembler == "IncompressibleLinearElasticity")
				incompressible_lin_elast_pressure_->assemble(is_volume, n_basis, bases, gbases, cache, stiffness);

			else
			{
				logger().warn("{} not found, fallback to default", assembler);
				assert(false);
				stokes_pressure_->assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			}
		}

		double AssemblerUtils::assemble_energy(const std::string &assembler,
											   const bool is_volume,
											   const std::vector<ElementBases> &bases,
											   const std::vector<ElementBases> &gbases,
											   const AssemblyValsCache &cache,
											   const double dt,
											   const Eigen::MatrixXd &displacement,
											   const Eigen::MatrixXd &displacement_prev) const
		{
			if (assembler == "SaintVenant")
				return saint_venant_elasticity_->assemble(is_volume, bases, gbases, cache, dt, displacement, displacement_prev);
			else if (assembler == "NeoHookean")
				return neo_hookean_elasticity_->assemble(is_volume, bases, gbases, cache, dt, displacement, displacement_prev);
			else if (assembler == "MooneyRivlin")
				return mooney_rivlin_elasticity_->assemble(is_volume, bases, gbases, cache, dt, displacement, displacement_prev);
			else if (assembler == "MultiModels")
				return multi_models_elasticity_->assemble(is_volume, bases, gbases, cache, dt, displacement, displacement_prev);
			else if (assembler == "Damping")
				return damping_->assemble(is_volume, bases, gbases, cache, dt, displacement, displacement_prev);

			else if (assembler == "UnconstrainedOgden")
				return unconstrained_ogden_elasticity_->assemble(is_volume, bases, gbases, cache, dt, displacement, displacement_prev);
			else if (assembler == "IncompressibleOgden")
				return incompressible_ogden_elasticity_->assemble(is_volume, bases, gbases, cache, dt, displacement, displacement_prev);
			else if (assembler == "LinearElasticity")
				return linear_elasticity_energy_->assemble(is_volume, bases, gbases, cache, dt, displacement, displacement_prev);
			else
				return 0;
		}

		void AssemblerUtils::assemble_energy_gradient(const std::string &assembler,
													  const bool is_volume,
													  const int n_basis,
													  const std::vector<ElementBases> &bases,
													  const std::vector<ElementBases> &gbases,
													  const AssemblyValsCache &cache,
													  const double dt,
													  const Eigen::MatrixXd &displacement,
													  const Eigen::MatrixXd &displacement_prev,
													  Eigen::MatrixXd &grad) const
		{
			if (assembler == "SaintVenant")
				saint_venant_elasticity_->assemble_grad(is_volume, n_basis, bases, gbases, cache, dt, displacement, displacement_prev, grad);
			else if (assembler == "NeoHookean")
				neo_hookean_elasticity_->assemble_grad(is_volume, n_basis, bases, gbases, cache, dt, displacement, displacement_prev, grad);
			else if (assembler == "MooneyRivlin")
				mooney_rivlin_elasticity_->assemble_grad(is_volume, n_basis, bases, gbases, cache, dt, displacement, displacement_prev, grad);
			else if (assembler == "MultiModels")
				multi_models_elasticity_->assemble_grad(is_volume, n_basis, bases, gbases, cache, dt, displacement, displacement_prev, grad);
			else if (assembler == "Damping")
				damping_->assemble_grad(is_volume, n_basis, bases, gbases, cache, dt, displacement, displacement_prev, grad);
			else if (assembler == "NavierStokes")
				navier_stokes_velocity_->assemble_grad(is_volume, n_basis, bases, gbases, cache, dt, displacement, displacement_prev, grad);
			else if (assembler == "LinearElasticity")
				linear_elasticity_energy_->assemble_grad(is_volume, n_basis, bases, gbases, cache, dt, displacement, displacement_prev, grad);
			else if (assembler == "UnconstrainedOgden")
				unconstrained_ogden_elasticity_->assemble_grad(is_volume, n_basis, bases, gbases, cache, dt, displacement, displacement_prev, grad);
			else if (assembler == "IncompressibleOgden")
				incompressible_ogden_elasticity_->assemble_grad(is_volume, n_basis, bases, gbases, cache, dt, displacement, displacement_prev, grad);
			else
				return;
		}

		void AssemblerUtils::assemble_energy_hessian(const std::string &assembler,
													 const bool is_volume,
													 const int n_basis,
													 const bool project_to_psd,
													 const std::vector<ElementBases> &bases,
													 const std::vector<ElementBases> &gbases,
													 const AssemblyValsCache &cache,
													 const double dt,
													 const Eigen::MatrixXd &displacement,
													 const Eigen::MatrixXd &displacement_prev,
													 utils::SpareMatrixCache &mat_cache,
													 StiffnessMatrix &hessian) const
		{
			if (assembler == "SaintVenant")
				saint_venant_elasticity_->assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, dt, displacement, displacement_prev, mat_cache, hessian);
			else if (assembler == "NeoHookean")
				neo_hookean_elasticity_->assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, dt, displacement, displacement_prev, mat_cache, hessian);
			else if (assembler == "MooneyRivlin")
				mooney_rivlin_elasticity_->assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, dt, displacement, displacement_prev, mat_cache, hessian);
			else if (assembler == "MultiModels")
				multi_models_elasticity_->assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, dt, displacement, displacement_prev, mat_cache, hessian);
			else if (assembler == "Damping")
				damping_->assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, dt, displacement, displacement_prev, mat_cache, hessian);
			else if (assembler == "NavierStokesPicard")
				navier_stokes_velocity_picard_->assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, dt, displacement, displacement_prev, mat_cache, hessian);
			else if (assembler == "NavierStokes")
				navier_stokes_velocity_->assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, dt, displacement, displacement_prev, mat_cache, hessian);
			else if (assembler == "LinearElasticity")
				linear_elasticity_energy_->assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, dt, displacement, displacement_prev, mat_cache, hessian);

			else if (assembler == "UnconstrainedOgden")
				unconstrained_ogden_elasticity_->assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, dt, displacement, displacement_prev, mat_cache, hessian);
			else if (assembler == "IncompressibleOgden")
				incompressible_ogden_elasticity_->assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, dt, displacement, displacement_prev, mat_cache, hessian);
			else
				return;
		}

		void AssemblerUtils::compute_scalar_value(const std::string &assembler,
												  const int el_id,
												  const ElementBases &bs,
												  const ElementBases &gbs,
												  const Eigen::MatrixXd &local_pts,
												  const Eigen::MatrixXd &fun,
												  std::vector<NamedMatrix> &result) const
		{
			result.resize(0);

			if (assembler == "Mass" || assembler == "Laplacian" || assembler == "Helmholtz" || assembler == "Bilaplacian")
				return;

			if (assembler == "Stokes" || assembler == "OperatorSplitting" || assembler == "NavierStokes")
				return;

			Eigen::MatrixXd tmp;

			if (assembler == "LinearElasticity")
				linear_elasticity_->
					.compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, tmp);
			else if (assembler == "HookeLinearElasticity")
				hooke_linear_elasticity_->
					.compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, tmp);

			else if (assembler == "SaintVenant")
				saint_venant_elasticity_->
					.compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, tmp);
			else if (assembler == "NeoHookean")
				neo_hookean_elasticity_->
					.compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, tmp);
			else if (assembler == "MooneyRivlin")
				mooney_rivlin_elasticity_->
					.compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, tmp);
			else if (assembler == "MultiModels")
				multi_models_elasticity_->
					.compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, tmp);
			else if (assembler == "UnconstrainedOgden")
				unconstrained_ogden_elasticity_->
					.compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, tmp);
			else if (assembler == "IncompressibleOgden")
				incompressible_ogden_elasticity_->
					.compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, tmp);

			else if (assembler == "IncompressibleLinearElasticity")
				incompressible_lin_elast_displacement_->
					.compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, tmp);
			else
			{
				logger().warn("{} not found, fallback to default", assembler);
				assert(false);
				return;
			}

			result.emplace_back("von_mises", tmp);
		}

		void AssemblerUtils::compute_tensor_value(const std::string &assembler,
												  const int el_id,
												  const ElementBases &bs,
												  const ElementBases &gbs,
												  const Eigen::MatrixXd &local_pts,
												  const Eigen::MatrixXd &fun,
												  std::vector<NamedMatrix> &result) const
		{
			result.resize(0);
			if (assembler == "Mass" || assembler == "Laplacian" || assembler == "Helmholtz" || assembler == "Bilaplacian")
				return;

			if (assembler == "Stokes" || assembler == "OperatorSplitting" || assembler == "NavierStokes")
				return;

			Eigen::MatrixXd cauchy, pk1, pk2, F;

			if (assembler == "LinearElasticity")
			{
				linear_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::CAUCHY, cauchy);
				linear_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK1, pk1);
				linear_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK2, pk2);
				linear_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::F, F);
			}
			else if (assembler == "HookeLinearElasticity")
			{
				hooke_linear_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::CAUCHY, cauchy);
				hooke_linear_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK1, pk1);
				hooke_linear_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK2, pk2);
				hooke_linear_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::F, F);
			}

			else if (assembler == "SaintVenant")
			{
				saint_venant_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::CAUCHY, cauchy);
				saint_venant_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK1, pk1);
				saint_venant_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK2, pk2);
				saint_venant_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::F, F);
			}
			else if (assembler == "NeoHookean")
			{
				neo_hookean_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::CAUCHY, cauchy);
				neo_hookean_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK1, pk1);
				neo_hookean_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK2, pk2);
				neo_hookean_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::F, F);
			}
			else if (assembler == "MooneyRivlin")
			{
				mooney_rivlin_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::CAUCHY, cauchy);
				mooney_rivlin_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK1, pk1);
				mooney_rivlin_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK2, pk2);
				mooney_rivlin_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::F, F);
			}
			else if (assembler == "MultiModels")
			{
				multi_models_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::CAUCHY, cauchy);
				multi_models_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK1, pk1);
				multi_models_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK2, pk2);
				multi_models_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::F, F);
			}
			else if (assembler == "UnconstrainedOgden")
			{
				unconstrained_ogden_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::CAUCHY, cauchy);
				unconstrained_ogden_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK1, pk1);
				unconstrained_ogden_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK2, pk2);
				unconstrained_ogden_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::F, F);
			}
			else if (assembler == "IncompressibleOgden")
			{
				incompressible_ogden_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::CAUCHY, cauchy);
				incompressible_ogden_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK1, pk1);
				incompressible_ogden_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK2, pk2);
				incompressible_ogden_elasticity_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::F, F);
			}
			else if (assembler == "IncompressibleLinearElasticity")
			{
				incompressible_lin_elast_displacement_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::CAUCHY, cauchy);
				incompressible_lin_elast_displacement_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK1, pk1);
				incompressible_lin_elast_displacement_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK2, pk2);
				incompressible_lin_elast_displacement_->
					.compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::F, F);
			}

			else
			{
				logger().warn("{} not found, fallback to default", assembler);
				assert(false);
				return;
			}

			result.emplace_back("cauchy_stess", cauchy);
			result.emplace_back("pk1_stess", pk1);
			result.emplace_back("pk2_stess", pk2);
			result.emplace_back("F", F);
		}

		VectorNd AssemblerUtils::compute_rhs(const std::string &assembler, const AutodiffHessianPt &pt) const
		{
			if (assembler == "Laplacian")
				return laplacian_->compute_rhs(pt);
			else if (assembler == "Helmholtz")
				return helmholtz_->compute_rhs(pt);
			else if (assembler == "Bilaplacian")
				return bilaplacian_main_->compute_rhs(pt);

			else if (assembler == "LinearElasticity")
				return linear_elasticity_->compute_rhs(pt);
			else if (assembler == "HookeLinearElasticity")
				return hooke_linear_elasticity_->compute_rhs(pt);

			else if (assembler == "SaintVenant")
				return saint_venant_elasticity_->compute_rhs(pt);
			else if (assembler == "NeoHookean")
				return neo_hookean_elasticity_->compute_rhs(pt);
			else if (assembler == "MooneyRivlin")
				return mooney_rivlin_elasticity_->compute_rhs(pt);
			else if (assembler == "MultiModels")
				return multi_models_elasticity_->compute_rhs(pt);
			else if (assembler == "UnconstrainedOgden")
				return unconstrained_ogden_elasticity_->compute_rhs(pt);
			else if (assembler == "IncompressibleOgden")
				return incompressible_ogden_elasticity_->compute_rhs(pt);

			else if (assembler == "Stokes" || assembler == "OperatorSplitting")
				return stokes_velocity_->compute_rhs(pt);
			else if (assembler == "NavierStokes")
				return navier_stokes_velocity_->compute_rhs(pt);
			else if (assembler == "IncompressibleLinearElasticity")
				return incompressible_lin_elast_displacement_->compute_rhs(pt);

			else
			{
				logger().warn("{} not found, fallback to default", assembler);

				assert(false);
				return laplacian_->compute_rhs(pt);
			}
		}

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		AssemblerUtils::local_assemble(const std::string &assembler, const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
		{
			if (assembler == "Laplacian")
				return laplacian_->assemble(LinearAssemblerData(vals, i, j, da));
			else if (assembler == "Helmholtz")
				return helmholtz_->assemble(LinearAssemblerData(vals, i, j, da));
			else if (assembler == "Bilaplacian")
				return bilaplacian_main_->assemble(LinearAssemblerData(vals, i, j, da));

			else if (assembler == "LinearElasticity")
				return linear_elasticity_->assemble(LinearAssemblerData(vals, i, j, da));
			else if (assembler == "HookeLinearElasticity")
				return hooke_linear_elasticity_->assemble(LinearAssemblerData(vals, i, j, da));

			// else if(assembler == "SaintVenant")
			// 	return saint_venant_elasticity_->assemble(LinearAssemblerData(vals, i, j, da));
			// else if(assembler == "NeoHookean")
			// 	return neo_hookean_elasticity_->assemble(LinearAssemblerData(vals, i, j, da));
			// else if(assembler == "MultiModels")
			// 	return multi_models_elasticity_->assemble(LinearAssemblerData(vals, i, j, da));
			// else if(assembler == "UnconstrainedOgden")
			// 	return unconstrained_ogden_elasticity_->assemble(LinearAssemblerData(vals, i, j, da));
			// else if(assembler == "IncompressibleOgden")
			// 	return incompressible_ogden_elasticity_->assemble(LinearAssemblerData(vals, i, j, da));

			else if (assembler == "Stokes" || assembler == "OperatorSplitting")
				return stokes_velocity_->assemble(LinearAssemblerData(vals, i, j, da));
			else if (assembler == "IncompressibleLinearElasticity")
				return incompressible_lin_elast_displacement_->assemble(LinearAssemblerData(vals, i, j, da));

			else
			{
				logger().warn("{} not found, fallback to default", assembler);

				assert(false);
				return laplacian_->assemble(LinearAssemblerData(vals, i, j, da));
			}
		}

		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> AssemblerUtils::kernel(const std::string &assembler, const int dim, const AutodiffGradPt &rvect, const AutodiffScalarGrad &r) const
		{
			if (assembler == "Laplacian")
				return laplacian_->kernel(dim, r);
			else if (assembler == "Helmholtz")
				return helmholtz_->kernel(dim, r);
			else if (assembler == "LinearElasticity")
				return linear_elasticity_->kernel(dim, rvect);
			// else if(assembler == "HookeLinearElasticity")
			// 	return hooke_linear_elasticity_->kernel(dim, r);

			// else if(assembler == "SaintVenant")
			// 	return saint_venant_elasticity_->kernel(dim, r);
			// else if(assembler == "NeoHookean")
			// 	return neo_hookean_elasticity_->kernel(dim, r);
			// else if(assembler == "MultiModels")
			// 	return multi_models_elasticity_->kernel(dim, r);
			// else if(assembler == "UnconstrainedOgden")
			// 	return unconstrained_ogden_elasticity_->kernel(dim, r);
			// else if(assembler == "IncompressibleOgden")
			// 	return incompressible_ogden_elasticity_->kernel(dim, r);

			else
			{
				logger().warn("{} not found, fallback to default", assembler);

				assert(false);
				return laplacian_->kernel(dim, r);
			}
		}

		void AssemblerUtils::set_size(const std::string &assembler, const int dim)
		{
			int size = dim;
			if (assembler == "Helmholtz" || assembler == "Laplacian")
				size = 1;
			mass_mat_->set_size(size);
			mass_mat_no_density_->set_size(size);

			linear_elasticity_->set_size(dim);
			linear_elasticity_energy_->set_size(dim);
			hooke_linear_elasticity_->set_size(dim);

			saint_venant_elasticity_->set_size(dim);
			neo_hookean_elasticity_->set_size(dim);
			mooney_rivlin_elasticity_->set_size(dim);
			multi_models_elasticity_->set_size(dim);
			unconstrained_ogden_elasticity_->set_size(dim);
			incompressible_ogden_elasticity_->set_size(dim);

			damping_->set_size(dim);

			incompressible_lin_elast_displacement_->set_size(dim);
			incompressible_lin_elast_mixed_->set_size(dim);
			incompressible_lin_elast_pressure_->set_size(dim);

			stokes_velocity_->set_size(dim);
			stokes_mixed_->set_size(dim);
			// stokes_pressure_->set_size(dim);

			navier_stokes_velocity_->set_size(dim);
			navier_stokes_velocity_picard_->set_size(dim);
		}

		void AssemblerUtils::init_multimodels(const std::vector<std::string> &materials)
		{
			multi_models_elasticity_->init_multimodels(materials);
		}

		void AssemblerUtils::set_materials(const std::vector<int> &body_ids, const json &body_params)
		{
			if (!body_params.is_array())
			{
				this->add_multimaterial(0, body_params);
				return;
			}

			std::map<int, json> materials;
			for (int i = 0; i < body_params.size(); ++i)
			{
				json mat = body_params[i];
				json id = mat["id"];
				if (id.is_array())
				{
					for (int j = 0; j < id.size(); ++j)
						materials[id[j]] = mat;
				}
				else
				{
					const int mid = id;
					materials[mid] = mat;
				}
			}

			std::set<int> missing;

			std::map<int, int> body_element_count;
			std::vector<int> eid_to_eid_in_body(body_ids.size());
			for (int e = 0; e < body_ids.size(); ++e)
			{
				const int bid = body_ids[e];
				body_element_count.try_emplace(bid, 0);
				eid_to_eid_in_body[e] = body_element_count[bid]++;
			}

			for (int e = 0; e < body_ids.size(); ++e)
			{
				const int bid = body_ids[e];
				const auto it = materials.find(bid);
				if (it == materials.end())
				{
					missing.insert(bid);
					continue;
				}

				const json &tmp = it->second;
				this->add_multimaterial(e, tmp);
			}

			for (int bid : missing)
			{
				logger().warn("Missing material parameters for body {}", bid);
			}
		}

		void AssemblerUtils::add_multimaterial(const int index, const json &params)
		{
			mass_mat_->add_multimaterial(index, params);

			laplacian_->add_multimaterial(index, params);
			helmholtz_->add_multimaterial(index, params);

			bilaplacian_main_->add_multimaterial(index, params);
			bilaplacian_mixed_->add_multimaterial(index, params);
			bilaplacian_aux_->add_multimaterial(index, params);

			linear_elasticity_->add_multimaterial(index, params);
			linear_elasticity_energy_->add_multimaterial(index, params);
			hooke_linear_elasticity_->add_multimaterial(index, params);

			saint_venant_elasticity_->add_multimaterial(index, params);
			neo_hookean_elasticity_->add_multimaterial(index, params);
			mooney_rivlin_elasticity_->add_multimaterial(index, params);
			multi_models_elasticity_->add_multimaterial(index, params);
			unconstrained_ogden_elasticity_->add_multimaterial(index, params);
			incompressible_ogden_elasticity_->add_multimaterial(index, params);

			damping_->add_multimaterial(index, params);

			stokes_velocity_->add_multimaterial(index, params);
			stokes_mixed_->add_multimaterial(index, params);
			stokes_pressure_->add_multimaterial(index, params);

			navier_stokes_velocity_->add_multimaterial(index, params);
			navier_stokes_velocity_picard_->add_multimaterial(index, params);

			incompressible_lin_elast_displacement_->add_multimaterial(index, params);
			incompressible_lin_elast_mixed_->add_multimaterial(index, params);
			incompressible_lin_elast_pressure_->add_multimaterial(index, params);
		}

		void AssemblerUtils::merge_mixed_matrices(
			const int n_bases, const int n_pressure_bases, const int problem_dim, const bool add_average,
			const StiffnessMatrix &velocity_stiffness, const StiffnessMatrix &mixed_stiffness, const StiffnessMatrix &pressure_stiffness,
			StiffnessMatrix &stiffness)
		{
			assert(velocity_stiffness.rows() == velocity_stiffness.cols());
			assert(velocity_stiffness.rows() == n_bases * problem_dim);

			assert(mixed_stiffness.size() == 0 || mixed_stiffness.rows() == n_bases * problem_dim);
			assert(mixed_stiffness.size() == 0 || mixed_stiffness.cols() == n_pressure_bases);

			assert(pressure_stiffness.size() == 0 || pressure_stiffness.rows() == n_pressure_bases);
			assert(pressure_stiffness.size() == 0 || pressure_stiffness.cols() == n_pressure_bases);

			const int avg_offset = add_average ? 1 : 0;

			std::vector<Eigen::Triplet<double>> blocks;
			blocks.reserve(velocity_stiffness.nonZeros() + 2 * mixed_stiffness.nonZeros() + pressure_stiffness.nonZeros() + 2 * avg_offset * velocity_stiffness.rows());

			for (int k = 0; k < velocity_stiffness.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(velocity_stiffness, k); it; ++it)
				{
					blocks.emplace_back(it.row(), it.col(), it.value());
				}
			}

			for (int k = 0; k < mixed_stiffness.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(mixed_stiffness, k); it; ++it)
				{
					blocks.emplace_back(it.row(), n_bases * problem_dim + it.col(), it.value());
					blocks.emplace_back(it.col() + n_bases * problem_dim, it.row(), it.value());
				}
			}

			for (int k = 0; k < pressure_stiffness.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(pressure_stiffness, k); it; ++it)
				{
					blocks.emplace_back(n_bases * problem_dim + it.row(), n_bases * problem_dim + it.col(), it.value());
				}
			}

			if (add_average)
			{
				const double val = 1.0 / n_pressure_bases;
				for (int i = 0; i < n_pressure_bases; ++i)
				{
					blocks.emplace_back(n_bases * problem_dim + i, n_bases * problem_dim + n_pressure_bases, val);
					blocks.emplace_back(n_bases * problem_dim + n_pressure_bases, n_bases * problem_dim + i, val);
				}
			}

			stiffness.resize(n_bases * problem_dim + n_pressure_bases + avg_offset, n_bases * problem_dim + n_pressure_bases + avg_offset);
			stiffness.setFromTriplets(blocks.begin(), blocks.end());
			stiffness.makeCompressed();

			// static int c = 0;
			// Eigen::saveMarket(stiffness, "stiffness.txt");
			// Eigen::saveMarket(velocity_stiffness, "velocity_stiffness.txt");
			// Eigen::saveMarket(mixed_stiffness, "mixed_stiffness.txt");
			// Eigen::saveMarket(pressure_stiffness, "pressure_stiffness.txt");
		}

		int AssemblerUtils::quadrature_order(const std::string &assembler, const int basis_degree, const BasisType &b_type, const int dim)
		{
			if (assembler == "Mass")
			{
				if (b_type == BasisType::SIMPLEX_LAGRANGE || b_type == BasisType::CUBE_LAGRANGE)
					return std::max(basis_degree * 2, 1);
				else
					return basis_degree * 2 + 1;
			}
			else if (assembler == "NavierStokes")
			{
				if (b_type == BasisType::SIMPLEX_LAGRANGE)
					return std::max((basis_degree - 1) + basis_degree, 1);
				else if (b_type == BasisType::CUBE_LAGRANGE)
					return std::max(basis_degree * 2, 1);
				else
					return basis_degree * 2 + 1;
			}
			else
			{
				if (b_type == BasisType::SIMPLEX_LAGRANGE)
					return std::max((basis_degree - 1) * 2, 1);
				else if (b_type == BasisType::CUBE_LAGRANGE)
					return std::max(basis_degree * 2, 1);
				else
					return (basis_degree - 1) * 2 + 1;
			}
		}

		std::map<std::string, AssemblerUtils::ParamFunc> AssemblerUtils::parameters(const std::string &assembler) const
		{
			std::map<std::string, ParamFunc> res;

			// "Laplacian" "Bilaplacian" "Damping" "MultiModels"

			if (assembler == "Helmholtz")
			{
				const double k = helmholtz_->k();
				res["k"] = [k](const RowVectorNd &, const RowVectorNd &, double, int) { return k; };
			}
			else if (assembler == "LinearElasticity" || assembler == "IncompressibleLinearElasticity" || assembler == "NeoHookean")
			{
				const auto &params = linear_elasticity_->lame_params();
				const int size = linear_elasticity_->size();

				res["lambda"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
					double lambda, mu;

					params.lambda_mu(uv, p, e, lambda, mu);
					return lambda;
				};

				res["mu"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
					double lambda, mu;

					params.lambda_mu(uv, p, e, lambda, mu);
					return mu;
				};

				res["E"] = [&params, size](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
					double lambda, mu;
					params.lambda_mu(uv, p, e, lambda, mu);

					if (size == 3)
						return mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
					else
						return 2 * mu * (2.0 * lambda + 2.0 * mu) / (lambda + 2.0 * mu);
				};

				res["nu"] = [&params, size](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
					double lambda, mu;

					params.lambda_mu(uv, p, e, lambda, mu);

					if (size == 3)
						return lambda / (2.0 * (lambda + mu));
					else
						return lambda / (lambda + 2.0 * mu);
				};
			}
			else if (assembler == "HookeLinearElasticity" || assembler == "SaintVenant")
			{
				const auto &elast_tensor = hooke_linear_elasticity_->elasticity_tensor();
				const int size = hooke_linear_elasticity_->size() == 2 ? 3 : 6;

				for (int i = 0; i < size; ++i)
				{
					for (int j = i; j < size; ++j)
					{
						res[fmt::format("C_{}{}", i, j)] = [&elast_tensor, i, j](const RowVectorNd &, const RowVectorNd &, double, int) {
							return elast_tensor(i, j);
						};
					}
				}
			}
			else if (assembler == "MooneyRivlin")
			{
				// NOTE: This assumes mooney_rivlin_elasticity_ will stay alive

				const auto &c1 = mooney_rivlin_elasticity_->c1();
				const auto &c2 = mooney_rivlin_elasticity_->c2();
				const auto &k = mooney_rivlin_elasticity_->k();

				res["c1"] = [&c1](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
					return c1(p, t, e);
				};

				res["c2"] = [&c2](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
					return c2(p, t, e);
				};

				res["k"] = [&k](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
					return k(p, t, e);
				};
			}
			else if (assembler == "UnconstrainedOgden")
			{
				// NOTE: This assumes unconstrained_ogden_elasticity_ will stay alive

				const auto &alphas = unconstrained_ogden_elasticity_->alphas();
				const auto &mus = unconstrained_ogden_elasticity_->mus();
				const auto &Ds = unconstrained_ogden_elasticity_->Ds();

				for (int i = 0; i < alphas.size(); ++i)
					res[fmt::format("alpha_{}", i)] = [&alphas, i](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
						return alphas[i](p, t, e);
					};

				for (int i = 0; i < mus.size(); ++i)
					res[fmt::format("mu_{}", i)] = [&mus, i](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
						return mus[i](p, t, e);
					};

				for (int i = 0; i < Ds.size(); ++i)
					res[fmt::format("D_{}", i)] = [&Ds, i](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
						return Ds[i](p, t, e);
					};
			}
			else if (assembler == "IncompressibleOgden")
			{
				// NOTE: This assumes incompressible_ogden_elasticity_.local_assembler() will stay alive

				const auto &coefficients = incompressible_ogden_elasticity_.local_assembler().coefficients();
				const auto &expoenents = incompressible_ogden_elasticity_.local_assembler().expoenents();
				const auto &k = incompressible_ogden_elasticity_.local_assembler().bulk_modulus();

				for (int i = 0; i < coefficients.size(); ++i)
					res[fmt::format("c_{}", i)] = [&coefficients, i](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
						return coefficients[i](p, t, e);
					};

				for (int i = 0; i < expoenents.size(); ++i)
					res[fmt::format("m_{}", i)] = [&expoenents, i](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
						return expoenents[i](p, t, e);
					};

				res["k"] = [&k](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
					return k(p, t, e);
				};
			}
			else if (assembler == "IncompressibleOgden")
			{
				// NOTE: This assumes incompressible_ogden_elasticity_ will stay alive

				const Eigen::VectorXd &coefficients = incompressible_ogden_elasticity_->coefficients();
				const Eigen::VectorXd &expoenents = incompressible_ogden_elasticity_->expoenents();
				const double &k = incompressible_ogden_elasticity_->bulk_modulus();

				for (int i = 0; i < coefficients.size(); ++i)
					res[fmt::format("c_{}", i)] = [&coefficients, i](const RowVectorNd &, const RowVectorNd &, double, int) { return coefficients[i]; };

				for (int i = 0; i < expoenents.size(); ++i)
					res[fmt::format("m_{}", i)] = [&expoenents, i](const RowVectorNd &, const RowVectorNd &, double, int) { return expoenents[i]; };

				res["k"] = [&k](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
					return k;
				};
			}
			else if (assembler == "Stokes" || assembler == "NavierStokes" || assembler == "OperatorSplitting")
			{
				const double nu = stokes_velocity_->viscosity();
				res["viscosity"] = [nu](const RowVectorNd &, const RowVectorNd &, double, int) { return nu; };
			}

			return res;
		}
	} // namespace assembler
} // namespace polyfem
