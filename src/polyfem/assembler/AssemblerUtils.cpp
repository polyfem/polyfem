#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <unsupported/Eigen/SparseExtra>

namespace polyfem
{
	using namespace basis;
	using namespace utils;

	namespace assembler
	{
		AssemblerUtils::AssemblerUtils()
		{
		}

		std::vector<std::string> AssemblerUtils::scalar_assemblers() { return {"Laplacian", "Helmholtz", "Bilaplacian"}; }
		std::vector<std::string> AssemblerUtils::tensor_assemblers()
		{
			return {
				"LinearElasticity",
				"HookeLinearElasticity",
				"SaintVenant",
				"NeoHookean",
				"MultiModels",
				// "Ogden",
				"Stokes",
				"NavierStokes",
				"IncompressibleLinearElasticity"};
		}

		bool AssemblerUtils::is_scalar(const std::string &assembler)
		{
			return assembler == "Laplacian" || assembler == "Helmholtz" || assembler == "Bilaplacian";
		}

		bool AssemblerUtils::is_fluid(const std::string &assembler)
		{
			return assembler == "Stokes" || assembler == "NavierStokes" || assembler == "OperatorSplitting";
		}

		bool AssemblerUtils::is_tensor(const std::string &assembler)
		{
			return assembler == "LinearElasticity" || assembler == "HookeLinearElasticity" || assembler == "SaintVenant" || assembler == "NeoHookean" || assembler == "MultiModels" /*|| assembler == "Ogden"*/ || assembler == "Stokes" || assembler == "IncompressibleLinearElasticity" || assembler == "NavierStokes" || assembler == "OperatorSplitting";
		}
		bool AssemblerUtils::is_mixed(const std::string &assembler)
		{
			return assembler == "OperatorSplitting" || assembler == "Stokes" || assembler == "IncompressibleLinearElasticity" || assembler == "Bilaplacian" || assembler == "NavierStokes";
		}

		bool AssemblerUtils::is_solution_displacement(const std::string &assembler)
		{
			return assembler == "LinearElasticity" || assembler == "HookeLinearElasticity" || assembler == "SaintVenant" || assembler == "NeoHookean" || assembler == "MultiModels" || /*assembler == "Ogden" ||*/
				   assembler == "IncompressibleLinearElasticity";
		}

		bool AssemblerUtils::is_linear(const std::string &assembler)
		{
			return assembler != "SaintVenant" && assembler != "NeoHookean" && assembler != "NavierStokes" && assembler != "MultiModels" /*&& assembler != "Ogden"*/;
		}

		void AssemblerUtils::assemble_problem(const std::string &assembler,
											  const bool is_volume,
											  const int n_basis,
											  const std::vector<ElementBases> &bases,
											  const std::vector<ElementBases> &gbases,
											  const AssemblyValsCache &cache,
											  StiffnessMatrix &stiffness) const
		{
			if (assembler == "Helmholtz")
				helmholtz_.assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			else if (assembler == "Laplacian")
				laplacian_.assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			else if (assembler == "Bilaplacian")
				bilaplacian_main_.assemble(is_volume, n_basis, bases, gbases, cache, stiffness);

			else if (assembler == "LinearElasticity")
				linear_elasticity_.assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			else if (assembler == "HookeLinearElasticity")
				hooke_linear_elasticity_.assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			else if (assembler == "Stokes" || assembler == "NavierStokes" || assembler == "OperatorSplitting")
				stokes_velocity_.assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			else if (assembler == "IncompressibleLinearElasticity")
				incompressible_lin_elast_displacement_.assemble(is_volume, n_basis, bases, gbases, cache, stiffness);

			else if (assembler == "SaintVenant")
				return;
			else if (assembler == "NeoHookean")
				return;
			else if (assembler == "MultiModels")
				return;
			// else if (assembler == "NavierStokes")
			// return;
			//else if(assembler == "Ogden")
			//	return;
			else
			{
				logger().warn("{} not found, fallback to default", assembler);
				assert(false);
				laplacian_.assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			}
		}

		void AssemblerUtils::assemble_mass_matrix(const std::string &assembler,
												  const bool is_volume,
												  const int n_basis,
												  const Density &density,
												  const std::vector<ElementBases> &bases,
												  const std::vector<ElementBases> &gbases,
												  const AssemblyValsCache &cache,
												  StiffnessMatrix &mass) const
		{
			//TODO use cache
			if (assembler == "Helmholtz" || assembler == "Laplacian")
				mass_mat_assembler_.assemble(is_volume, 1, n_basis, density, bases, gbases, cache, mass);
			else
				mass_mat_assembler_.assemble(is_volume, is_volume ? 3 : 2, n_basis, density, bases, gbases, cache, mass);
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
			//TODO add cache
			if (assembler == "Bilaplacian")
				bilaplacian_mixed_.assemble(is_volume, n_psi_basis, n_phi_basis, psi_bases, phi_bases, gbases, psi_cache, phi_cache, stiffness);

			else if (assembler == "Stokes" || assembler == "NavierStokes" || assembler == "OperatorSplitting")
				stokes_mixed_.assemble(is_volume, n_psi_basis, n_phi_basis, psi_bases, phi_bases, gbases, psi_cache, phi_cache, stiffness);
			else if (assembler == "IncompressibleLinearElasticity")
				incompressible_lin_elast_mixed_.assemble(is_volume, n_psi_basis, n_phi_basis, psi_bases, phi_bases, gbases, psi_cache, phi_cache, stiffness);

			else
			{
				logger().warn("{} not found, fallback to default", assembler);
				assert(false);
				stokes_mixed_.assemble(is_volume, n_psi_basis, n_phi_basis, psi_bases, phi_bases, gbases, psi_cache, phi_cache, stiffness);
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
				bilaplacian_aux_.assemble(is_volume, n_basis, bases, gbases, cache, stiffness);

			else if (assembler == "Stokes" || assembler == "NavierStokes" || assembler == "OperatorSplitting")
				stokes_pressure_.assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			else if (assembler == "IncompressibleLinearElasticity")
				incompressible_lin_elast_pressure_.assemble(is_volume, n_basis, bases, gbases, cache, stiffness);

			else
			{
				logger().warn("{} not found, fallback to default", assembler);
				assert(false);
				stokes_pressure_.assemble(is_volume, n_basis, bases, gbases, cache, stiffness);
			}
		}

		double AssemblerUtils::assemble_energy(const std::string &assembler,
											   const bool is_volume,
											   const std::vector<ElementBases> &bases,
											   const std::vector<ElementBases> &gbases,
											   const AssemblyValsCache &cache,
											   const Eigen::MatrixXd &displacement) const
		{
			if (assembler == "SaintVenant")
				return saint_venant_elasticity_.assemble(is_volume, bases, gbases, cache, displacement);
			else if (assembler == "NeoHookean")
				return neo_hookean_elasticity_.assemble(is_volume, bases, gbases, cache, displacement);
			else if (assembler == "MultiModels")
				return multi_models_elasticity_.assemble(is_volume, bases, gbases, cache, displacement);

			//else if(assembler == "Ogden")
			//	return ogden_elasticity_.assemble(is_volume, bases, gbases, cache, displacement);
			else if (assembler == "LinearElasticity")
				return linear_elasticity_energy_.assemble(is_volume, bases, gbases, cache, displacement);
			else
				return 0;
		}

		void AssemblerUtils::assemble_energy_gradient(const std::string &assembler,
													  const bool is_volume,
													  const int n_basis,
													  const std::vector<ElementBases> &bases,
													  const std::vector<ElementBases> &gbases,
													  const AssemblyValsCache &cache,
													  const Eigen::MatrixXd &displacement,
													  Eigen::MatrixXd &grad) const
		{
			if (assembler == "SaintVenant")
				saint_venant_elasticity_.assemble_grad(is_volume, n_basis, bases, gbases, cache, displacement, grad);
			else if (assembler == "NeoHookean")
				neo_hookean_elasticity_.assemble_grad(is_volume, n_basis, bases, gbases, cache, displacement, grad);
			else if (assembler == "MultiModels")
				multi_models_elasticity_.assemble_grad(is_volume, n_basis, bases, gbases, cache, displacement, grad);

			else if (assembler == "NavierStokes")
				navier_stokes_velocity_.assemble_grad(is_volume, n_basis, bases, gbases, cache, displacement, grad);
			else if (assembler == "LinearElasticity")
				linear_elasticity_energy_.assemble_grad(is_volume, n_basis, bases, gbases, cache, displacement, grad);
			//else if(assembler == "Ogden")
			//	ogden_elasticity_.assemble_grad(is_volume, n_basis, bases, gbases, cache, displacement, grad);
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
													 const Eigen::MatrixXd &displacement,
													 utils::SpareMatrixCache &mat_cache,
													 StiffnessMatrix &hessian) const
		{
			if (assembler == "SaintVenant")
				saint_venant_elasticity_.assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, displacement, mat_cache, hessian);
			else if (assembler == "NeoHookean")
				neo_hookean_elasticity_.assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, displacement, mat_cache, hessian);
			else if (assembler == "MultiModels")
				multi_models_elasticity_.assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, displacement, mat_cache, hessian);

			else if (assembler == "NavierStokesPicard")
				navier_stokes_velocity_picard_.assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, displacement, mat_cache, hessian);
			else if (assembler == "NavierStokes")
				navier_stokes_velocity_.assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, displacement, mat_cache, hessian);
			else if (assembler == "LinearElasticity")
				linear_elasticity_energy_.assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, displacement, mat_cache, hessian);

			//else if(assembler == "Ogden")
			//	ogden_elasticity_.assemble_hessian(is_volume, n_basis, project_to_psd, bases, gbases, cache, displacement, mat_cache, hessian);
			else
				return;
		}

		void AssemblerUtils::compute_scalar_value(const std::string &assembler,
												  const int el_id,
												  const ElementBases &bs,
												  const ElementBases &gbs,
												  const Eigen::MatrixXd &local_pts,
												  const Eigen::MatrixXd &fun,
												  Eigen::MatrixXd &result) const
		{
			if (assembler == "Laplacian" || assembler == "Helmholtz" || assembler == "Bilaplacian")
				return;

			else if (assembler == "LinearElasticity")
				linear_elasticity_.local_assembler().compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, result);
			else if (assembler == "HookeLinearElasticity")
				hooke_linear_elasticity_.local_assembler().compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, result);

			else if (assembler == "SaintVenant")
				saint_venant_elasticity_.local_assembler().compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, result);
			else if (assembler == "NeoHookean")
				neo_hookean_elasticity_.local_assembler().compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, result);
			else if (assembler == "MultiModels")
				multi_models_elasticity_.local_assembler().compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, result);
			//else if(assembler == "Ogden")
			//	ogden_elasticity_.local_assembler().compute_von_mises_stresses(bs, gbs, local_pts, fun, result);

			else if (assembler == "Stokes" || assembler == "OperatorSplitting")
				stokes_velocity_.local_assembler().compute_norm_velocity(bs, gbs, local_pts, fun, result);
			else if (assembler == "NavierStokes")
				navier_stokes_velocity_.local_assembler().compute_norm_velocity(bs, gbs, local_pts, fun, result);

			else if (assembler == "IncompressibleLinearElasticity")
				incompressible_lin_elast_displacement_.local_assembler().compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, result);

			else
			{
				logger().warn("{} not found, fallback to default", assembler);
				assert(false);
				linear_elasticity_.local_assembler().compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, result);
			}
		}

		void AssemblerUtils::compute_tensor_value(const std::string &assembler,
												  const int el_id,
												  const ElementBases &bs,
												  const ElementBases &gbs,
												  const Eigen::MatrixXd &local_pts,
												  const Eigen::MatrixXd &fun,
												  Eigen::MatrixXd &result) const
		{
			if (assembler == "Laplacian" || assembler == "Helmholtz" || assembler == "Bilaplacian")
				return;

			else if (assembler == "LinearElasticity")
				linear_elasticity_.local_assembler().compute_stress_tensor(el_id, bs, gbs, local_pts, fun, result);
			else if (assembler == "HookeLinearElasticity")
				hooke_linear_elasticity_.local_assembler().compute_stress_tensor(el_id, bs, gbs, local_pts, fun, result);

			else if (assembler == "SaintVenant")
				saint_venant_elasticity_.local_assembler().compute_stress_tensor(el_id, bs, gbs, local_pts, fun, result);
			else if (assembler == "NeoHookean")
				neo_hookean_elasticity_.local_assembler().compute_stress_tensor(el_id, bs, gbs, local_pts, fun, result);
			else if (assembler == "MultiModels")
				multi_models_elasticity_.local_assembler().compute_stress_tensor(el_id, bs, gbs, local_pts, fun, result);
			//else if(assembler == "Ogden")
			//	ogden_elasticity_.local_assembler().compute_stress_tensor(bs, gbs, local_pts, fun, result);

			else if (assembler == "Stokes" || assembler == "OperatorSplitting") //WARNING stokes and NS dont have el_id
				stokes_velocity_.local_assembler().compute_stress_tensor(bs, gbs, local_pts, fun, result);
			else if (assembler == "NavierStokes")
				navier_stokes_velocity_.local_assembler().compute_stress_tensor(bs, gbs, local_pts, fun, result);
			else if (assembler == "IncompressibleLinearElasticity")
				incompressible_lin_elast_displacement_.local_assembler().compute_stress_tensor(el_id, bs, gbs, local_pts, fun, result);

			else
			{
				logger().warn("{} not found, fallback to default", assembler);
				assert(false);
				linear_elasticity_.local_assembler().compute_stress_tensor(el_id, bs, gbs, local_pts, fun, result);
			}
		}

		VectorNd AssemblerUtils::compute_rhs(const std::string &assembler, const AutodiffHessianPt &pt) const
		{
			if (assembler == "Laplacian")
				return laplacian_.local_assembler().compute_rhs(pt);
			else if (assembler == "Helmholtz")
				return helmholtz_.local_assembler().compute_rhs(pt);
			else if (assembler == "Bilaplacian")
				return bilaplacian_main_.local_assembler().compute_rhs(pt);

			else if (assembler == "LinearElasticity")
				return linear_elasticity_.local_assembler().compute_rhs(pt);
			else if (assembler == "HookeLinearElasticity")
				return hooke_linear_elasticity_.local_assembler().compute_rhs(pt);

			else if (assembler == "SaintVenant")
				return saint_venant_elasticity_.local_assembler().compute_rhs(pt);
			else if (assembler == "NeoHookean")
				return neo_hookean_elasticity_.local_assembler().compute_rhs(pt);
			else if (assembler == "MultiModels")
				return multi_models_elasticity_.local_assembler().compute_rhs(pt);
			//else if(assembler == "Ogden")
			//	return ogden_elasticity_.local_assembler().compute_rhs(pt);

			else if (assembler == "Stokes" || assembler == "OperatorSplitting")
				return stokes_velocity_.local_assembler().compute_rhs(pt);
			else if (assembler == "NavierStokes")
				return navier_stokes_velocity_.local_assembler().compute_rhs(pt);
			else if (assembler == "IncompressibleLinearElasticity")
				return incompressible_lin_elast_displacement_.local_assembler().compute_rhs(pt);

			else
			{
				logger().warn("{} not found, fallback to default", assembler);

				assert(false);
				return laplacian_.local_assembler().compute_rhs(pt);
			}
		}

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		AssemblerUtils::local_assemble(const std::string &assembler, const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
		{
			if (assembler == "Laplacian")
				return laplacian_.local_assembler().assemble(vals, i, j, da);
			else if (assembler == "Helmholtz")
				return helmholtz_.local_assembler().assemble(vals, i, j, da);
			else if (assembler == "Bilaplacian")
				return bilaplacian_main_.local_assembler().assemble(vals, i, j, da);

			else if (assembler == "LinearElasticity")
				return linear_elasticity_.local_assembler().assemble(vals, i, j, da);
			else if (assembler == "HookeLinearElasticity")
				return hooke_linear_elasticity_.local_assembler().assemble(vals, i, j, da);

			// else if(assembler == "SaintVenant")
			// 	return saint_venant_elasticity_.local_assembler().assemble(vals, i, j, da);
			// else if(assembler == "NeoHookean")
			// 	return neo_hookean_elasticity_.local_assembler().assemble(vals, i, j, da);
			// else if(assembler == "MultiModels")
			// 	return multi_models_elasticity_.local_assembler().assemble(vals, i, j, da);
			// else if(assembler == "Ogden")
			// 	return ogden_elasticity_.local_assembler().assemble(vals, i, j, da);

			else if (assembler == "Stokes" || assembler == "OperatorSplitting")
				return stokes_velocity_.local_assembler().assemble(vals, i, j, da);
			else if (assembler == "IncompressibleLinearElasticity")
				return incompressible_lin_elast_displacement_.local_assembler().assemble(vals, i, j, da);

			else
			{
				logger().warn("{} not found, fallback to default", assembler);

				assert(false);
				return laplacian_.local_assembler().assemble(vals, i, j, da);
			}
		}

		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> AssemblerUtils::kernel(const std::string &assembler, const int dim, const AutodiffGradPt &rvect, const AutodiffScalarGrad &r) const
		{
			if (assembler == "Laplacian")
				return laplacian_.local_assembler().kernel(dim, r);
			else if (assembler == "Helmholtz")
				return helmholtz_.local_assembler().kernel(dim, r);
			else if (assembler == "LinearElasticity")
				return linear_elasticity_.local_assembler().kernel(dim, rvect);
			// else if(assembler == "HookeLinearElasticity")
			// 	return hooke_linear_elasticity_.local_assembler().kernel(dim, r);

			// else if(assembler == "SaintVenant")
			// 	return saint_venant_elasticity_.local_assembler().kernel(dim, r);
			// else if(assembler == "NeoHookean")
			// 	return neo_hookean_elasticity_.local_assembler().kernel(dim, r);
			// else if(assembler == "MultiModels")
			// 	return multi_models_elasticity_.local_assembler().kernel(dim, r);
			// else if(assembler == "Ogden")
			// 	return ogden_elasticity_.local_assembler().kernel(dim, r);

			else
			{
				logger().warn("{} not found, fallback to default", assembler);

				assert(false);
				return laplacian_.local_assembler().kernel(dim, r);
			}
		}

		void AssemblerUtils::set_size(const int dim)
		{
			linear_elasticity_.local_assembler().set_size(dim);
			linear_elasticity_energy_.local_assembler().set_size(dim);
			hooke_linear_elasticity_.local_assembler().set_size(dim);

			saint_venant_elasticity_.local_assembler().set_size(dim);
			neo_hookean_elasticity_.local_assembler().set_size(dim);
			multi_models_elasticity_.local_assembler().set_size(dim);
			// ogden_elasticity_.local_assembler().set_size(dim);

			incompressible_lin_elast_displacement_.local_assembler().set_size(dim);
			incompressible_lin_elast_mixed_.local_assembler().set_size(dim);
			incompressible_lin_elast_pressure_.local_assembler().set_size(dim);

			stokes_velocity_.local_assembler().set_size(dim);
			stokes_mixed_.local_assembler().set_size(dim);
			// stokes_pressure_.local_assembler().set_size(dim);

			navier_stokes_velocity_.local_assembler().set_size(dim);
			navier_stokes_velocity_picard_.local_assembler().set_size(dim);
		}

		void AssemblerUtils::update_lame_params(const LameParameters &newParams)
		{
			linear_elasticity_.local_assembler().set_params(newParams);
			linear_elasticity_energy_.local_assembler().set_params(newParams);
			// hooke_linear_elasticity_.local_assembler().set_params(newParams);

			// saint_venant_elasticity_.local_assembler().set_params(newParams);
			neo_hookean_elasticity_.local_assembler().set_params(newParams);
			multi_models_elasticity_.local_assembler().set_params(newParams);
			// ogden_elasticity_.local_assembler().set_params(newParams);

			incompressible_lin_elast_displacement_.local_assembler().set_params(newParams);
			incompressible_lin_elast_mixed_.local_assembler().set_params(newParams);
			incompressible_lin_elast_pressure_.local_assembler().set_params(newParams);
		}

		void AssemblerUtils::init_multimodels(const std::vector<std::string> &materials)
		{
			multi_models_elasticity_.local_assembler().init_multimodels(materials);
		}

		void AssemblerUtils::add_multimaterial(const int index, const json &params)
		{
			laplacian_.local_assembler().add_multimaterial(index, params);
			helmholtz_.local_assembler().add_multimaterial(index, params);

			bilaplacian_main_.local_assembler().add_multimaterial(index, params);
			bilaplacian_mixed_.local_assembler().add_multimaterial(index, params);
			bilaplacian_aux_.local_assembler().add_multimaterial(index, params);

			linear_elasticity_.local_assembler().add_multimaterial(index, params);
			linear_elasticity_energy_.local_assembler().add_multimaterial(index, params);
			hooke_linear_elasticity_.local_assembler().add_multimaterial(index, params);

			saint_venant_elasticity_.local_assembler().add_multimaterial(index, params);
			neo_hookean_elasticity_.local_assembler().add_multimaterial(index, params);
			multi_models_elasticity_.local_assembler().add_multimaterial(index, params);
			// ogden_elasticity_.local_assembler().add_multimaterial(index, params);

			stokes_velocity_.local_assembler().add_multimaterial(index, params);
			stokes_mixed_.local_assembler().add_multimaterial(index, params);
			stokes_pressure_.local_assembler().add_multimaterial(index, params);

			navier_stokes_velocity_.local_assembler().add_multimaterial(index, params);
			navier_stokes_velocity_picard_.local_assembler().add_multimaterial(index, params);

			incompressible_lin_elast_displacement_.local_assembler().add_multimaterial(index, params);
			incompressible_lin_elast_mixed_.local_assembler().add_multimaterial(index, params);
			incompressible_lin_elast_pressure_.local_assembler().add_multimaterial(index, params);
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
	} // namespace assembler
} // namespace polyfem
