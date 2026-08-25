#pragma once

#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/varforms/BilaplacianVarForm.hpp>
#include <polyfem/varforms/FluidVarForm.hpp>
#include <polyfem/varforms/LinearElasticVarForm.hpp>
#include <polyfem/varforms/NavierStokesFSIVarForm.hpp>
#include <polyfem/varforms/NonlinearElasticVarForm.hpp>
#include <polyfem/varforms/ScalarVarForm.hpp>
#include <polyfem/solver/forms/FSIInterfaceForm.hpp>

#include <stdexcept>

namespace polyfem::test
{
	struct VarFormDebugData
	{
		const mesh::Mesh *mesh = nullptr;
		const assembler::Assembler *assembler = nullptr;
		const std::vector<basis::ElementBases> *bases = nullptr;
		const std::vector<basis::ElementBases> *geometry_bases = nullptr;
		const std::vector<mesh::LocalBoundary> *total_local_boundary = nullptr;
		int n_bases = 0;
		int n_obstacle_vertices = 0;
		std::string root_path;
	};

	struct NavierStokesFSIDebugData
	{
		std::shared_ptr<solver::NavierStokesFSIForm> ale_form;
		std::shared_ptr<solver::FSIInterfaceForm> interface_form;
		std::shared_ptr<solver::NavierStokesFSIAveragePressureForm> average_pressure_form;
		const std::vector<basis::ElementBases> *pressure_bases = nullptr;
		const std::vector<basis::ElementBases> *mesh_displacement_bases = nullptr;
		const std::vector<basis::ElementBases> *geometry_bases = nullptr;
		const assembler::AssemblyValsCache *pressure_cache = nullptr;
		const assembler::AssemblyValsCache *mesh_displacement_cache = nullptr;
		std::shared_ptr<solver::NLProblem> problem;
		std::shared_ptr<varform::NonlinearElasticTransientVarForm> solid_varform;
		const mesh::Mesh *fluid_mesh = nullptr;
		const mesh::Mesh *solid_mesh = nullptr;
		int velocity_ndof = 0;
		int pressure_ndof = 0;
		int mesh_displacement_ndof = 0;
		int solid_displacement_ndof = 0;
		int solid_displacement_offset = 0;
		int fluid_multiplier_ndof = 0;
		int mesh_multiplier_ndof = 0;
		int fluid_multiplier_offset = 0;
		int mesh_multiplier_offset = 0;
		int average_pressure_offset = 0;
		int interface_size = 0;
		bool is_volume = false;
	};

	class VarFormTestAccess
	{
	public:
		static void prepare(varform::VarForm &form)
		{
			form.prepare();
		}

		static VarFormDebugData debug_data(const varform::VarForm &form)
		{
			const io::OutputSpace output_space = form.output_space();

			if (const auto *elastic = dynamic_cast<const varform::ElasticVarForm *>(&form))
			{
				return {
					output_space.mesh,
					elastic->primary_assembler_.get(),
					&elastic->space_.basis_list(),
					output_space.geometry_bases,
					output_space.total_local_boundary,
					elastic->space_.n_bases,
					output_space.obstacle ? output_space.obstacle->n_vertices() : 0,
					form.root_path};
			}

			if (const auto *fluid = dynamic_cast<const varform::FluidVarForm *>(&form))
			{
				return {
					output_space.mesh,
					fluid->primary_assembler_.get(),
					&fluid->space_.basis_list(),
					output_space.geometry_bases,
					output_space.total_local_boundary,
					fluid->space_.n_bases,
					0,
					form.root_path};
			}

			if (const auto *bilaplacian = dynamic_cast<const varform::BilaplacianVarForm *>(&form))
			{
				return {
					output_space.mesh,
					bilaplacian->primary_assembler_.get(),
					&bilaplacian->space_.basis_list(),
					output_space.geometry_bases,
					output_space.total_local_boundary,
					bilaplacian->space_.n_bases,
					0,
					form.root_path};
			}

			if (const auto *scalar = dynamic_cast<const varform::ScalarVarForm *>(&form))
			{
				return {
					output_space.mesh,
					scalar->primary_assembler_.get(),
					&scalar->space_.basis_list(),
					output_space.geometry_bases,
					output_space.total_local_boundary,
					scalar->space_.n_bases,
					0,
					form.root_path};
			}

			throw std::runtime_error("Unsupported VarForm test debug data request.");
		}

		static bool build_stiffness_mat(varform::VarForm &form, StiffnessMatrix &stiffness)
		{
			if (auto *linear_elastic = dynamic_cast<varform::LinearElasticVarForm *>(&form))
			{
				linear_elastic->build_stiffness_mat(stiffness);
				return true;
			}
			if (auto *scalar = dynamic_cast<varform::ScalarVarForm *>(&form))
			{
				scalar->build_stiffness_mat(stiffness);
				return true;
			}
			return false;
		}

		static const StiffnessMatrix &mass_matrix(const varform::VarForm &form)
		{
			if (const auto *elastic = dynamic_cast<const varform::ElasticVarForm *>(&form))
				return elastic->mass_;
			if (const auto *fluid = dynamic_cast<const varform::FluidVarForm *>(&form))
				return fluid->mass_;
			if (const auto *bilaplacian = dynamic_cast<const varform::BilaplacianVarForm *>(&form))
				return bilaplacian->mass_;
			if (const auto *scalar = dynamic_cast<const varform::ScalarVarForm *>(&form))
				return scalar->mass_;
			throw std::runtime_error("Unsupported VarForm test mass matrix request.");
		}

		static NavierStokesFSIDebugData navier_stokes_fsi_data(const varform::VarForm &form)
		{
			const auto *fsi = dynamic_cast<const varform::NavierStokesFSIVarForm *>(&form);
			if (fsi == nullptr)
				throw std::runtime_error("VarForm is not NavierStokesFSI.");
			const io::OutputSpace solid_output = fsi->solid_varform_
													 ? fsi->solid_varform_->output_space()
													 : io::OutputSpace{};
			return {
				fsi->ale_form_,
				fsi->interface_form_,
				fsi->average_pressure_form_,
				&fsi->pressure_space_.basis_list(),
				&fsi->mesh_displacement_space_.basis_list(),
				&fsi->space_.geometry_basis_list(),
				&fsi->pressure_ass_vals_cache_,
				&fsi->mesh_displacement_ass_vals_cache_,
				fsi->fsi_problem_,
				fsi->solid_varform_,
				fsi->mesh_.get(),
				solid_output.mesh,
				fsi->primary_ndof(),
				fsi->pressure_space_.n_bases,
				fsi->mesh_displacement_ndof(),
				fsi->solid_displacement_ndof(),
				fsi->solid_displacement_offset(),
				fsi->fluid_interface_multiplier_ndof(),
				fsi->mesh_interface_multiplier_ndof(),
				fsi->fluid_interface_multiplier_offset(),
				fsi->mesh_interface_multiplier_offset(),
				fsi->average_pressure_offset(),
				int(fsi->interface_2d_.size() + fsi->interface_3d_.size()),
				fsi->mesh_ != nullptr && fsi->mesh_->is_volume()};
		}
	};
} // namespace polyfem::test
