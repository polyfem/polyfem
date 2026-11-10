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
	};

	struct ElasticFormDebugData
	{
		const mesh::Mesh &mesh;
		const varform::FESpace &space;
		std::vector<basis::ElementBases> &bases;
		const varform::VarFormBoundaryState &boundary;
		assembler::AssemblyValsCache &assembly_cache;
		assembler::AssemblyValsCache &mass_assembly_cache;
		const StiffnessMatrix &mass;
		const Eigen::MatrixXd &rhs;
		assembler::Assembler &assembler;
		assembler::Mass &mass_assembler;
		const assembler::Problem &problem;
		std::shared_ptr<assembler::RhsAssembler> rhs_assembler;
		QuadratureOrders boundary_samples;
		double average_mass;
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

		static ElasticFormDebugData elastic_form_data(varform::ElasticVarForm &form)
		{
			return {
				*form.mesh_, form.space_, *form.space_.bases, form.boundary_, form.ass_vals_cache_, form.mass_ass_vals_cache_,
				form.mass_, form.rhs_, *form.primary_assembler_, *form.mass_assembler_,
				*form.problem, form.rhs_assembler_, form.elastic_boundary_samples(), form.avg_mass_};
		}

		static void build_collision_mesh(varform::ElasticVarForm &form, ipc::CollisionMesh &collision_mesh)
		{
			const io::OutputSpace output = form.output_space();
			if (output.obstacle == nullptr)
				throw std::runtime_error("Elastic contact fixture requires obstacle storage.");
			varform::NonlinearElasticVarForm::build_collision_mesh(
				*form.mesh_, form.space_.n_bases, form.space_.basis_list(),
				form.space_.geometry_basis_list(), form.boundary_.total_local_boundary,
				*output.obstacle, form.args, form.resources_, form.space_.space_in_node_to_node,
				collision_mesh);
		}

		static std::shared_ptr<assembler::PressureAssembler> build_pressure_assembler(varform::ElasticVarForm &form)
		{
			const io::OutputSpace output = form.output_space();
			if (output.obstacle == nullptr)
				throw std::runtime_error("Elastic pressure fixture requires obstacle storage.");
			return std::make_shared<assembler::PressureAssembler>(
				*form.primary_assembler_, *form.mesh_, *output.obstacle,
				form.boundary_.local_pressure_boundary, form.boundary_.local_pressure_cavity,
				form.boundary_.boundary_nodes, form.elastic_primitive_to_node(), form.elastic_node_to_primitive(),
				form.space_.n_bases, form.mesh_->dimension(), form.space_.basis_list(),
				form.space_.geometry_basis_list(), *form.problem);
		}

		static void set_materials(varform::ElasticVarForm &form, assembler::Assembler &assembler)
		{
			form.set_materials(assembler, form.mesh_->dimension());
		}

		static void build_mesh_matrices(const varform::ElasticVarForm &form, Eigen::MatrixXd &vertices, Eigen::MatrixXi &elements)
		{
			form.build_mesh_matrices(vertices, elements);
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
					output_space.obstacle ? output_space.obstacle->n_vertices() : 0};
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
					0};
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
					0};
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
					0};
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
