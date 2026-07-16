#pragma once

#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/varforms/BilaplacianVarForm.hpp>
#include <polyfem/varforms/FluidVarForm.hpp>
#include <polyfem/varforms/LinearElasticVarForm.hpp>
#include <polyfem/varforms/ScalarVarForm.hpp>

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
	};
} // namespace polyfem::test
