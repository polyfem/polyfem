#pragma once

#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/varforms/LinearElasticVarForm.hpp>
#include <polyfem/varforms/ScalarVarForm.hpp>

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
			return {
				form.mesh_.get(),
				form.assembler.get(),
				&form.bases,
				&form.geom_bases(),
				&form.total_local_boundary,
				form.n_bases,
				output_space.obstacle ? output_space.obstacle->n_vertices() : 0,
				form.root_path};
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
			return form.mass;
		}
	};
} // namespace polyfem::test
