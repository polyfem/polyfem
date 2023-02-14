#pragma once

#include <polyfem/solver/forms/ParametrizationForm.hpp>
#include "VariableToSimulation.hpp"
#include <polyfem/State.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/AMIPSEnergy.hpp>

namespace polyfem::solver
{
	class AMIPSForm : public ParametrizationForm
	{
	public:
		AMIPSForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const CompositeParametrization &parametrizations, const State &state, const json &args)
			: ParametrizationForm(parametrizations),
			  variable_to_simulations_(variable_to_simulations),
			  state_(state)
		{
			amips_energy_.local_assembler().set_size(state.mesh->dimension());

			json transform_params = {};
			transform_params["canonical_transformation"] = json::array();
			if (!state.mesh->is_volume())
			{
				Eigen::MatrixXd regular_tri(3, 3);
				regular_tri << 0, 0, 1,
					1, 0, 1,
					1. / 2., std::sqrt(3) / 2., 1;
				regular_tri.transposeInPlace();
				Eigen::MatrixXd regular_tri_inv = regular_tri.inverse();

				const auto &mesh2d = *dynamic_cast<mesh::Mesh2D *>(state.mesh.get());
				for (int e = 0; e < state.mesh->n_elements(); e++)
				{
					Eigen::MatrixXd transform;
					mesh2d.compute_face_jacobian(e, regular_tri_inv, transform);
					transform_params["canonical_transformation"].push_back(json({
						{
							transform(0, 0),
							transform(0, 1),
						},
						{
							transform(1, 0),
							transform(1, 1),
						},
					}));
				}
			}
			else
			{
				Eigen::MatrixXd regular_tet(4, 4);
				regular_tet << 0, 0, 0, 1,
					1, 0, 0, 1,
					1. / 2., std::sqrt(3) / 2., 0, 1,
					1. / 2., 1. / 2. / std::sqrt(3), std::sqrt(3) / 2., 1;
				regular_tet.transposeInPlace();
				Eigen::MatrixXd regular_tet_inv = regular_tet.inverse();

				const auto &mesh3d = *dynamic_cast<mesh::Mesh3D *>(state.mesh.get());
				for (int e = 0; e < state.mesh->n_elements(); e++)
				{
					Eigen::MatrixXd transform;
					mesh3d.compute_cell_jacobian(e, regular_tet_inv, transform);
					transform_params["canonical_transformation"].push_back(json({
						{
							transform(0, 0),
							transform(0, 1),
							transform(0, 2),
						},
						{
							transform(1, 0),
							transform(1, 1),
							transform(1, 2),
						},
						{
							transform(2, 0),
							transform(2, 1),
							transform(2, 2),
						},
					}));
				}
			}
			amips_energy_.local_assembler().add_multimaterial(0, transform_params);
		}

		inline double value_unweighted_with_param(const Eigen::VectorXd &x) const override
		{
			return amips_energy_.assemble(state_.mesh->is_volume(), state_.geom_bases(), state_.geom_bases(), state_.ass_vals_cache, 0, x, Eigen::VectorXd(), false);
		}

		inline void first_derivative_unweighted_with_param(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
		{
			Eigen::MatrixXd grad;
			amips_energy_.assemble_grad(state_.mesh->is_volume(), state_.n_bases, state_.geom_bases(), state_.geom_bases(), state_.ass_vals_cache, 0, x, Eigen::VectorXd(), grad);
			assert(grad.cols() == 1);
			gradv = grad;
		}

	private:
		std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations_;
		const State &state_;

		NLAssembler<GenericElastic<AMIPSEnergy>> amips_energy_;
	};
} // namespace polyfem::solver