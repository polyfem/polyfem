#pragma once

#include <polyfem/solver/forms/adjoint_forms/VariableToSimulation.hpp>
#include "VariableToSimulation.hpp"
#include <polyfem/State.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/AMIPSEnergy.hpp>

#include <polyfem/solver/AdjointTools.hpp>

namespace polyfem::solver
{
	class AMIPSForm : public AdjointForm
	{
	public:
		AMIPSForm(const VariableToSimulationGroup &variable_to_simulation, const State &state)
			: AdjointForm(variable_to_simulation),
			  state_(state)
		{
			logger().error("Don't use amips in the optimization!");

			amips_energy_ = assembler::AssemblerUtils::make_assembler("AMIPS");
			amips_energy_->set_size(state.mesh->dimension());

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
			transform_params["solve_displacement"] = true;
			amips_energy_->add_multimaterial(0, transform_params, state.units);

			Eigen::MatrixXd V;
			state_.get_vertices(V);
			state_.get_elements(F);
			X_rest = utils::flatten(V);
			rest_geom_bases_ = state_.geom_bases();
			rest_ass_vals_cache_.init(state_.mesh->is_volume(), rest_geom_bases_, rest_geom_bases_);
		}

		virtual std::string name() const override { return "AMIPS"; }

		double value_unweighted(const Eigen::VectorXd &x) const override
		{
			Eigen::VectorXd X = get_updated_mesh_nodes(x);

			return amips_energy_->assemble_energy(state_.mesh->is_volume(), rest_geom_bases_, rest_geom_bases_, rest_ass_vals_cache_, 0, 0, AdjointTools::map_primitive_to_node_order(state_, X - X_rest), Eigen::VectorXd());
		}

		void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
		{
			gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::Shape, &state_, x, [this, &x]() {
				const Eigen::VectorXd X = get_updated_mesh_nodes(x);
				Eigen::MatrixXd grad;
				amips_energy_->assemble_gradient(state_.mesh->is_volume(), state_.n_geom_bases, rest_geom_bases_, rest_geom_bases_, rest_ass_vals_cache_, 0, 0, AdjointTools::map_primitive_to_node_order(state_, X - X_rest), Eigen::VectorXd(), grad); // grad wrt. gbases
				return AdjointTools::map_node_to_primitive_order(state_, grad);
			});
		}

		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			Eigen::VectorXd X = get_updated_mesh_nodes(x1);
			Eigen::MatrixXd V1 = utils::unflatten(X, state_.mesh->dimension());
			bool flipped = AdjointTools::is_flipped(V1, F);

			if (flipped)
				adjoint_logger().trace("[{}] Step flips elements.", name());

			return !flipped;
		}

		/*
		void solution_changed(const Eigen::VectorXd &newX) override
		{
			Eigen::MatrixXd V;
			state_.get_vertices(V);
			X_rest = utils::flatten(V);
			rest_geom_bases_ = state_.geom_bases();
			rest_ass_vals_cache_.init(state_.mesh->is_volume(), rest_geom_bases_, rest_geom_bases_);
		}
		*/

	private:
		Eigen::VectorXd get_updated_mesh_nodes(const Eigen::VectorXd &x) const
		{
			Eigen::VectorXd X = X_rest;
			variable_to_simulations_.compute_state_variable(ParameterType::Shape, &state_, x, X);
			return X;
		}

		const State &state_;

		Eigen::VectorXd X_rest;
		Eigen::MatrixXi F;
		std::vector<polyfem::basis::ElementBases> rest_geom_bases_;
		assembler::AssemblyValsCache rest_ass_vals_cache_;

		std::shared_ptr<assembler::Assembler> amips_energy_;
	};
} // namespace polyfem::solver
