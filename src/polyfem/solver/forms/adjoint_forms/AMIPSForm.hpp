#pragma once

#include <polyfem/solver/forms/adjoint_forms/VariableToSimulation.hpp>
#include "VariableToSimulation.hpp"
#include <polyfem/State.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/AMIPSEnergy.hpp>

namespace polyfem::solver
{
	namespace
	{
		double triangle_jacobian(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3)
		{
			Eigen::VectorXd a = v2 - v1, b = v3 - v1;
			return a(0) * b(1) - b(0) * a(1);
		}

		double tet_determinant(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4)
		{
			Eigen::Matrix3d mat;
			mat.col(0) << v2 - v1;
			mat.col(1) << v3 - v1;
			mat.col(2) << v4 - v1;
			return mat.determinant();
		}

		void scaled_jacobian(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, Eigen::VectorXd &quality)
		{
			const int dim = F.cols() - 1;

			quality.setZero(F.rows());
			if (dim == 2)
			{
				for (int i = 0; i < F.rows(); i++)
				{
					Eigen::RowVector3d e0;
					e0(2) = 0;
					e0.head(2) = V.row(F(i, 2)) - V.row(F(i, 1));
					Eigen::RowVector3d e1;
					e1(2) = 0;
					e1.head(2) = V.row(F(i, 0)) - V.row(F(i, 2));
					Eigen::RowVector3d e2;
					e2(2) = 0;
					e2.head(2) = V.row(F(i, 1)) - V.row(F(i, 0));

					double l0 = e0.norm();
					double l1 = e1.norm();
					double l2 = e2.norm();

					double A = 0.5 * (e0.cross(e1)).norm();
					double Lmax = std::max(l0 * l1, std::max(l1 * l2, l0 * l2));

					quality(i) = 2 * A * (2 / sqrt(3)) / Lmax;
				}
			}
			else
			{
				for (int i = 0; i < F.rows(); i++)
				{
					Eigen::RowVector3d e0 = V.row(F(i, 1)) - V.row(F(i, 0));
					Eigen::RowVector3d e1 = V.row(F(i, 2)) - V.row(F(i, 1));
					Eigen::RowVector3d e2 = V.row(F(i, 0)) - V.row(F(i, 2));
					Eigen::RowVector3d e3 = V.row(F(i, 3)) - V.row(F(i, 0));
					Eigen::RowVector3d e4 = V.row(F(i, 3)) - V.row(F(i, 1));
					Eigen::RowVector3d e5 = V.row(F(i, 3)) - V.row(F(i, 2));

					double l0 = e0.norm();
					double l1 = e1.norm();
					double l2 = e2.norm();
					double l3 = e3.norm();
					double l4 = e4.norm();
					double l5 = e5.norm();

					double J = std::abs((e0.cross(e3)).dot(e2));

					double a1 = l0 * l2 * l3;
					double a2 = l0 * l1 * l4;
					double a3 = l1 * l2 * l5;
					double a4 = l3 * l4 * l5;

					double a = std::max({a1, a2, a3, a4, J});
					quality(i) = J * sqrt(2) / a;
				}
			}
		}

		bool is_flipped(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
		{
			if (F.cols() == 3)
			{
				for (int i = 0; i < F.rows(); i++)
					if (triangle_jacobian(V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2))) <= 0)
						return true;
			}
			else if (F.cols() == 4)
			{
				for (int i = 0; i < F.rows(); i++)
					if (tet_determinant(V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2)), V.row(F(i, 3))) <= 0)
						return true;
			}
			else
			{
				return true;
			}

			return false;
		}
	} // namespace

	class AMIPSForm : public AdjointForm
	{
	public:
		AMIPSForm(const std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulation, const State &state)
			: AdjointForm(variable_to_simulation),
			  state_(state)
		{
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
			X_init = utils::flatten(V);
			init_geom_bases_ = state_.geom_bases();
		}

		double value_unweighted(const Eigen::VectorXd &x) const override
		{
			Eigen::VectorXd X = get_updated_mesh_nodes(x);

			double energy = amips_energy_->assemble_energy(state_.mesh->is_volume(), init_geom_bases_, init_geom_bases_, init_ass_vals_cache_, 0, AdjointTools::map_primitive_to_node_order(state_, X - X_init), Eigen::VectorXd());

			return energy;
		}

		void compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
		{
			Eigen::VectorXd X = get_updated_mesh_nodes(x);

			Eigen::MatrixXd grad;
			amips_energy_->assemble_gradient(state_.mesh->is_volume(), state_.n_geom_bases, init_geom_bases_, init_geom_bases_, init_ass_vals_cache_, 0, AdjointTools::map_primitive_to_node_order(state_, X - X_init), Eigen::VectorXd(), grad); // grad wrt. gbases
			grad = AdjointTools::map_node_to_primitive_order(state_, grad);                                                                                                                                                                       // grad wrt. vertices

			assert(grad.cols() == 1);

			gradv.setZero(x.size());
			for (auto &p : variable_to_simulations_)
			{
				for (const auto &state : p->get_states())
					if (state.get() != &state_)
						continue;
				if (p->get_parameter_type() != ParameterType::Shape)
					continue;
				gradv += p->apply_parametrization_jacobian(grad, x);
			}
		}

		Eigen::MatrixXd compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const override
		{
			return Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size());
		}

		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			Eigen::VectorXd X = get_updated_mesh_nodes(x1);
			Eigen::MatrixXd V1 = utils::unflatten(X, state_.mesh->dimension());
			bool flipped = is_flipped(V1, F);

			if (flipped)
				logger().trace("Step flips elements.");

			return !flipped;
		}

	private:
		Eigen::VectorXd get_updated_mesh_nodes(const Eigen::VectorXd &x) const
		{
			Eigen::VectorXd X = X_init;

			for (auto &p : variable_to_simulations_)
			{
				for (const auto &state : p->get_states())
					if (state.get() != &state_)
						continue;
				if (p->get_parameter_type() != ParameterType::Shape)
					continue;
				auto state_variable = p->get_parametrization().eval(x);
				auto output_indexing = p->get_output_indexing(x);
				for (int i = 0; i < output_indexing.size(); ++i)
					X(output_indexing(i)) = state_variable(i);
			}

			return X;
		}

		const State &state_;

		Eigen::VectorXd X_init;
		Eigen::MatrixXi F;
		std::vector<polyfem::basis::ElementBases> init_geom_bases_;
		assembler::AssemblyValsCache init_ass_vals_cache_;

		std::shared_ptr<assembler::Assembler> amips_energy_;
	};
} // namespace polyfem::solver