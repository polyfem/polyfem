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

		inline bool is_step_valid_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			Eigen::MatrixXd V0;
			Eigen::MatrixXi F;
			state_.get_vf(V0, F);
			Eigen::MatrixXd V1 = utils::unflatten(x1, state_.mesh->dimension());
			bool flipped = is_flipped(V1, F);
			return !flipped;
		}

	private:
		std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations_;
		const State &state_;

		NLAssembler<GenericElastic<AMIPSEnergy>> amips_energy_;
	};
} // namespace polyfem::solver