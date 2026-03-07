#include "AMIPSForm.hpp"

#include <polyfem/State.hpp>

#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/AMIPSEnergy.hpp>
#include <polyfem/utils/GeometryUtils.hpp>

namespace polyfem::solver
{
	namespace
	{
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
	} // namespace

	double MinJacobianForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const bool is_volume = state_.mesh->is_volume();
		double min_jacs = std::numeric_limits<double>::max();
		for (size_t e = 0; e < state_.geom_bases().size(); ++e)
		{
			if (state_.mesh->is_polytope(e))
				continue;

			const auto &gbasis = state_.geom_bases()[e];
			const int n_local_bases = int(gbasis.bases.size());

			quadrature::Quadrature quad;
			gbasis.compute_quadrature(quad);

			std::vector<assembler::AssemblyValues> tmp;

			Eigen::MatrixXd dx = Eigen::MatrixXd::Zero(quad.points.rows(), quad.points.cols());
			Eigen::MatrixXd dy = Eigen::MatrixXd::Zero(quad.points.rows(), quad.points.cols());
			Eigen::MatrixXd dz;
			if (is_volume)
				dz = Eigen::MatrixXd::Zero(quad.points.rows(), quad.points.cols());

			gbasis.evaluate_grads(quad.points, tmp);

			for (int j = 0; j < n_local_bases; ++j)
			{
				const basis::Basis &b = gbasis.bases[j];

				for (std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					dx += tmp[j].grad.col(0) * b.global()[ii].node * b.global()[ii].val;
					dy += tmp[j].grad.col(1) * b.global()[ii].node * b.global()[ii].val;
					if (is_volume)
						dz += tmp[j].grad.col(2) * b.global()[ii].node * b.global()[ii].val;
				}
			}

			for (long i = 0; i < dx.rows(); ++i)
			{
				if (is_volume)
				{
					Eigen::Matrix3d tmp;
					tmp << dx.row(i), dy.row(i), dz.row(i);
					min_jacs = std::min(min_jacs, tmp.determinant());
				}
				else
				{
					Eigen::Matrix2d tmp;
					tmp << dx.row(i), dy.row(i);
					min_jacs = std::min(min_jacs, tmp.determinant());
				}
			}
		}

		return min_jacs;
	}

	void MinJacobianForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		log_and_throw_adjoint_error("{} is not differentiable!", name());
	}

	AMIPSForm::AMIPSForm(const VariableToSimulationGroup &variable_to_simulation, const State &state)
		: AdjointForm(variable_to_simulation),
		  state_(state)
	{
		amips_energy_ = assembler::AssemblerUtils::make_assembler("AMIPSAutodiff");
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
		init_geom_bases_ = state_.geom_bases();
	}

	double AMIPSForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd X = get_updated_mesh_nodes(x);

		return amips_energy_->assemble_energy(state_.mesh->is_volume(), init_geom_bases_, init_geom_bases_, init_ass_vals_cache_, 0, 0, AdjointTools::map_primitive_to_node_order(state_, X - X_rest), Eigen::VectorXd());
	}

	void AMIPSForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::Shape, &state_, x, [this, &x]() {
			const Eigen::VectorXd X = get_updated_mesh_nodes(x);
			Eigen::MatrixXd grad;
			amips_energy_->assemble_gradient(state_.mesh->is_volume(), state_.n_geom_bases, init_geom_bases_, init_geom_bases_, init_ass_vals_cache_, 0, 0, AdjointTools::map_primitive_to_node_order(state_, X - X_rest), Eigen::VectorXd(), grad); // grad wrt. gbases
			return AdjointTools::map_node_to_primitive_order(state_, grad);
		});
	}

	bool AMIPSForm::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		Eigen::VectorXd X = get_updated_mesh_nodes(x1);
		Eigen::MatrixXd V1 = utils::unflatten(X, state_.mesh->dimension());
		bool flipped = utils::is_flipped(V1, F);

		if (flipped)
			adjoint_logger().trace("[{}] Step flips elements.", name());

		return !flipped;
	}
} // namespace polyfem::solver
