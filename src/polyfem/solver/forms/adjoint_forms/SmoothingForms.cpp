#include "SmoothingForms.hpp"

namespace polyfem::solver
{
	void BoundarySmoothingForm::init_form()
	{
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state_.get_vf(V, F);

		const int dim = V.cols();
		const int n_verts = V.rows();

		// collect active nodes
		std::vector<bool> active_mask;
		active_mask.assign(n_verts, false);
		std::vector<int> tmp = {}; // args_["surface_selection"];
		std::set<int> surface_ids = std::set(tmp.begin(), tmp.end());

		const auto &gbases = state_.geom_bases();
		for (const auto &lb : state_.total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); i++)
			{
				const int global_primitive_id = lb.global_primitive_id(i);
				const int boundary_id = state_.mesh->get_boundary_id(global_primitive_id);
				if (!surface_ids.empty() && surface_ids.find(boundary_id) == surface_ids.end())
					continue;

				const auto nodes = gbases[e].local_nodes_for_primitive(lb.global_primitive_id(i), *state_.mesh);

				for (int n = 0; n < nodes.size(); n++)
				{
					const auto &global = gbases[e].bases[nodes(n)].global();
					for (int g = 0; g < global.size(); g++)
						active_mask[global[g].index] = true;
				}
			}
		}

		adj.setZero();
		adj.resize(n_verts, n_verts);
		std::vector<Eigen::Triplet<bool>> T_adj;

		ipc::CollisionMesh collision_mesh;
		state_.build_collision_mesh(collision_mesh, state_.n_geom_bases, state_.geom_bases());
		for (int e = 0; e < collision_mesh.num_edges(); e++)
		{
			int v1 = collision_mesh.to_full_vertex_id(collision_mesh.edges()(e, 0));
			int v2 = collision_mesh.to_full_vertex_id(collision_mesh.edges()(e, 1));
			if (active_mask[v1] && active_mask[v2])
			{
				T_adj.emplace_back(v1, v2, true);
				T_adj.emplace_back(v2, v1, true);
			}
		}
		adj.setFromTriplets(T_adj.begin(), T_adj.end());

		std::vector<int> degrees(n_verts, 0);
		for (int k = 0; k < adj.outerSize(); ++k)
			for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, k); it; ++it)
				degrees[k]++;

		L.setZero();
		L.resize(n_verts, n_verts);
		if (!scale_invariant_)
		{
			std::vector<Eigen::Triplet<double>> T_L;
			for (int k = 0; k < adj.outerSize(); ++k)
			{
				if (degrees[k] == 0 || !active_mask[k])
					continue;
				T_L.emplace_back(k, k, degrees[k]);
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, k); it; ++it)
				{
					assert(it.row() == k);
					T_L.emplace_back(it.row(), it.col(), -1);
				}
			}
			L.setFromTriplets(T_L.begin(), T_L.end());
			L.prune([](int i, int j, double val) { return abs(val) > 1e-12; });
		}
	}

	double BoundarySmoothingForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state_.get_vf(V, F);
		const int dim = V.cols();
		const int n_verts = V.rows();

		double val = 0;
		if (scale_invariant_)
		{
			for (int b = 0; b < adj.rows(); b++)
			{
				polyfem::RowVectorNd s;
				s.setZero(V.cols());
				double sum_norm = 0;
				int valence = 0;
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					assert(it.col() != b);
					s += V.row(b) - V.row(it.col());
					sum_norm += (V.row(b) - V.row(it.col())).norm();
					valence += 1;
				}
				if (valence)
				{
					s = s / sum_norm;
					val += pow(s.norm(), power_);
				}
			}
		}
		else
			val = (L * V).eval().squaredNorm();

		return val;
	}

	void BoundarySmoothingForm::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state_.get_vf(V, F);
		const int dim = V.cols();
		const int n_verts = V.rows();

		Eigen::VectorXd grad;
		if (scale_invariant_)
		{
			grad.setZero(V.size());
			for (int b = 0; b < adj.rows(); b++)
			{
				polyfem::RowVectorNd s;
				s.setZero(dim);
				double sum_norm = 0;
				auto sum_normalized = s;
				int valence = 0;
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					assert(it.col() != b);
					auto x = V.row(b) - V.row(it.col());
					s += x;
					sum_norm += x.norm();
					sum_normalized += x.normalized();
					valence += 1;
				}
				if (valence)
				{
					s = s / sum_norm;

					for (int d = 0; d < dim; d++)
					{
						grad(b * dim + d) += (s(d) * valence - s.squaredNorm() * sum_normalized(d)) * power_ * pow(s.norm(), power_ - 2.) / sum_norm;
					}

					for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
					{
						for (int d = 0; d < dim; d++)
						{
							grad(it.col() * dim + d) -= (s(d) + s.squaredNorm() * (V(it.col(), d) - V(b, d)) / (V.row(b) - V.row(it.col())).norm()) * power_ * pow(s.norm(), power_ - 2.) / sum_norm;
						}
					}
				}
			}
		}
		else
			grad = utils::flatten(2 * (L.transpose() * (L * V)));

		gradv.setZero(x.size());
		for (auto &p : variable_to_simulations_)
		{
			if (&p->get_state() != &state_)
				continue;
			if (p->get_parameter_type() != ParameterType::Shape)
				continue;
			gradv += p->get_parametrization().apply_jacobian(grad, x);
		}
	}
} // namespace polyfem::solver