#include "SmoothingForms.hpp"
#include <polyfem/State.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::solver
{
	Eigen::MatrixXd BoundarySmoothingForm::compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const
	{
		return Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size());
	}

	void BoundarySmoothingForm::init_form()
	{
		const auto &mesh = *(state_.mesh);
		const int dim = mesh.dimension();
		const int n_verts = mesh.n_vertices();
		assert(mesh.is_simplicial());

		std::vector<int> tmp = {}; // args_["surface_selection"];
		std::set<int> surface_ids = std::set(tmp.begin(), tmp.end());

		// collect active nodes
		std::vector<bool> active_mask;
		active_mask.assign(n_verts, false);
		std::vector<Eigen::Triplet<bool>> T_adj;

		for (int b = 0; b < mesh.n_boundary_elements(); b++)
		{
			const int boundary_id = mesh.get_boundary_id(b);
			if (!surface_ids.empty() && surface_ids.find(boundary_id) == surface_ids.end())
				continue;
			
			for (int lv = 0; lv < dim; lv++)
			{
				active_mask[mesh.boundary_element_vertex(b, lv)] = true;
			}

			for (int lv1 = 0; lv1 < dim; lv1++)
				for (int lv2 = 0; lv2 < lv1; lv2++)
				{
					T_adj.emplace_back(mesh.boundary_element_vertex(b, lv2), mesh.boundary_element_vertex(b, lv1), true);
					T_adj.emplace_back(mesh.boundary_element_vertex(b, lv1), mesh.boundary_element_vertex(b, lv2), true);
				}
		}

		adj.setZero();
		adj.resize(n_verts, n_verts);
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
				if (!active_mask[k])
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
		const auto &mesh = *(state_.mesh);
		const int dim = mesh.dimension();
		const int n_verts = mesh.n_vertices();

		double val = 0;
		if (scale_invariant_)
		{
			for (int b = 0; b < adj.rows(); b++)
			{
				polyfem::RowVectorNd s;
				s.setZero(dim);
				double sum_norm = 0;
				int valence = 0;
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					assert(it.col() != b);
					auto x = mesh.point(b) - mesh.point(it.col());
					s += x;
					sum_norm += x.norm();
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
		{
			Eigen::MatrixXd V;
			state_.get_vertices(V);

			val = (L * V).eval().squaredNorm();
		}

		return val;
	}

	void BoundarySmoothingForm::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const auto &mesh = *(state_.mesh);
		const int dim = mesh.dimension();
		const int n_verts = mesh.n_vertices();

		Eigen::VectorXd grad;
		if (scale_invariant_)
		{
			grad.setZero(n_verts * dim);
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
					auto x = mesh.point(b) - mesh.point(it.col());
					s += x;
					sum_norm += x.norm();
					sum_normalized += x.normalized();
					valence += 1;
				}
				if (valence)
				{
					s = s / sum_norm;
					const double coeff = power_ * pow(s.norm(), power_ - 2.) / sum_norm;

					grad.segment(b * dim, dim) += (s * valence - s.squaredNorm() * sum_normalized) * coeff;
					for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
						grad.segment(it.col() * dim, dim) -= (s + s.squaredNorm() * (mesh.point(it.col()) - mesh.point(b)).normalized()) * coeff;
				}
			}
		}
		else
		{
			Eigen::MatrixXd V;
			state_.get_vertices(V);
			
			grad = utils::flatten(2 * (L.transpose() * (L * V)));
		}

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
} // namespace polyfem::solver