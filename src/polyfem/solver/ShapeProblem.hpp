#pragma once

#include <polyfem/OptimizationProblem.hpp>
#include <igl/slim.h>

#include <ipc/collisions/collision_constraint.hpp>
#include <ipc/broad_phase/broad_phase.hpp>

namespace polyfem
{
	// only works for linear geometry
	struct boundary_smoothing
	{
		boundary_smoothing() = default;

		void build_laplacian(const int n_verts, const int dim_, const Eigen::MatrixXi &boundary_edges, const std::vector<int>& boundary_nodes_, const std::set<int> &fixed_nodes_)
		{
			dim = dim_;
			boundary_nodes = boundary_nodes_; 

			fixed_nodes_mask.assign(n_verts, false);
			for (int fn : fixed_nodes_)
				fixed_nodes_mask[fn] = true;
			
			L.setZero();
			L.resize(n_verts, n_verts);
			// construct adjacency matrix, only boundary
			adj.setZero();
			adj.resize(n_verts, n_verts); 
			std::vector<Eigen::Triplet<bool> > T_adj;
			for (int e = 0; e < boundary_edges.rows(); e++) {
				T_adj.emplace_back(boundary_edges(e, 0), boundary_edges(e, 1), true);
				T_adj.emplace_back(boundary_edges(e, 1), boundary_edges(e, 0), true);
			}
			adj.setFromTriplets(T_adj.begin(), T_adj.end());

			// compute graph degrees for vertices, only boundary
			std::vector<int> degrees(n_verts, 0);
			for (int k = 0; k < adj.outerSize(); ++k)
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, k); it; ++it)
					degrees[k]++;

			std::vector<Eigen::Triplet<double> > T_L;
			for (int k=0; k < adj.outerSize(); ++k) {
				if (degrees[k] == 0 || fixed_nodes_mask[k])
					continue;
				T_L.emplace_back(k, k, degrees[k]);
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj,k); it; ++it) {
					assert(it.row() == k);
					T_L.emplace_back(it.row(), it.col(), -1);
				}
			}
			L.setFromTriplets(T_L.begin(), T_L.end());

			L.prune([](int i, int j, double val) { return abs(val) > 1e-12; });
		}

		void smoothing_grad(const Eigen::MatrixXd& V, Eigen::VectorXd& grad)
		{
			grad.setZero(V.rows() * dim);
			// add smoothing energy grad to one form
			Eigen::MatrixXd smoothing_grad = 2 * (L.transpose() * (L * V));
			
			for (int b : boundary_nodes)
				grad.block(b * dim, 0, dim, 1) = smoothing_grad.block(b, 0, 1, dim).transpose();
		}

		double smoothing_energy(const Eigen::MatrixXd& V)
		{
			Eigen::MatrixXd laplacian = L * V;
			return laplacian.squaredNorm();
		}

		double weighted_smoothing_energy(const Eigen::MatrixXd& V)
		{
			double energy = 0;
			for (int b : boundary_nodes)
			{
				if (fixed_nodes_mask[b])
					continue;
				polyfem::RowVectorNd s;
				s.setZero(dim);
				double sum_norm = 0;
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					assert(it.col() != b);
					s += V.row(b) - V.row(it.col());
					sum_norm += (V.row(b) - V.row(it.col())).norm();
				}
				s = s / sum_norm;
				energy += pow(s.norm(), p);
			}
			return energy;
		}

		void weighted_smoothing_grad(const Eigen::MatrixXd& V, Eigen::VectorXd& grad)
		{
			grad.setZero(V.rows() * dim);
			for (int b : boundary_nodes)
			{
				if (fixed_nodes_mask[b])
					continue;
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
				s = s / sum_norm;

				for (int d = 0; d < dim; d++) 
				{
					grad(b * dim + d) += (s(d) * valence - s.squaredNorm() * sum_normalized(d)) * p * pow(s.norm(), p - 2.) / sum_norm;
				}

				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					for (int d = 0; d < dim; d++)
					{
						grad(it.col() * dim + d) -= (s(d) + s.squaredNorm() * (V(it.col(), d) - V(b, d)) / (V.row(b) - V.row(it.col())).norm()) * p * pow(s.norm(), p - 2.) / sum_norm;
					}
				}
			}
		}

		int dim;
		int p = 2;
		std::vector<int> boundary_nodes;
		std::vector<int> fixed_nodes_mask;
		Eigen::SparseMatrix<bool, Eigen::RowMajor> adj;
		Eigen::SparseMatrix<double, Eigen::RowMajor> L;
	};

	class ShapeProblem : public OptimizationProblem
	{
	public:
		ShapeProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_, const json &args);

		void initialize_mesh();

		double target_value(const TVector &x);
		double volume_value(const TVector &x);
		double smooth_value(const TVector &x);
		double barrier_energy(const TVector &x);

		void target_gradient(const TVector &x, TVector &gradv);
		void volume_gradient(const TVector &x, TVector &gradv);
		void smooth_gradient(const TVector &x, TVector &gradv);
		void barrier_gradient(const TVector &x, TVector &gradv);

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;

		double value(const TVector &x, const bool only_elastic) { return value(x); };
		void gradient(const TVector &x, TVector &gradv, const bool only_elastic) { gradient(x, gradv); };

		void smoothing(const TVector &x, TVector &new_x) override;
		bool is_step_valid(const TVector &x0, const TVector &x1);
		bool is_intersection_free(const TVector &x);
		bool is_step_collision_free(const TVector &x0, const TVector &x1);
		double max_step_size(const TVector &x0, const TVector &x1);
		double heuristic_max_step(const TVector &dx);

		void line_search_begin(const TVector &x0, const TVector &x1);
		void line_search_end(bool failed);
		void post_step(const int iter_num, const TVector &x0) override;

		void solution_changed(const TVector &newX) override;

		void save_to_file(const TVector &x0) override;

		bool remesh(TVector &x);
		void build_fixed_nodes();

		std::function<void(const TVector& x, Eigen::MatrixXd& V)> x_to_param;
		std::function<void(TVector& x, const Eigen::MatrixXd& V)> param_to_x;
		std::function<void(TVector& grad_x, const TVector& grad_v)> dparam_to_dx;

	private:
		double smoothing_weight = 0.;
		double target_weight = 1.;

		double target_volume = 0.;
		double volume_weight = 0.;
		bool penalize_large_volume = true;

		int adjust_smooth_period = 1e9;
		double smooth_ratio = 100;

		Eigen::MatrixXi elements;

		std::set<int> fixed_nodes;

		bool mesh_flipped = false;

		std::shared_ptr<CompositeFunctional> j_volume;
		boundary_smoothing boundary_smoother;

		// only used for problems with contact

		bool has_collision;
		
		double _dhat;
		double _prev_distance;
		double _barrier_stiffness;

		ipc::BroadPhaseMethod _broad_phase_method;
		double _ccd_tolerance;
		int _ccd_max_iterations;

		ipc::Constraints _constraint_set;
		ipc::FrictionConstraints _friction_constraint_set;
		ipc::Candidates _candidates;
		bool _use_cached_candidates = false;

		void update_constraint_set(const Eigen::MatrixXd &displaced_surface);
	};
} // namespace polyfem
