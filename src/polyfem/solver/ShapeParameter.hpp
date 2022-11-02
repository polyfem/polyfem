#pragma once

#include "Parameter.hpp"
#include <igl/slim.h>

#include <ipc/collisions/collision_constraint.hpp>
#include <ipc/broad_phase/broad_phase.hpp>

namespace polyfem
{
	// only works for linear geometry
	struct boundary_smoothing
	{
		boundary_smoothing() = default;

		void build_laplacian(const int n_verts, const int dim_, const Eigen::MatrixXi &boundary_edges, const std::vector<int> &boundary_nodes_, const std::set<int> &fixed_nodes_)
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
			std::vector<Eigen::Triplet<bool>> T_adj;
			for (int e = 0; e < boundary_edges.rows(); e++)
			{
				T_adj.emplace_back(boundary_edges(e, 0), boundary_edges(e, 1), true);
				T_adj.emplace_back(boundary_edges(e, 1), boundary_edges(e, 0), true);
			}
			adj.setFromTriplets(T_adj.begin(), T_adj.end());

			// compute graph degrees for vertices, only boundary
			std::vector<int> degrees(n_verts, 0);
			for (int k = 0; k < adj.outerSize(); ++k)
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, k); it; ++it)
					degrees[k]++;

			std::vector<Eigen::Triplet<double>> T_L;
			for (int k = 0; k < adj.outerSize(); ++k)
			{
				if (degrees[k] == 0 || fixed_nodes_mask[k])
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

		void smoothing_grad(const Eigen::MatrixXd &V, Eigen::VectorXd &grad)
		{
			grad.setZero(V.rows() * dim);
			// add smoothing energy grad to one form
			Eigen::MatrixXd smoothing_grad = 2 * (L.transpose() * (L * V));

			for (int b : boundary_nodes)
				grad.block(b * dim, 0, dim, 1) = smoothing_grad.block(b, 0, 1, dim).transpose();
		}

		double smoothing_energy(const Eigen::MatrixXd &V)
		{
			Eigen::MatrixXd laplacian = L * V;
			return laplacian.squaredNorm();
		}

		double weighted_smoothing_energy(const Eigen::MatrixXd &V)
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

		void weighted_smoothing_grad(const Eigen::MatrixXd &V, Eigen::VectorXd &grad)
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

	class ShapeParameter : public Parameter
	{
	public:
		ShapeParameter(std::vector<std::shared_ptr<State>> states_ptr);

		void update() override
		{
		}

		void map(const Eigen::MatrixXd &x, Eigen::MatrixXd &q) override
		{
		}

		void smoothing(const Eigen::VectorXd &x, Eigen::VectorXd &new_x) override;
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		bool is_intersection_free(const Eigen::VectorXd &x) override;
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		void line_search_end(bool failed) override;
		void post_step(const int iter_num, const Eigen::VectorXd &x0) override;

		bool pre_solve(const Eigen::VectorXd &newX) override;
		void post_solve(const Eigen::VectorXd &newX) override;

		bool remesh(Eigen::VectorXd &x) override;
		void build_fixed_nodes();
		void build_tied_nodes();

		std::function<void(const Eigen::VectorXd &x, const Eigen::MatrixXd &position, Eigen::MatrixXd &V)> x_to_param;
		std::function<void(Eigen::VectorXd &x, const Eigen::MatrixXd &V)> param_to_x;
		std::function<void(Eigen::VectorXd &grad_x, const Eigen::VectorXd &grad_v)> dparam_to_dx;

		std::map<int, std::vector<int>> optimization_boundary_to_node;

		const json &get_shape_params() { return shape_params; }

	private:
		std::vector<std::shared_ptr<State>> states_ptr_;

		Eigen::MatrixXd V_rest;
		Eigen::MatrixXi elements;

		std::set<int> fixed_nodes;
		std::vector<bool> free_dimension;

		std::vector<bool> tied_nodes_mask;
		std::vector<std::array<int, 2>> tied_nodes;

		bool mesh_flipped = false;

		json shape_params, slim_params;

		double target_weight = 1;

		// volume constraints
		bool has_volume_constraint;
		json volume_params;
		std::shared_ptr<CompositeFunctional> j_volume;

		// boundary smoothing
		bool has_boundary_smoothing;
		json boundary_smoothing_params;
		boundary_smoothing boundary_smoother;

		// below only used for problems with contact

		bool has_collision;

		double _dhat;
		double _prev_distance;
		double _barrier_stiffness;

		ipc::BroadPhaseMethod _broad_phase_method;
		double _ccd_tolerance;
		int _ccd_max_iterations;

		ipc::Constraints _constraint_set;
		ipc::CollisionMesh collision_mesh;
		ipc::FrictionConstraints _friction_constraint_set;
		ipc::Candidates _candidates;
		bool _use_cached_candidates = false;

		void update_constraint_set(const Eigen::MatrixXd &displaced_surface);
	};
} // namespace polyfem