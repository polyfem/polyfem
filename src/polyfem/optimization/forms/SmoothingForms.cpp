#include <polyfem/optimization/forms/SmoothingForms.hpp>

#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>

#include <cassert>
#include <memory>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

namespace polyfem::solver
{
	BoundarySmoothingForm::BoundarySmoothingForm(
		const VariableToSimulationGroup &variable_to_simulations,
		std::shared_ptr<const varform::DifferentiableVarForm> varform,
		const bool scale_invariant,
		const int power,
		const std::vector<int> &surface_selections,
		const std::vector<int> &active_dims) : AdjointForm(variable_to_simulations), varform_(std::move(varform)), scale_invariant_(scale_invariant), power_(power), active_dims_(active_dims)
	{
		const auto &mesh = varform_->get_mesh();
		const int dim = mesh.dimension();
		const int n_verts = mesh.n_vertices();
		assert(mesh.is_simplicial());
		// empty implies all active.
		if (active_dims_.empty())
		{
			active_dims_.resize(dim);
			std::iota(active_dims_.begin(), active_dims_.end(), 0);
		}

		surface_ids_ = std::set(surface_selections.begin(), surface_selections.end());

		// collect active nodes
		std::vector<bool> active_mask;
		active_mask.assign(n_verts, false);
		std::vector<Eigen::Triplet<bool>> T_adj;

		for (int b = 0; b < mesh.n_boundary_elements(); b++)
		{
			const int boundary_id = mesh.get_boundary_id(b);
			if (!surface_ids_.empty() && surface_ids_.find(boundary_id) == surface_ids_.end())
				continue;

			for (int lv = 0; lv < dim; lv++)
			{
				active_mask[mesh.boundary_element_vertex(b, lv)] = true;
			}

			for (int lv1 = 0; lv1 < dim; lv1++)
				for (int lv2 = 0; lv2 < lv1; lv2++)
				{
					const int v1 = mesh.boundary_element_vertex(b, lv1);
					const int v2 = mesh.boundary_element_vertex(b, lv2);
					T_adj.emplace_back(v2, v1, true);
					T_adj.emplace_back(v1, v2, true);
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
				T_L.emplace_back(k, k, 1);
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, k); it; ++it)
				{
					assert(it.row() == k);
					T_L.emplace_back(it.row(), it.col(), -1. / degrees[k]);
				}
			}
			L.setFromTriplets(T_L.begin(), T_L.end());
			L.prune([](int i, int j, double val) { return abs(val) > 1e-12; });
		}
	}

	double BoundarySmoothingForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const auto &mesh = varform_->get_mesh();
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
					polyfem::RowVectorNd x = mesh.point(b) - mesh.point(it.col());
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
			varform_->get_vertices(V);

			val = (L * V(Eigen::all, active_dims_)).squaredNorm();
		}

		return val;
	}

	void BoundarySmoothingForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const auto &mesh = varform_->get_mesh();
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
				polyfem::RowVectorNd sum_normalized = s;
				int valence = 0;
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					assert(it.col() != b);
					polyfem::RowVectorNd x = mesh.point(b) - mesh.point(it.col());
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
			varform_->get_vertices(V);

			Eigen::MatrixXd grad_mat = 2 * (L.transpose() * (L * V));
			for (int d = 0; d < dim; d++)
				if (std::find(active_dims_.begin(), active_dims_.end(), d) == active_dims_.end())
					grad_mat.col(d).setZero();
			grad = utils::flatten(grad_mat);
		}

		gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::Shape, *varform_, x, [&grad]() {
			return grad;
		});
	}

	ElasticMaterialSmoothingForm::ElasticMaterialSmoothingForm(
		const VariableToSimulationGroup &variable_to_simulations,
		std::shared_ptr<const varform::DifferentiableVarForm> varform,
		const std::vector<int> &volume_selections)
		: AdjointForm(variable_to_simulations), varform_(std::move(varform))
	{
		auto &mesh = varform_->get_mesh();
		if (!mesh.is_conforming())
			log_and_throw_adjoint_error("Elastic material smoothing form does not support non-conforming meshes!");

		int elastic_mappings = 0;
		for (auto &v2s : variable_to_simulations_.data)
		{
			if (v2s->parameter_type() == ParameterType::LameParameter
				&& v2s->affects_varform(*varform_))
			{
				++elastic_mappings;
			}
		}
		if (elastic_mappings != 1)
		{
			// Unclear how to define neighboring elements when multiple varforms (potentially multiple meshes) exists.
			log_and_throw_adjoint_error("Elastic material smoothing form does not support more than one affects varform!");
		}

		std::set<int> volume_ids(volume_selections.begin(), volume_selections.end());
		auto is_selected = [&mesh, &volume_ids](const int element) {
			// empty volume selection implies all active.
			return volume_ids.empty() || volume_ids.count(mesh.get_body_id(element)) > 0;
		};

		// Build adjacency information.
		if (mesh.is_volume())
		{
			const auto *mesh3d = dynamic_cast<const mesh::Mesh3D *>(&mesh);
			assert(mesh3d != nullptr);

			for (int element = 0; element < mesh3d->n_cells(); ++element)
			{
				if (!is_selected(element))
					continue;

				// Iterate through all faces of an element then query the interfacing neighbor.
				for (int face = 0; face < mesh3d->n_cell_faces(element); ++face)
				{
					// get_index_from_element takes global element id, local face id, local vertex id then
					// return a navigation index struct. You can consider navigation index as a descriptor
					// that maps between vertex <-> edge <-> face <-> element. With this info, we then call
					// switch_element to query the neighboring element. Since the purpose is to find neighbors,
					// vertex info is redundant hence the dummy 0.
					auto index = mesh3d->get_index_from_element(element, face, 0);
					int neighbor = mesh3d->switch_element(index).element;
					// neighbor == -1 indicates boundary face.
					if (neighbor >= 0 && is_selected(neighbor))
						adjacent_elements_.emplace_back(element, neighbor);
				}
			}
		}
		else
		{
			const auto *mesh2d = dynamic_cast<const mesh::Mesh2D *>(&mesh);
			assert(mesh2d != nullptr);

			for (int element = 0; element < mesh2d->n_faces(); ++element)
			{
				if (!is_selected(element))
					continue;

				for (int edge = 0; edge < mesh2d->n_face_vertices(element); ++edge)
				{
					auto index = mesh2d->get_index_from_face(element, edge);
					int neighbor = mesh2d->switch_face(index).face;
					// neighbor == -1 indicates boundary edge.
					if (neighbor >= 0 && is_selected(neighbor))
						adjacent_elements_.emplace_back(element, neighbor);
				}
			}
		}
	}

	Eigen::VectorXd ElasticMaterialSmoothingForm::lame_parameters(const Eigen::VectorXd &x) const
	{
		int n_elements = varform_->get_mesh().n_elements();
		Eigen::VectorXd parameters = Eigen::VectorXd::Zero(2 * n_elements);
		variable_to_simulations_.compute_state_variable(
			ParameterType::LameParameter, *varform_, x, parameters);
		return parameters;
	}

	double ElasticMaterialSmoothingForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		if (adjacent_elements_.empty())
			return 0;

		int n_elements = varform_->get_mesh().n_elements();
		Eigen::VectorXd parameters = lame_parameters(x);
		auto lmd = parameters.head(n_elements);
		auto mu = parameters.tail(n_elements);

		double value = 0;
		// See class level doc.
		for (auto &[a, b] : adjacent_elements_)
		{
			double lmd_penalty = 1 - lmd(a) / lmd(b);
			double mu_penalty = 1 - mu(a) / mu(b);
			value += lmd_penalty * lmd_penalty + mu_penalty * mu_penalty;
		}

		return value / adjacent_elements_.size();
	}

	void ElasticMaterialSmoothingForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		if (adjacent_elements_.empty())
		{
			gradv = Eigen::VectorXd::Zero(x.size());
			return;
		}

		int n_elements = varform_->get_mesh().n_elements();
		Eigen::VectorXd parameters = lame_parameters(x);
		auto lmd = parameters.head(n_elements);
		auto mu = parameters.tail(n_elements);
		Eigen::VectorXd grad = Eigen::VectorXd::Zero(parameters.size());

		// See class level doc.
		for (auto &[a, b] : adjacent_elements_)
		{
			double lmd_ratio = lmd(a) / lmd(b);
			grad(a) += 2 * (lmd_ratio - 1) / lmd(b);
			grad(b) += 2 * (1 - lmd_ratio) * lmd(a) / (lmd(b) * lmd(b));

			double mu_ratio = mu(a) / mu(b);
			grad(n_elements + a) += 2 * (mu_ratio - 1) / mu(b);
			grad(n_elements + b) += 2 * (1 - mu_ratio) * mu(a) / (mu(b) * mu(b));
		}

		grad /= adjacent_elements_.size();
		gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::LameParameter, *varform_, x, [&grad]() {
			return grad;
		});
	}
} // namespace polyfem::solver
