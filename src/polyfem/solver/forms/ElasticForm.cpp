#include "ElasticForm.hpp"

#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/quadrature/TetQuadrature.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/Jacobian.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace polyfem::assembler;
using namespace polyfem::utils;
using namespace polyfem::quadrature;

namespace polyfem::solver
{
	namespace
	{
		Eigen::MatrixXd refined_nodes(const int dim, const int i)
		{
			Eigen::MatrixXd A(dim + 1, dim);
			if (dim == 2)
			{
				A << 0., 0.,
					1., 0.,
					0., 1.;
				switch (i)
				{
				case 0:
					break;
				case 1:
					A.col(0).array() += 1;
					break;
				case 2:
					A.col(1).array() += 1;
					break;
				case 3:
					A.array() -= 1;
					A *= -1;
					break;
				default:
					throw std::runtime_error("Invalid node index");
				}
			}
			else
			{
				A << 0, 0, 0,
					1, 0, 0,
					0, 1, 0,
					0, 0, 1;
				switch (i)
				{
				case 0:
					break;
				case 1:
					A.col(0).array() += 1;
					break;
				case 2:
					A.col(1).array() += 1;
					break;
				case 3:
					A.col(2).array() += 1;
					break;
				case 4:
				{
					Eigen::VectorXd tmp = 1. - A.col(0).array() - A.col(1).array();
					A.col(2) += A.col(1);
					A.col(1) = A.col(0);
					A.col(0) = tmp;
					break;
				}
				case 5:
				{
					Eigen::VectorXd tmp = 1. - A.col(0).array();
					A.col(1) += A.col(0);
					A.col(0) = tmp;
					break;
				}
				case 6:
				{
					Eigen::VectorXd tmp0 = A.col(0);
					Eigen::VectorXd tmp1 = A.col(1);
					A.col(0) = tmp1;
					A.col(1) = 1. - tmp0.array() - tmp1.array();
					A.col(2) += tmp0 + tmp1;
					break;
				}
				case 7:
				{
					Eigen::VectorXd tmp = A.col(1);
					A.col(0) += tmp;
					A.col(1) = 1. - tmp.array();
					A.col(2) += tmp;
					break;
				}
				default:
					throw std::runtime_error("Invalid node index");
				}
			}
			return A / 2;
		}

		/// @brief given the position of the vertices of a triangle, extract the subtriangle vertices based on the tree
		/// @param pts vertices of the triangle
		/// @param tree refinement hiararchy
		/// @return vertices of the subtriangles, and the refinement level of each subtriangle
		std::tuple<Eigen::MatrixXd, std::vector<int>> extract_subelement(const Eigen::MatrixXd &pts, const Tree &tree)
		{
			if (!tree.has_children())
				return {pts, std::vector<int>{0}};

			const int dim = pts.cols();
			Eigen::MatrixXd out;
			std::vector<int> levels;
			for (int i = 0; i < tree.n_children(); i++)
			{
				Eigen::MatrixXd uv;
				uv.setZero(dim + 1, dim + 1);
				uv.rightCols(dim) = refined_nodes(dim, i);
				if (dim == 2)
					uv.col(0) = 1. - uv.col(2).array() - uv.col(1).array();
				else
					uv.col(0) = 1. - uv.col(3).array() - uv.col(1).array() - uv.col(2).array();

				Eigen::MatrixXd pts_ = uv * pts;

				auto [tmp, L] = extract_subelement(pts_, tree.child(i));
				if (out.size() == 0)
					out = tmp;
				else
				{
					out.conservativeResize(out.rows() + tmp.rows(), Eigen::NoChange);
					out.bottomRows(tmp.rows()) = tmp;
				}
				for (int &i : L)
					++i;
				levels.insert(levels.end(), L.begin(), L.end());
			}
			return {out, levels};
		}

		quadrature::Quadrature refine_quadrature(const Tree &tree, const int dim, const int order)
		{
			Eigen::MatrixXd pts(dim + 1, dim);
			if (dim == 2)
				pts << 0., 0.,
					1., 0.,
					0., 1.;
			else
				pts << 0, 0, 0,
					1, 0, 0,
					0, 1, 0,
					0, 0, 1;
			auto [quad_points, levels] = extract_subelement(pts, tree);

			Quadrature tmp, quad;
			if (dim == 2)
			{
				TriQuadrature tri_quadrature(true);
				tri_quadrature.get_quadrature(order, tmp);
				tmp.points.conservativeResize(tmp.points.rows(), dim + 1);
				tmp.points.col(dim) = 1. - tmp.points.col(0).array() - tmp.points.col(1).array();
			}
			else
			{
				TetQuadrature tet_quadrature(true);
				// The corner rule for order 4 (Liu-Vinokur 9) has a large negative
				// weight at the centroid, which corrupts non-polynomial integrands
				// like NeoHookean near inversion. Order 5 has all-positive weights.
				const int safe_order = (order == 4) ? 5 : order;
				tet_quadrature.get_quadrature(safe_order, tmp);
				tmp.points.conservativeResize(tmp.points.rows(), dim + 1);
				tmp.points.col(dim) = 1. - tmp.points.col(0).array() - tmp.points.col(1).array() - tmp.points.col(2).array();
			}

			quad.points.resize(tmp.size() * levels.size(), dim);
			quad.weights.resize(tmp.size() * levels.size());

			for (int i = 0; i < levels.size(); i++)
			{
				quad.points.middleRows(i * tmp.size(), tmp.size()) = tmp.points * quad_points.middleRows(i * (dim + 1), dim + 1);
				quad.weights.segment(i * tmp.size(), tmp.size()) = tmp.weights / pow(2, dim * levels[i]);
			}
			assert(fabs(quad.weights.sum() - tmp.weights.sum()) < 1e-8);

			return quad;
		}

		void update_quadrature(const int invalidID, const int dim, Tree &tree, const int quad_order, basis::ElementBases &bs, const basis::ElementBases &gbs, assembler::AssemblyValsCache &ass_vals_cache)
		{
			// update quadrature to capture the point with negative jacobian
			const Quadrature quad = refine_quadrature(tree, dim, quad_order);

			// capture the flipped point by refining the quadrature
			bs.set_quadrature([quad](Quadrature &quad_) {
				quad_ = quad;
			});
			logger().debug("New number of quadrature points: {}, level: {}", quad.size(), tree.depth());

			if (ass_vals_cache.is_initialized())
				ass_vals_cache.update(invalidID, dim == 3, bs, gbs);
		}
	} // namespace

	ElasticForm::ElasticForm(const int n_bases,
							 std::vector<basis::ElementBases> &bases,
							 const std::vector<basis::ElementBases> &geom_bases,
							 const assembler::Assembler &assembler,
							 assembler::AssemblyValsCache &ass_vals_cache,
							 const double t, const double dt,
							 const bool is_volume,
							 const double jacobian_threshold,
							 const ElementInversionCheck check_inversion,
							 const unsigned conservative_max_iter)
		: n_bases_(n_bases),
		  bases_(bases),
		  geom_bases_(geom_bases),
		  assembler_(assembler),
		  ass_vals_cache_(ass_vals_cache),
		  t_(t),
		  jacobian_threshold_(jacobian_threshold),
		  check_inversion_(check_inversion),
		  conservative_max_iter_(conservative_max_iter),
		  dt_(dt),
		  is_volume_(is_volume)
	{
		if (assembler_.is_linear())
			compute_cached_stiffness();
		// mat_cache_ = std::make_unique<utils::DenseMatrixCache>();
		mat_cache_ = std::make_unique<utils::SparseMatrixCache>();
		quadrature_hierarchy_.resize(bases_.size());

		quadrature_order_ = AssemblerUtils::quadrature_order(assembler_.name(), bases_[0].bases[0].order(), AssemblerUtils::BasisType::SIMPLEX_LAGRANGE, is_volume_ ? 3 : 2);

		if (check_inversion_ != ElementInversionCheck::Discrete)
		{
			Eigen::VectorXd x0;
			x0.setZero(n_bases_ * (is_volume_ ? 3 : 2));
			if (!is_step_collision_free(x0, x0))
				log_and_throw_error("Initial state has inverted elements!");

			int basis_order = 0;
			int gbasis_order = 0;
			for (int e = 0; e < bases_.size(); e++)
			{
				if (basis_order == 0)
					basis_order = bases_[e].bases.front().order();
				else if (basis_order != bases_[e].bases.front().order())
					log_and_throw_error("Non-uniform basis order not supported for conservative Jacobian check!!");
				if (gbasis_order == 0)
					gbasis_order = geom_bases_[e].bases.front().order();
				else if (gbasis_order != geom_bases_[e].bases.front().order())
					log_and_throw_error("Non-uniform gbasis order not supported for conservative Jacobian check!!");
			}

			// Replace the corner quadrature set by LagrangeBasis for every element.
			// The default corner rule for order 4 has a negative centroid weight
			// (see refine_quadrature); override it here so the base quadrature is
			// also safe before any adaptive refinement occurs.
			const int dim = is_volume_ ? 3 : 2;
			for (int e = 0; e < (int)bases_.size(); e++)
				update_quadrature(e, dim, quadrature_hierarchy_[e], quadrature_order_, bases_[e], geom_bases_[e], ass_vals_cache_);
		}
	}

	double ElasticForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return assembler_.assemble_energy(
			is_volume_,
			bases_, geom_bases_, ass_vals_cache_, t_, dt_, x, x_prev_);
	}

	Eigen::VectorXd ElasticForm::value_per_element_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd out = assembler_.assemble_energy_per_element(
			is_volume_, bases_, geom_bases_, ass_vals_cache_, t_, dt_, x, x_prev_);
		assert(abs(out.sum() - value_unweighted(x)) < std::max(1e-10 * out.sum(), 1e-10));
		return out;
	}

	void ElasticForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		Eigen::MatrixXd grad;
		assembler_.assemble_gradient(is_volume_, n_bases_, bases_, geom_bases_,
									 ass_vals_cache_, t_, dt_, x, x_prev_, grad);
		gradv = grad;
	}

	void ElasticForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("elastic hessian");

		hessian.resize(x.size(), x.size());

		if (assembler_.is_linear())
		{
			assert(cached_stiffness_.rows() == x.size() && cached_stiffness_.cols() == x.size());
			hessian = cached_stiffness_;
		}
		else
		{
			// NOTE: mat_cache_ is marked as mutable so we can modify it here
			assembler_.assemble_hessian(
				is_volume_, n_bases_, project_to_psd_, bases_,
				geom_bases_, ass_vals_cache_, t_, dt_, x, x_prev_, *mat_cache_, hessian);
		}
	}

	void ElasticForm::finish()
	{
		for (auto &t : quadrature_hierarchy_)
			t = Tree();
		pending_refinement_.reset();
	}

	double ElasticForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		if (check_inversion_ == ElementInversionCheck::Discrete)
			return 1.;

		const int dim = is_volume_ ? 3 : 2;

		double step, invalidStep;
		int invalidID;

		Tree subdivision_tree;
		{
			double transient_check_time = 0;
			{
				POLYFEM_SCOPED_TIMER("Transient Jacobian Check", transient_check_time);
				std::tie(step, invalidID, invalidStep, subdivision_tree) = max_time_step(dim, bases_, geom_bases_, x0, x1, .25, 0., conservative_max_iter_);
			}

			logger().log(step == 0 ? spdlog::level::err : (step == 1. ? spdlog::level::trace : spdlog::level::debug),
						 "Jacobian max step size: {} at element {}, invalid step size: {}, runtime {} sec, tree depth {}", step, invalidID, invalidStep, transient_check_time, subdivision_tree.depth());
		}

		if (invalidID >= 0 && step == 0)
		{
			// A step of 0 can never be accepted, so it will never reach
			// post_step(); commit the refinement candidate immediately so
			// the next iteration's energy/gradient/Hessian actually sees
			// this near-degenerate point instead of repeating the same
			// failure against stale quadrature.
			commit_refinement(invalidID, subdivision_tree);
			pending_refinement_.reset();
		}
		else if (invalidID >= 0 && step <= 0.5)
		{
			// Store as a candidate only; committed in post_step() if/when
			// this step is accepted (see pending_refinement_ in the header).
			pending_refinement_ = std::make_pair(invalidID, std::move(subdivision_tree));
		}
		else
		{
			pending_refinement_.reset();
		}

		return step;
	}

	bool ElasticForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		if (check_inversion_ == ElementInversionCheck::Discrete)
			return true;

		const auto [isvalid, id, tree] = is_valid(is_volume_ ? 3 : 2, bases_, geom_bases_, x1, 0., conservative_max_iter_);
		return isvalid;
	}

	bool ElasticForm::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// check inversion on quadrature points
		Eigen::VectorXd grad;
		first_derivative(x1, grad);
		if (grad.array().isNaN().any())
			return false;

		return true;

		// Check the scalar field in the output does not contain NANs.
		// WARNING: Does not work because the energy is not evaluated at the same quadrature points.
		//          This causes small step lengths in the LS.
		// TVector x1_full;
		// reduced_to_full(x1, x1_full);
		// return state_.check_scalar_value(x1_full, true, false);
		// return true;
	}

	void ElasticForm::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		if (!pending_refinement_.has_value())
			return;

		auto [id, subdivision_tree] = *pending_refinement_;
		commit_refinement(id, subdivision_tree);

		pending_refinement_.reset();
	}

	void ElasticForm::commit_refinement(const int id, utils::Tree &subdivision_tree) const
	{
		const int dim = is_volume_ ? 3 : 2;

		// Merge into quadrature_hierarchy_ and rebuild the live quadrature for
		// this element, if the tree is actually refined.
		if (quadrature_hierarchy_[id].merge(subdivision_tree))
		{
			auto &bs = bases_[id];
			auto &gbs = geom_bases_[id];
			update_quadrature(id, dim, quadrature_hierarchy_[id], quadrature_order_, bs, gbs, ass_vals_cache_);
		}
	}

	void ElasticForm::compute_cached_stiffness()
	{
		if (assembler_.is_linear() && cached_stiffness_.size() == 0)
		{
			assembler_.assemble(is_volume_, n_bases_, bases_, geom_bases_,
								ass_vals_cache_, t_, cached_stiffness_);
		}
	}
} // namespace polyfem::solver
