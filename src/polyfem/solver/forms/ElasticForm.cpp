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
#include <limits>
#include <vector>

using namespace polyfem::assembler;
using namespace polyfem::utils;
using namespace polyfem::quadrature;

namespace polyfem::solver
{
	namespace
	{
		// Conservative modes use corner rules so the energy samples J at the
		// corners. A negative weight at any point turns the log barrier there
		// into a well, and the discrete energy becomes unbounded below as J
		// vanishes at that point: the triangle order-6 rule (P4) has them at
		// the corners, the tet order-4 rule (P3) at the centroid. Use the
		// lowest order at or above the requested one whose rule has every
		// corner and only positive weights. A family of corner rules with
		// positive weights at every order would avoid the larger rules.
		Quadrature corner_quadrature(const int dim, const int order)
		{
			const auto is_usable = [dim](const Quadrature &quad) {
				if (quad.weights.size() == 0 || quad.weights.minCoeff() <= 0)
					return false;
				int n_corners = 0;
				for (Eigen::Index i = 0; i < quad.points.rows(); ++i)
				{
					const auto p = quad.points.row(i);
					if ((p.array() == 0 || p.array() == 1).all() && p.sum() <= 1)
						++n_corners;
				}
				return n_corners == dim + 1;
			};

			constexpr int max_bump = 3;
			Quadrature quad;
			for (int o = order; o <= order + max_bump; ++o)
			{
				if (dim == 2)
					TriQuadrature(true).get_quadrature(o, quad);
				else
					TetQuadrature(true).get_quadrature(o, quad);
				if (is_usable(quad))
					return quad;
			}
			log_and_throw_error("No corner quadrature with positive weights of order {} to {} in {}D", order, order + max_bump, dim);
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

			Quadrature tmp = corner_quadrature(dim, order);
			Quadrature quad;
			tmp.points.conservativeResize(tmp.points.rows(), dim + 1);
			tmp.points.col(dim) = 1. - tmp.points.leftCols(dim).rowwise().sum().array();

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

		if (is_conservative(check_inversion_))
		{
			Eigen::VectorXd x0;
			x0.setZero(n_bases_ * (is_volume_ ? 3 : 2));
			// Call is_valid() directly rather than going through one of the
			// step hooks: which of those actually certifies depends on the
			// mode, but the rest quadrature setup below assumes a valid rest
			// state either way.
			rest_nodes_ = utils::rest_nodes(is_volume_ ? 3 : 2, bases_, geom_bases_);
			// The energy evaluates det J from the same control points the
			// checks see; compute them once rather than on every assembly.
			ass_vals_cache_.set_control_points(utils::element_control_points(is_volume_ ? 3 : 2, bases_, geom_bases_));
			const auto [isvalid, id, tree] = is_valid(
				is_volume_ ? 3 : 2, bases_, geom_bases_, x0, jacobian_threshold_, conservative_max_iter_, &rest_nodes_);
			if (!isvalid)
				log_and_throw_error("Initial state has inverted elements (element {})!", id);

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

			refines_quadrature_ = basis_order > 1 || gbasis_order > 1;

			// Replace the corner quadrature set by LagrangeBasis for every element.
			// Some default corner rules have negative weights (see
			// corner_quadrature); override them here so the base quadrature is
			// also safe before any adaptive refinement occurs.
			const int dim = is_volume_ ? 3 : 2;
			if (refines_quadrature_)
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
		// Report what the conservative check cost over this solve. The call
		// counts matter as much as the totals: the static check runs once per
		// line-search trial point, the continuous one once per line search.
		if (static_check_calls_ > 0)
		{
			logger().debug(
				"Static Jacobian check: {} calls, {} rejections, {:.3f}s total ({:.3f} ms/call)",
				static_check_calls_, static_check_rejections_, static_check_time_,
				1e3 * static_check_time_ / static_check_calls_);
		}
		if (continuous_check_calls_ > 0)
		{
			logger().debug(
				"Continuous Jacobian check: {} calls, {:.3f}s total ({:.3f} ms/call)",
				continuous_check_calls_, continuous_check_time_,
				1e3 * continuous_check_time_ / continuous_check_calls_);
		}
		static_check_time_ = continuous_check_time_ = 0;
		static_check_calls_ = continuous_check_calls_ = static_check_rejections_ = 0;

		for (auto &t : quadrature_hierarchy_)
			t = Tree();
		pending_refinement_.reset();
	}

	double ElasticForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// Only the continuous check caps the step: it is the one that certifies
		// the whole segment. The static check instead rejects individual
		// candidates from is_step_valid(), so it leaves the step alone here.
		if (check_inversion_ != ElementInversionCheck::ConservativeContinuous)
			return 1.;

		const int dim = is_volume_ ? 3 : 2;

		// Largest gap allowed between the certified step and the first
		// inversion, as a fraction of the step. It only bounds the gap from
		// above: when the inversion starts away from a subdivision vertex the
		// check cannot stop until a sample shows it, by which point the gap is
		// already far smaller. Keep it small enough that the margin applied
		// below, rather than this slack, sets how far a capped step stays
		// from the inversion.
		constexpr double precision = 1e-2;

		double step, invalidStep;
		int invalidID;

		Tree subdivision_tree;
		{
			double transient_check_time = 0;
			{
				POLYFEM_SCOPED_TIMER("Transient Jacobian Check", transient_check_time);
				std::tie(step, invalidID, invalidStep, subdivision_tree) = max_time_step(dim, bases_, geom_bases_, x0, x1, precision, jacobian_threshold_, conservative_max_iter_, &rest_nodes_);
			}

			continuous_check_time_ += transient_check_time;
			++continuous_check_calls_;

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
		else if (invalidID >= 0)
		{
			// Any capped step: the path inverts somewhere the quadrature does
			// not sample, and left that way the next direction heads back to
			// it. Store as a candidate only; committed in post_step() if/when
			// this step is accepted (see pending_refinement_ in the header).
			pending_refinement_ = std::make_pair(invalidID, std::move(subdivision_tree));
		}
		else
		{
			pending_refinement_.reset();
		}

		// Stop short of the certified step. It can lie arbitrarily close to the
		// first inversion, and a step taken all the way to it can leave the
		// Jacobian somewhere within rounding of zero. No later check can then
		// certify any step from there, and the solve stalls. The margin is still
		// inside the certified range, which the widening in max_time_step()
		// covers at any correctly rounded point.
		constexpr double margin = 0.9;
		return step < 1 ? margin * step : step;
	}

	bool ElasticForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// NOTE: this hook is named for contact, but ElasticForm uses it to mean
		// "no element is inverted". polysolve's line search never calls it;
		// ALSolver does, on whole configurations, a handful of times per solve.
		// The line search is covered by max_step_size() (continuous) and
		// is_step_acceptable() (static).
		if (!is_conservative(check_inversion_))
			return true;

		const auto [isvalid, id, tree] = is_valid(is_volume_ ? 3 : 2, bases_, geom_bases_, x1, jacobian_threshold_, conservative_max_iter_, &rest_nodes_);
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

	bool ElasticForm::is_step_acceptable(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// The static check certifies one configuration, the same question the
		// quadrature sampling answers, but over the whole element. It runs only
		// on steps that already passed the line search's descent criteria:
		// running it on every trial point costs far more, and vetoing a trial
		// the energy cannot tell apart from a valid one leaves the line search
		// nothing to back away from.
		if (check_inversion_ != ElementInversionCheck::ConservativeStatic)
			return true;

		double check_time = 0;
		bool isvalid = false;
		int id = -1;
		Tree witness;
		{
			POLYFEM_SCOPED_TIMER("Static Jacobian Check", check_time);
			std::tie(isvalid, id, witness) = is_valid(
				is_volume_ ? 3 : 2, bases_, geom_bases_, x1, jacobian_threshold_, conservative_max_iter_, &rest_nodes_);
		}
		static_check_time_ += check_time;
		++static_check_calls_;

		if (isvalid)
			return true;

		++static_check_rejections_;
		logger().debug("Static Jacobian check rejected step at element {} (witness depth {})", id, witness.depth());

		// Refine towards the violation, so that energy evaluated at this
		// configuration samples it and is no longer finite. A rejection ends
		// the line search, and the next direction is computed on the refined
		// quadrature, so a later trial that reaches this configuration is
		// turned down by the line search's ordinary finiteness checks. Merge
		// the whole witness path rather than the usual two levels: the
		// violation is certified at a vertex of the deepest witness simplex,
		// and that vertex is only a quadrature point (corner quadrature) once
		// that level of refinement exists.
		if (id >= 0)
			commit_refinement(id, witness, std::numeric_limits<int>::max());

		return false;
	}

	void ElasticForm::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		if (!pending_refinement_.has_value())
			return;

		auto [id, subdivision_tree] = *pending_refinement_;
		commit_refinement(id, subdivision_tree);

		pending_refinement_.reset();
	}

	void ElasticForm::commit_refinement(const int id, utils::Tree &subdivision_tree, const int max_depth) const
	{
		if (!refines_quadrature_)
			return;

		const int dim = is_volume_ ? 3 : 2;

		// Merge into quadrature_hierarchy_ and rebuild the live quadrature for
		// this element, if the tree is actually refined.
		if (quadrature_hierarchy_[id].merge(subdivision_tree, max_depth))
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
