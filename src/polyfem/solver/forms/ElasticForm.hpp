#pragma once

#include "Form.hpp"

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>

#include <polyfem/utils/Jacobian.hpp>
#include <polyfem/utils/MatrixCache.hpp>
#include <polyfem/utils/Types.hpp>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace polyfem::solver
{
	/// @brief Strategy used to keep elements from inverting.
	enum class ElementInversionCheck
	{
		/// @brief No geometric check: inversion is only noticed indirectly,
		/// when the energy/gradient sampled at the quadrature points turns
		/// non-finite. Cheap, but blind to inversion between sample points.
		Discrete,
		/// @brief Certify that each step the line search accepts is
		/// inversion-free (utils::is_valid). Substitute for the quadrature
		/// sampling above: it answers the same question, but over the whole
		/// element, and refines the quadrature where it finds a violation.
		ConservativeStatic,
		/// @brief Certify the whole segment between two configurations and
		/// return the largest inversion-free step (utils::max_time_step).
		/// Caps the step before the line search instead of testing a point.
		ConservativeContinuous,
	};
	NLOHMANN_JSON_SERIALIZE_ENUM(
		polyfem::solver::ElementInversionCheck,
		{{ElementInversionCheck::Discrete, "Discrete"},
		 {ElementInversionCheck::ConservativeStatic, "ConservativeStatic"},
		 {ElementInversionCheck::ConservativeContinuous, "ConservativeContinuous"}})

	/// @brief Does this strategy certify validity, rather than relying on
	/// sampled quantities turning non-finite?
	inline bool is_conservative(const ElementInversionCheck check)
	{
		return check != ElementInversionCheck::Discrete;
	}

	/// @brief Form of the elasticity potential and forces
	class ElasticForm : public Form
	{
		friend class ElasticForceDerivative;

	public:
		/// @brief Construct a new Elastic Form object
		/// @param state Reference to the simulation state
		ElasticForm(const int n_bases,
					std::vector<basis::ElementBases> &bases,
					const std::vector<basis::ElementBases> &geom_bases,
					const assembler::Assembler &assembler,
					assembler::AssemblyValsCache &ass_vals_cache,
					const double t, const double dt,
					const bool is_volume,
					const double jacobian_threshold = 0.,
					const ElementInversionCheck check_inversion = ElementInversionCheck::Discrete,
					const unsigned conservative_max_iter = 100000);

		std::string name() const override { return "elastic"; }

	protected:
		/// @brief Compute the elastic potential value
		/// @param x Current solution
		/// @return Value of the elastic potential
		virtual double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the value of the form multiplied with the weigth
		/// @param x Current solution
		/// @return Computed value
		Eigen::VectorXd value_per_element_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	public:
		/// @brief Determine if a step from solution x0 to solution x1 is allowed
		/// @param x0 Current solution
		/// @param x1 Proposed next solution
		/// @return True if the step is allowed
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		/// @brief Checks if the step is inversion free
		/// @return True if the step is inversion free else false
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		/// @brief Static inversion check on a step the line search is about to
		/// accept (ConservativeStatic only). On rejection, refines the quadrature
		/// of the offending element towards the violation, so the energy there
		/// is no longer finite and the line search backs away from it.
		/// @return True if the step is inversion free
		bool is_step_acceptable(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		/// @brief Update time-dependent fields
		/// @param t Current time
		/// @param x Current solution at time t
		void update_quantities(const double t, const Eigen::VectorXd &x) override
		{
			t_ = t;
			x_prev_ = x;
		}

		/// @brief Determine the maximum step size allowable between the current and next solution
		/// @param x0 Current solution (step size = 0)
		/// @param x1 Next solution (step size = 1)
		/// @return Maximum allowable step size
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		/// @brief Reset adaptive quadrature refinement after each complete nonlinear solve.
		void finish() override;

		/// @brief Commit the quadrature refinement candidate found by the
		/// most recent max_step_size() call, once the step is accepted.
		void post_step(const polysolve::nonlinear::PostStepData &data) override;

	private:
		const int n_bases_;
		std::vector<basis::ElementBases> &bases_;
		const std::vector<basis::ElementBases> &geom_bases_;

		const assembler::Assembler &assembler_; ///< Reference to the assembler
		assembler::AssemblyValsCache &ass_vals_cache_;
		double t_;
		const double jacobian_threshold_;
		const ElementInversionCheck check_inversion_;
		const unsigned conservative_max_iter_;
		const double dt_;
		const bool is_volume_;

		StiffnessMatrix cached_stiffness_;                      ///< Cached stiffness matrix for linear elasticity
		mutable std::unique_ptr<utils::MatrixCache> mat_cache_; ///< Matrix cache (mutable because it is modified in second_derivative_unweighted)

		/// @brief Compute the stiffness matrix (cached)
		void compute_cached_stiffness();

		/// @brief Merge a subdivision tree into quadrature_hierarchy_ for the
		/// given element and rebuild its live quadrature if it actually refined.
		/// Shared by post_step() (deferred commit, once a step is accepted) and
		/// max_step_size() (immediate commit, only when step == 0 -- such a step
		/// can never reach post_step(), so deferring it would just discard the
		/// refinement candidate and leave the next iteration stuck on the same
		/// too-coarse quadrature).
		/// @param max_depth Most levels to add in one merge. The continuous check
		/// adds two at a time; the static check merges its whole witness path,
		/// since only the deepest level puts a quadrature point on the violation.
		void commit_refinement(const int id, utils::Tree &subdivision_tree, const int max_depth = 2) const;

		Eigen::VectorXd x_prev_;

		mutable std::vector<utils::Tree> quadrature_hierarchy_;
		int quadrature_order_;

		/// @brief False when displacement and geometry are both linear. The
		/// Jacobian is then constant per element, so the conservative check
		/// still runs but there is nothing between quadrature points to refine
		/// towards: commit_refinement() is a no-op and the quadrature is left as
		/// the bases built it.
		bool refines_quadrature_ = false;

		/// @brief Pending (element id, subdivision tree) candidate from the
		/// last max_step_size() call; replaced (not accumulated) each call,
		/// committed in post_step() once a step is accepted.
		mutable std::optional<std::pair<int, utils::Tree>> pending_refinement_;

		/// @brief Cumulative cost of the conservative checks, reported by
		/// finish() at the end of each nonlinear solve.
		///
		/// The two are called at very different rates -- the static check once
		/// per line-search trial point, the continuous one once per line search
		/// -- so the totals only mean something next to their call counts.
		mutable double static_check_time_ = 0;
		mutable double continuous_check_time_ = 0;
		mutable long static_check_calls_ = 0;
		mutable long continuous_check_calls_ = 0;
		mutable long static_check_rejections_ = 0;
	};
} // namespace polyfem::solver
