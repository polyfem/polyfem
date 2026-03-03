#pragma once

#include "Form.hpp"

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>

#include <polyfem/utils/Jacobian.hpp>
#include <polyfem/utils/MatrixCache.hpp>
#include <polyfem/utils/Types.hpp>

#include <memory>
#include <vector>

namespace polyfem::solver
{
	enum class ElementInversionCheck { Discrete, Conservative };
	NLOHMANN_JSON_SERIALIZE_ENUM(
		polyfem::solver::ElementInversionCheck,
		{{ElementInversionCheck::Discrete, "Discrete"},
		{ElementInversionCheck::Conservative, "Conservative"}})

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
					const ElementInversionCheck check_inversion = ElementInversionCheck::Discrete);

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

		/// @brief Update cached fields upon a change in the solution
		/// @param new_x New solution
		void solution_changed(const Eigen::VectorXd &new_x) override;

		/// @brief Reset adaptive quadrature refinement after each complete nonlinear solve.
		void finish() override;

	private:
		const int n_bases_;
		std::vector<basis::ElementBases> &bases_;
		const std::vector<basis::ElementBases> &geom_bases_;

		const assembler::Assembler &assembler_; ///< Reference to the assembler
		assembler::AssemblyValsCache &ass_vals_cache_;
		double t_;
		const double jacobian_threshold_;
		const ElementInversionCheck check_inversion_;
		const double dt_;
		const bool is_volume_;

		StiffnessMatrix cached_stiffness_;                      ///< Cached stiffness matrix for linear elasticity
		mutable std::unique_ptr<utils::MatrixCache> mat_cache_; ///< Matrix cache (mutable because it is modified in second_derivative_unweighted)

		/// @brief Compute the stiffness matrix (cached)
		void compute_cached_stiffness();

		Eigen::VectorXd x_prev_;

		mutable std::vector<utils::Tree> quadrature_hierarchy_;
		int quadrature_order_;
	};
} // namespace polyfem::solver
