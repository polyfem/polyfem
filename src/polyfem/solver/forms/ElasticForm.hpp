#pragma once

#include "Form.hpp"

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	/// @brief Form of the elasticity potential and forces
	class ElasticForm : public Form
	{
	public:
		/// @brief Construct a new Elastic Form object
		/// @param state Reference to the simulation state
		ElasticForm(const int n_bases,
					const std::vector<basis::ElementBases> &bases,
					const std::vector<basis::ElementBases> &geom_bases,
					const assembler::Assembler &assembler,
					const assembler::AssemblyValsCache &ass_vals_cache,
					const double dt,
					const bool is_volume);

		std::string name() const override { return "elastic"; }

	protected:
		/// @brief Compute the elastic potential value
		/// @param x Current solution
		/// @return Value of the elastic potential
		virtual double value_unweighted(const Eigen::VectorXd &x) const override;

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

		/// @brief Update time-dependent fields
		/// @param t Current time
		/// @param x Current solution at time t
		void update_quantities(const double t, const Eigen::VectorXd &x) override { x_prev_ = x; }

		/// @brief Compute the derivative of the force wrt lame/damping parameters, then multiply the resulting matrix with adjoint_sol.
		/// @param[in] x Current solution
		/// @param[in] adjoint Current adjoint solution
		/// @param[out] term Derivative of force multiplied by the adjoint
		void force_material_derivative(const Eigen::MatrixXd &x, const Eigen::MatrixXd &x_prev, const Eigen::MatrixXd &adjoint, Eigen::VectorXd &term);

		/// @brief Compute the derivative of the force wrt vertex positions, then multiply the resulting matrix with adjoint_sol.
		/// @param[in] n_verts Number of vertices
		/// @param[in] x Current solution
		/// @param[in] adjoint Current adjoint solution
		/// @param[out] term Derivative of force multiplied by the adjoint
		void force_shape_derivative(const int n_verts, const Eigen::MatrixXd &x, const Eigen::MatrixXd &x_prev, const Eigen::MatrixXd &adjoint, Eigen::VectorXd &term);

	private:
		const int n_bases_;
		const std::vector<basis::ElementBases> &bases_;
		const std::vector<basis::ElementBases> &geom_bases_;

		const assembler::Assembler &assembler_; ///< Reference to the assembler
		const assembler::AssemblyValsCache &ass_vals_cache_;
		const double dt_;
		const bool is_volume_;

		StiffnessMatrix cached_stiffness_;           ///< Cached stiffness matrix for linear elasticity
		mutable utils::SparseMatrixCache mat_cache_; ///< Matrix cache (mutable because it is modified in second_derivative_unweighted)

		/// @brief Compute the stiffness matrix (cached)
		void compute_cached_stiffness();

		Eigen::VectorXd x_prev_;
	};
} // namespace polyfem::solver
