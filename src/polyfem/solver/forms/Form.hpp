#pragma once

#include <polyfem/utils/Types.hpp>
#include <polysolve/nonlinear/PostStepData.hpp>

#include <polyfem/assembler/Assembler.hpp>

#include <MeshFEM/SystemAssembler.hh>
#include <MeshFEM/newton_optimizer/NewtonHessian.hh>

#include <filesystem>

namespace polyfem::solver
{
	class Form
	{
	public:
		virtual ~Form() {}

		virtual std::string name() const = 0;

		/// @brief Initialize the form
		/// @param x Current solution
		virtual void init(const Eigen::VectorXd &x) {}
		
		virtual void finish() {}

		/// @brief Compute the value of the form multiplied with the weigth
		/// @param x Current solution
		/// @return Computed value
		inline virtual double value(const Eigen::VectorXd &x) const
		{
			return (weight() / scale_) * value_unweighted(x);
		}

		/// @brief Compute the value of the form multiplied with the weigth
		/// @param x Current solution
		/// @return Computed value
		inline Eigen::VectorXd value_per_element(const Eigen::VectorXd &x) const
		{
			return (weight() / scale_) * value_per_element_unweighted(x);
		}

		/// @brief Compute the first derivative of the value wrt x multiplied with the weigth
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		inline virtual void first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
		{
			first_derivative_unweighted(x, gradv);
			gradv *= weight() / scale_;
		}

		/// @brief Compute the second derivative of the value wrt x multiplied with the weigth
		/// @note This is not marked const because ElasticForm needs to cache the matrix assembly.
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		inline void second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
		{
			second_derivative_unweighted(x, hessian);
			hessian *= weight() / scale_;
		}

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		virtual void second_derivative(const Eigen::VectorXd &x, NewtonHessian &hessian) const
		{
			accumulateHessian(weight() / scale_, x, hessian);
		};

		/// @brief Determine if a step from solution x0 to solution x1 is allowed
		/// @param x0 Current solution
		/// @param x1 Proposed next solution
		/// @return True if the step is allowed
		virtual bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const { return true; }

		/// @brief Determine the maximum step size allowable between the current and next solution
		/// @param x0 Current solution (step size = 0)
		/// @param x1 Next solution (step size = 1)
		/// @return Maximum allowable step size
		virtual double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const { return 1; }

		/// @brief Initialize variables used during the line search
		/// @param x0 Current solution
		/// @param x1 Next solution
		virtual void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) {}

		/// @brief Clear variables used during the line search
		virtual void line_search_end() {}

		/// @brief Update fields after a step in the optimization
		/// @param iter_num Optimization iteration number
		/// @param x Current solution
		/// @param data Data containing info about the current iteration
		virtual void post_step(const polysolve::nonlinear::PostStepData &data) {}

		/// @brief Update cached fields upon a change in the solution
		/// @param new_x New solution
		virtual void solution_changed(const Eigen::VectorXd &new_x) {}

		/// @brief Update time-dependent fields
		/// @param t Current time
		/// @param x Current solution at time t
		virtual void update_quantities(const double t, const Eigen::VectorXd &x) {}

		/// @brief Initialize lagged fields
		/// TODO: more than one step
		/// @param x Current solution
		virtual void init_lagging(const Eigen::VectorXd &x) {};

		/// @brief Update lagged fields
		/// @param x Current solution
		virtual void update_lagging(const Eigen::VectorXd &x, const int iter_num) {};

		/// @brief Get the maximum number of lagging iteration allowable.
		virtual int max_lagging_iterations() const { return 1; }

		/// @brief Does this form require lagging?
		/// @return True if the form requires lagging
		virtual bool uses_lagging() const { return false; }

		/// @brief Set project to psd
		/// @param val If true, the form's second derivative is projected to be positive semidefinite
		virtual void set_project_to_psd(bool val) { project_to_psd_ = val; }

		/// @brief Get if the form's second derivative is projected to psd
		bool is_project_to_psd() const { return project_to_psd_; }

		/// @brief Enable the form
		void enable() { enabled_ = true; }
		/// @brief Disable the form
		void disable() { enabled_ = false; }
		/// @brief Set if the form is enabled
		void set_enabled(const bool enabled) { enabled_ = enabled; }

		/// @brief Determine if the form is enabled
		/// @return True if the form is enabled else false
		bool enabled() const { return enabled_; }

		/// @brief Get the form's multiplicative constant weight
		virtual double weight() const { return weight_; }

		/// @brief Set the form's multiplicative constant weight
		/// @param weight New weight to use
		void set_weight(const double weight) { weight_ = weight; }

		// NOTE: The following functions are really specific to the different form and should be implemented in the derived class.

		/// @brief Checks if the step is collision free
		/// @return True if the step is collision free else false
		virtual bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const { return true; }

		void set_output_dir(const std::string &output_dir) { output_dir_ = output_dir; }

		/// @brief sets the scale for the form
		/// @param scale
		void virtual set_scale(const double scale) { scale_ = scale; }

		// BlockCSCHessian additions
		virtual NewtonHessian hessianSparsityPattern() const { 
			// throw std::runtime_error("hessianSparsityPattern not implemented by" + name());
			return NewtonHessian();	
		}
		virtual bool sparsityPatternIsStatic() const { return false; }
		virtual void accumulateHessian(const double weight, const Eigen::VectorXd &x, NewtonHessian &hessian) const { 
			// throw std::runtime_error("accumulateHessian2D not implemented by" + name()); 
		}

		void setSystemAssembler(std::shared_ptr<SystemAssembler<2>> a) { m_assembler2D = a; }
		void setSystemAssembler(std::shared_ptr<SystemAssembler<3>> a) { m_assembler3D = a; }

		template<class StencilCallable>
		NewtonHessian buildSparsityPattern(size_t num_stencils, StencilCallable &&stencil) const {
			if (m_assembler2D) return m_assembler2D->sparsityPattern(num_stencils, std::forward<StencilCallable>(stencil));
			if (m_assembler3D) return m_assembler3D->sparsityPattern(num_stencils, std::forward<StencilCallable>(stencil));
			throw std::runtime_error("Form has no assembler.");
		}

		template<class EvalCallable, class StencilCallable>
		void accumulateHessianContribs(NewtonHessian &H, size_t num_stencils, EvalCallable &&eval, StencilCallable &&stencil) const {
			if (m_assembler2D) return m_assembler2D->assembleHessian(H, num_stencils, std::forward<EvalCallable>(eval), std::forward<StencilCallable>(stencil));
			if (m_assembler3D) return m_assembler3D->assembleHessian(H, num_stencils, std::forward<EvalCallable>(eval), std::forward<StencilCallable>(stencil));
			throw std::runtime_error("Form has no assembler.");
		}

	protected:
		bool project_to_psd_ = false; ///< If true, the form's second derivative is projected to be positive semidefinite

		double weight_ = 1; ///< weight of the form (e.g., AL penalty weight or Δt²)

		bool enabled_ = true; ///< If true, the form is enabled

		std::string output_dir_;

		std::string resolve_output_path(const std::string &path) const
		{
			if (output_dir_.empty() || path.empty() || std::filesystem::path(path).is_absolute())
				return path;
			return std::filesystem::weakly_canonical(std::filesystem::path(output_dir_) / path).string();
		}

		/// @brief Compute the value of the form
		/// @param x Current solution
		/// @return Computed value
		virtual double value_unweighted(const Eigen::VectorXd &x) const = 0;

		/// @brief Compute the value of the form multiplied per element
		/// @param x Current solution
		/// @return Computed value per element
		virtual Eigen::VectorXd value_per_element_unweighted(const Eigen::VectorXd &x) const
		{
			throw std::runtime_error("Not implemented");
		}

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const = 0;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		virtual void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const = 0;

		mutable std::shared_ptr<SystemAssembler<2>> m_assembler2D;
		mutable std::shared_ptr<SystemAssembler<3>> m_assembler3D;
	private:
		double scale_ = 1; ///< scale of the form
	};
} // namespace polyfem::solver
