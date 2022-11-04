#pragma once

#include "ElasticParameter.hpp"
#include "ShapeParameter.hpp"

namespace polyfem::solver
{
	class Objective
	{
	public:
		Objective(const json &args);
		virtual ~Objective() = default;

		virtual double value() const = 0;
		Eigen::VectorXd gradient(const State& state, const Parameter &param) const
		{
			return param.project(compute_partial_gradient(param) + compute_adjoint_term(state, param));
		}

		Eigen::VectorXd gradient(const Parameter &param) const
		{
			log_and_throw_error("Shouldn't use this!");
			return Eigen::VectorXd();
		}

		virtual Eigen::VectorXd compute_adjoint_rhs(const State& state) const = 0; // compute $\partial_u J$

	protected:
		virtual Eigen::VectorXd compute_partial_gradient(const Parameter &param) const = 0; // compute $\partial_q J$
		static  Eigen::VectorXd compute_adjoint_term(const State& state, const Parameter &param);
	};

	class StressObjective: public Objective
	{
	public:
		StressObjective(const State &state, const std::shared_ptr<const ElasticParameter> &elastic_param, const json &args);
		~StressObjective() = default;

		double value() const override;
		Eigen::VectorXd compute_adjoint_rhs(const State& state) const override;

	protected:
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;

		const State &state_;
		IntegrableFunctional j_;
		int power_;
		std::string formulation_;

		std::shared_ptr<const ElasticParameter> elastic_param_; // stress needs elastic param to evaluate

		std::string transient_integral_type_;
		std::set<int> interested_ids_;
	};

	class SumObjective: public Objective
	{
	public:
		SumObjective(const json &args);
		~SumObjective() = default;

		double value() const override;
		Eigen::VectorXd compute_adjoint_rhs(const State& state) const override;

	protected:
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;
		
		std::vector<Objective> objs;
	};

	class BoundarySmoothingObjective: public Objective
	{
	public:
		BoundarySmoothingObjective(const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		~BoundarySmoothingObjective() = default;
		
		void init(const std::shared_ptr<const ShapeParameter> shape_param);

		double value() const override;
		Eigen::VectorXd compute_adjoint_rhs(const State& state) const override;

	protected:
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;

		std::shared_ptr<const ShapeParameter> shape_param_;

		const json args_;

        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
		
		std::vector<bool> active_mask;
		std::vector<int> boundary_nodes;
		Eigen::SparseMatrix<bool, Eigen::RowMajor> adj;
		Eigen::SparseMatrix<double, Eigen::RowMajor> L;
	};
} // namespace polyfem::solver