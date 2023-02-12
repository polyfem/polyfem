#pragma once

#include "Parametrization.hpp"

#include <polyfem/mesh/Mesh.hpp>

#include <Eigen/Core>
#include <map>

namespace polyfem::solver
{
	class ParametrizationFactory
	{
	private:
		ParametrizationFactory() {}

	public:
		static std::vector<std::shared_ptr<Parametrization>> build(const json &params, const int full_size);
	};

	class ExponentialMap : public Parametrization
	{
	public:
		ExponentialMap(const int from = -1, const int to = -1);

		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const int from_, to_;
	};

	class PowerMap : public Parametrization
	{
	public:
		PowerMap(const double power = 1, const int from = -1, const int to = -1) : power_(power), from_(from), to_(to) { assert(from_ < to_); assert(power_ > 0); }

		int size(const int x_size) const override { return x_size; }
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const int from_, to_;
		const double power_;
	};

	class ENu2LambdaMu : public Parametrization
	{
	public:
		ENu2LambdaMu(const bool is_volume);

		int size(const int x_size) const override { return x_size; }

		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const bool is_volume_;
	};

	class PerBody : public Parametrization
	{
	public:
		PerBody(const bool is_volume);

		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		std::map<int, std::array<int, 2>> body_id_map_; // from body_id to {elem_id, index}
		int n_elem_;
		const int full_size_;
		const int reduced_size_;
		const mesh::Mesh &mesh_;
	};

	class SliceMap : public Parametrization
	{
	public:
		SliceMap(const int from = -1, const int to = -1): from_(from), to_(to)
		{
			if (to_ - from_ <= 0)
				log_and_throw_error("Invalid Slice Map input!");
		}

		int size(const int x_size) const override { return from_ - to_; }

		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override
		{
			return x.segment(from_, to_ - from_);
		}
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override
		{
			return grad.segment(from_, to_ - from_);
		}

	private:
		const int from_, to_;
	};

	class AppendConstantMap : public Parametrization
	{
	public:
		AppendConstantMap(const int size = -1, const double val = 0);

		int size(const int x_size) const override;
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const int size_;
		const double val_;
	};
} // namespace polyfem::solver
