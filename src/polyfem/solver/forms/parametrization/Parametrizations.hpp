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

		int size(const int x_size) const override { return x_size; }
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const int from_, to_;
	};

	class PowerMap : public Parametrization
	{
	public:
		PowerMap(const double power = 1, const int from = -1, const int to = -1) : power_(power), from_(from), to_(to) { assert(from_ < to_ || from_ < 0); assert(power_ > 0); }

		int size(const int x_size) const override { return x_size; }
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const double power_;
		const int from_, to_;
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
		PerBody(const mesh::Mesh &mesh);

		int size(const int x_size) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const mesh::Mesh &mesh_;
		int full_size_;
		int reduced_size_;
		std::map<int, std::array<int, 2>> body_id_map_; // from body_id to {elem_id, index}
	};

	class SliceMap : public Parametrization
	{
	public:
		SliceMap(const int from, const int to);

		int size(const int x_size) const override { return from_ - to_; }

		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

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
