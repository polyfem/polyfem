#pragma once

#include "Parameterization.hpp"

#include <polyfem/mesh/Mesh.hpp>

#include <Eigen/Core>
#include <map>

namespace polyfem::solver
{
	class ParameterizationFactory
	{
	private:
		ParameterizationFactory() {}

	public:
		static std::vector<std::shared_ptr<Parameterization>> build(const json &params, const int full_size);
	};

	class ExponentialMap : public Parameterization
	{
	public:
		ExponentialMap(const int from = -1, const int to = -1);

		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const int from_, to_;
	};

	class LambdaMu2ENu : public Parameterization
	{
	public:
		LambdaMu2ENu(const bool is_volume);

		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const bool is_volume_;
	};

	class PerBody : public Parameterization
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
} // namespace polyfem::solver
