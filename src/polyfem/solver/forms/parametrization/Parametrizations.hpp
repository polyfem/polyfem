#pragma once

#include "Parametrization.hpp"
#include <polyfem/Common.hpp>
#include <map>

namespace polyfem::mesh
{
	class Mesh;
}

namespace polyfem::basis
{
	class ElementBases;
}

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
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const int from_, to_;
	};

	class Scaling : public Parametrization
	{
	public:
		Scaling(const double scale, const int from = -1, const int to = -1);

		int size(const int x_size) const override { return x_size; }
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const int from_, to_;
		const double scale_;
	};

	class PowerMap : public Parametrization
	{
	public:
		PowerMap(const double power = 1, const int from = -1, const int to = -1) : power_(power), from_(from), to_(to)
		{
			assert(from_ < to_ || from_ < 0);
			assert(power_ > 0);
		}

		int size(const int x_size) const override { return x_size; }
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override;
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

		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const bool is_volume_;
	};

	class PerBody2PerNode : public Parametrization
	{
	public:
		PerBody2PerNode(const mesh::Mesh &mesh, const std::vector<basis::ElementBases> &bases, const int n_bases);

		int size(const int x_size) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const mesh::Mesh &mesh_;
		const std::vector<basis::ElementBases> &bases_;
		int full_size_;
		int reduced_size_;
		Eigen::VectorXi node_id_to_body_id_;
	};

	class PerBody2PerElem : public Parametrization
	{
	public:
		PerBody2PerElem(const mesh::Mesh &mesh);

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
		SliceMap(const int from = -1, const int to = -1, const int total = -1);

		int size(const int x_size) const override { return to_ - from_; }

		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const int from_, to_, total_;
	};

	class InsertConstantMap : public Parametrization
	{
	public:
		InsertConstantMap(const int size = -1, const double val = 0, const int start_index = -1);
		InsertConstantMap(const Eigen::VectorXd &values);

		int size(const int x_size) const override;
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		// const int size_;
		// const double val_;
		int start_index_ = -1;
		Eigen::VectorXd values_;
	};

	class LinearFilter : public Parametrization
	{
	public:
		LinearFilter(const mesh::Mesh &mesh, const double radius);

		int size(const int x_size) const override { return x_size; }
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		Eigen::SparseMatrix<double> tt_radius_adjacency;
		Eigen::VectorXd tt_radius_adjacency_row_sum;
	};

	class CustomSymmetric : public Parametrization
	{
	public:
		CustomSymmetric(const json &args);

		int size(const int x_size) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		std::vector<int> fixed_entries;
		std::vector<std::pair<int, int>> equal_pairs;
		std::vector<std::pair<int, int>> sum_equal_pairs;
	};

} // namespace polyfem::solver
