#pragma once

#include "Parametrization.hpp"
#include <polyfem/Common.hpp>

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
	class ExponentialMap : public Parametrization
	{
	public:
		ExponentialMap(const int from = -1, const int to = -1);

		int inverse_size(int y_size) const override;
		int size(const int x_size) const override;
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const int from_, to_;
	};

	class Scaling : public Parametrization
	{
	public:
		Scaling(const double scale, const int from = -1, const int to = -1);

		int inverse_size(int y_size) const override;
		int size(const int x_size) const override;
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const int from_, to_;
		const double scale_;
	};

	class PowerMap : public Parametrization
	{
	public:
		PowerMap(const double power = 1, const int from = -1, const int to = -1);

		int inverse_size(int y_size) const override;
		int size(const int x_size) const override;
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

		int inverse_size(int y_size) const override;
		int size(const int x_size) const override;
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const bool is_volume_;
	};

	/// @brief Map per body to per FE node in node major layout (x1 y1 z1 x2 y2 z2...)
	///
	/// The order of the input body is pseudo randomly determined by the element order of input mesh,
	/// *Which is different from body id*.
	class PerBody2PerNode : public Parametrization
	{
	public:
		PerBody2PerNode(const mesh::Mesh &mesh, const std::vector<basis::ElementBases> &bases, const int n_bases);

		int size(const int x_size) const override;
		int inverse_size(int y_size) const override;
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const mesh::Mesh &mesh_;
		const std::vector<basis::ElementBases> &bases_;
		int full_size_;                             /// FE node num.
		int reduced_size_;                          /// Body num.
		Eigen::VectorXi compacted_body_node_num_;   /// Number of nodes of a body.
		Eigen::VectorXi node_id_to_compacted_body_; /// FE node index to body index.
	};

	/// @brief Map per body to per element in dim major layout (x1 x2 ... y1 y1 ... z1 z2 ...)
	class PerBody2PerElem : public Parametrization
	{
	public:
		PerBody2PerElem(const mesh::Mesh &mesh);

		int size(const int x_size) const override;
		int inverse_size(int y_size) const override;
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const mesh::Mesh &mesh_;
		int full_size_;                                /// Element num.
		int reduced_size_;                             /// Body num.
		Eigen::VectorXi compacted_body_elem_num_;      /// Number if elements of a body.
		Eigen::VectorXi elem_id_to_compacted_body_id_; /// Element index to body index.
	};

	class SliceMap : public Parametrization
	{
	public:
		SliceMap(const int from = -1, const int to = -1, const int total = -1);

		int size(const int x_size) const override { return to_ - from_; }
		int inverse_size(int y_size) const override;
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const int from_, to_, total_;
	};

	class InsertConstantMap : public Parametrization
	{
	public:
		InsertConstantMap(const int size = -1, const double val = 0, const int start_index = -1);
		InsertConstantMap(const Eigen::VectorXd &values, const int start_index = -1);

		int size(const int x_size) const override;
		int inverse_size(int y_size) const override;
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		// const int size_;
		// const double val_;
		int start_index_ = -1;
		Eigen::VectorXd values_;
	};

	/// @brief Maps to average of neighboring
	class LinearFilter : public Parametrization
	{
	public:
		LinearFilter(const mesh::Mesh &mesh, const double radius);

		int size(const int x_size) const override;
		int inverse_size(int y_size) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		Eigen::SparseMatrix<double> tt_radius_adjacency;
		Eigen::VectorXd tt_radius_adjacency_row_sum;
	};

	class ScalarVelocityParametrization : public Parametrization
	{
	public:
		ScalarVelocityParametrization(const double start_val, const double dt);

		int size(const int x_size) const override;
		int inverse_size(int y_size) const override;
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

	private:
		const double start_val_;
		const double dt_;
	};
} // namespace polyfem::solver
