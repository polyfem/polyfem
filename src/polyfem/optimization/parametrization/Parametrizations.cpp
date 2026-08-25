#include "Parametrizations.hpp"

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/Common.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <string_view>

namespace polyfem::solver
{
	namespace
	{
		constexpr std::string_view ERR_STRING =
			"Please check your variable to simulation and composition chain.";

		void check_from_to(int from, int to, std::string_view name)
		{
			// If from != -1, [from, to] implies a range.
			// range size must > 0.
			if (from >= 0 && from >= to)
			{
				log_and_throw_adjoint_error(
					"Invalid composition {}. Reason: [from, to] = [{}, {}] is not a valid range. {}", name, from, to, ERR_STRING);
			}
		}

		void check_non_empty(int size, std::string_view name)
		{
			if (size <= 0)
				log_and_throw_adjoint_error(
					"Invalid composition {}. Reason: Empty or negative optimization parameter DOF {}. {}", name, size, ERR_STRING);
		}

	} // namespace

	ExponentialMap::ExponentialMap(const int from, const int to)
		: from_(from), to_(to)
	{
		check_from_to(from, to, "exp");
	}

	int ExponentialMap::inverse_size(int y_size) const
	{
		check_non_empty(y_size, "ExponentialMap::inverse_size");
		return y_size;
	}

	int ExponentialMap::size(const int x_size) const
	{
		return x_size;
	}

	Eigen::VectorXd ExponentialMap::inverse_eval(const Eigen::VectorXd &y) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd res = y;
			res.segment(from_, to_ - from_) = y.segment(from_, to_ - from_).array().log();
			return res;
		}
		else
			return y.array().log();
	}

	Eigen::VectorXd ExponentialMap::eval(const Eigen::VectorXd &x) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd y = x;
			y.segment(from_, to_ - from_) = x.segment(from_, to_ - from_).array().exp();
			return y;
		}
		else
			return x.array().exp();
	}

	Eigen::VectorXd ExponentialMap::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd res = grad.array();
			res.segment(from_, to_ - from_) = x.segment(from_, to_ - from_).array().exp() * grad.segment(from_, to_ - from_).array();
			return res;
		}
		else
			return x.array().exp() * grad.array();
	}

	Scaling::Scaling(const double scale, const int from, const int to)
		: from_(from), to_(to), scale_(scale)
	{
		check_from_to(from, to, "scale");
	}

	int Scaling::inverse_size(int y_size) const
	{
		check_non_empty(y_size, "Scaling::inverse_size");
		return y_size;
	}

	int Scaling::size(const int x_size) const
	{
		return x_size;
	}

	Eigen::VectorXd Scaling::inverse_eval(const Eigen::VectorXd &y) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd res = y;
			res.segment(from_, to_ - from_) = y.segment(from_, to_ - from_).array() / scale_;
			return res;
		}
		else
			return y.array() / scale_;
	}

	Eigen::VectorXd Scaling::eval(const Eigen::VectorXd &x) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd y = x;
			y.segment(from_, to_ - from_) = x.segment(from_, to_ - from_).array() * scale_;
			return y;
		}
		else
			return x.array() * scale_;
	}

	Eigen::VectorXd Scaling::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd res = grad.array();
			res.segment(from_, to_ - from_) = scale_ * grad.segment(from_, to_ - from_).array();
			return res;
		}
		else
			return scale_ * grad.array();
	}

	PowerMap::PowerMap(const double power, const int from, const int to)
		: power_(power), from_(from), to_(to)
	{
		check_from_to(from, to, "power");
	}

	int PowerMap::inverse_size(int y_size) const
	{
		check_non_empty(y_size, "PowerMap::inverse_size");
		return y_size;
	}

	int PowerMap::size(const int x_size) const
	{
		return x_size;
	}

	Eigen::VectorXd PowerMap::inverse_eval(const Eigen::VectorXd &y) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd res = y;
			res.segment(from_, to_ - from_) = y.segment(from_, to_ - from_).array().pow(1. / power_);
			return res;
		}
		else
			return y.array().pow(1. / power_);
	}

	Eigen::VectorXd PowerMap::eval(const Eigen::VectorXd &x) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd y = x;
			y.segment(from_, to_ - from_) = x.segment(from_, to_ - from_).array().pow(power_);
			return y;
		}
		else
			return x.array().pow(power_);
	}

	Eigen::VectorXd PowerMap::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd res = grad;
			res.segment(from_, to_ - from_) = grad.segment(from_, to_ - from_).array() * x.segment(from_, to_ - from_).array().pow(power_ - 1) * power_;
			return res;
		}
		else
			return grad.array() * x.array().pow(power_ - 1) * power_;
	}

	ENu2LambdaMu::ENu2LambdaMu(const bool is_volume)
		: is_volume_(is_volume)
	{
	}

	int ENu2LambdaMu::inverse_size(int y_size) const
	{
		check_non_empty(y_size, "E-nu-to-lambda-mu");
		if (y_size % 2 != 0)
		{
			log_and_throw_adjoint_error("Invalid composition E-nu-to-lambda-mu. Reason: Expect output dof be multiple of 2 but instead get {}. {}", y_size, ERR_STRING);
		}
		return y_size;
	}

	int ENu2LambdaMu::size(const int x_size) const
	{
		return x_size;
	}

	Eigen::VectorXd ENu2LambdaMu::inverse_eval(const Eigen::VectorXd &y) const
	{
		const int size = y.size() / 2;
		assert(size * 2 == y.size());

		Eigen::VectorXd x(y.size());
		for (int i = 0; i < size; i++)
		{
			x(i) = convert_to_E(is_volume_, y(i), y(i + size));
			x(i + size) = convert_to_nu(is_volume_, y(i), y(i + size));
		}

		return x;
	}

	Eigen::VectorXd ENu2LambdaMu::eval(const Eigen::VectorXd &x) const
	{
		const int size = x.size() / 2;
		assert(size * 2 == x.size());

		Eigen::VectorXd y;
		y.setZero(x.size());
		for (int i = 0; i < size; i++)
		{
			y(i) = convert_to_lambda(is_volume_, x(i), x(i + size));
			y(i + size) = convert_to_mu(x(i), x(i + size));
		}

		return y;
	}

	Eigen::VectorXd ENu2LambdaMu::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		const int size = grad.size() / 2;
		assert(size * 2 == grad.size());
		assert(size * 2 == x.size());

		Eigen::VectorXd grad_E_nu;
		grad_E_nu.setZero(grad.size());
		for (int i = 0; i < size; i++)
		{
			const Eigen::Matrix2d jacobian = d_lambda_mu_d_E_nu(is_volume_, x(i), x(i + size));
			grad_E_nu(i) = grad(i) * jacobian(0, 0) + grad(i + size) * jacobian(1, 0);
			grad_E_nu(i + size) = grad(i) * jacobian(0, 1) + grad(i + size) * jacobian(1, 1);
		}

		return grad_E_nu;
	}

	PerBody2PerNode::PerBody2PerNode(const mesh::Mesh &mesh, const std::vector<basis::ElementBases> &bases, const int n_bases) : mesh_(mesh), bases_(bases), full_size_(n_bases)
	{
		reduced_size_ = 0;
		std::map<int, int> body_id_to_compacted_id;
		for (int e = 0; e < mesh.n_elements(); e++)
		{
			const int body_id = mesh.get_body_id(e);
			if (!body_id_to_compacted_id.count(body_id))
			{
				body_id_to_compacted_id[body_id] = reduced_size_;
				reduced_size_++;
			}
		}
		logger().info("{} objects found!", reduced_size_);

		compacted_body_node_num_ = Eigen::VectorXi::Zero(reduced_size_);
		node_id_to_compacted_body_ = Eigen::VectorXi::Constant(full_size_, -1);
		for (int e = 0; e < bases.size(); e++)
		{
			const int body_id = mesh_.get_body_id(e);
			const int id = body_id_to_compacted_id.at(body_id);
			for (const auto &bs : bases[e].bases)
			{
				for (const auto &g : bs.global())
				{
					if (node_id_to_compacted_body_(g.index) < 0)
					{
						compacted_body_node_num_(id)++;
						node_id_to_compacted_body_(g.index) = id;
					}
					else if (node_id_to_compacted_body_(g.index) != id)
					{
						log_and_throw_adjoint_error("Same node on different bodies!");
					}
				}
			}
		}
	}

	int PerBody2PerNode::inverse_size(int y_size) const
	{
		check_non_empty(y_size, "PerBody2PerNode::inverse_size");
		if (y_size % full_size_ != 0)
		{
			log_and_throw_adjoint_error("Invalid composition per-body-to-per-node. Reason: Expect output dof be multiple of mesh node num but instead get mesh node num = {} and output size = {}. {}", full_size_, y_size, ERR_STRING);
		}
		int dim = y_size / full_size_;
		return reduced_size_ * dim;
	}

	Eigen::VectorXd PerBody2PerNode::inverse_eval(const Eigen::VectorXd &y) const
	{
		// Inverse does not exists. Choose average as a good enough alternative.

		Eigen::VectorXd x = Eigen::VectorXd::Zero(inverse_size(y.size()));
		assert(y.size() % full_size_ == 0);
		int dim = y.size() / full_size_;
		for (int i = 0; i < full_size_; i++)
		{
			for (int d = 0; d < dim; d++)
			{
				x(node_id_to_compacted_body_(i) * dim + d) += y(i * dim + d);
			}
		}
		for (int i = 0; i < reduced_size_; i++)
		{
			for (int d = 0; d < dim; d++)
			{
				assert(compacted_body_node_num_(i) != 0);
				x(i * dim + d) /= static_cast<double>(compacted_body_node_num_(i));
			}
		}

		return x;
	}

	Eigen::VectorXd PerBody2PerNode::eval(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd y;
		y.setZero(size(x.size()));
		const int dim = x.size() / reduced_size_;

		for (int i = 0; i < full_size_; i++)
		{
			for (int d = 0; d < dim; d++)
			{
				y(i * dim + d) = x(node_id_to_compacted_body_(i) * dim + d);
			}
		}

		return y;
	}

	int PerBody2PerNode::size(const int x_size) const
	{
		assert(x_size % reduced_size_ == 0);
		return (x_size / reduced_size_) * full_size_;
	}

	Eigen::VectorXd PerBody2PerNode::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		assert(grad.size() == size(x.size()));
		Eigen::VectorXd grad_body;
		grad_body.setZero(x.size());
		const int dim = x.size() / reduced_size_;

		for (int i = 0; i < full_size_; i++)
		{
			for (int d = 0; d < dim; d++)
			{
				grad_body(node_id_to_compacted_body_(i) * dim + d) += grad(i * dim + d);
			}
		}

		return grad_body;
	}

	PerBody2PerElem::PerBody2PerElem(const mesh::Mesh &mesh) : mesh_(mesh), full_size_(mesh_.n_elements())
	{
		reduced_size_ = 0;
		std::map<int, int> compacted_body_id_map;
		for (int e = 0; e < mesh.n_elements(); e++)
		{
			const int body_id = mesh.get_body_id(e);
			if (!compacted_body_id_map.count(body_id))
			{
				compacted_body_id_map[body_id] = reduced_size_;
				reduced_size_++;
			}
		}
		logger().info("{} objects found!", reduced_size_);

		compacted_body_elem_num_ = Eigen::VectorXi::Zero(reduced_size_);
		elem_id_to_compacted_body_id_ = Eigen::VectorXi::Constant(full_size_, -1);
		for (int e = 0; e < mesh.n_elements(); e++)
		{
			const int body_id = mesh_.get_body_id(e);
			const int id = compacted_body_id_map.at(body_id);
			compacted_body_elem_num_(id)++;
			elem_id_to_compacted_body_id_(e) = id;
		}
	}

	int PerBody2PerElem::inverse_size(int y_size) const
	{
		check_non_empty(y_size, "PerBody2PerElem::inverse_size");
		if (y_size % full_size_ != 0)
		{
			log_and_throw_adjoint_error("Invalid composition per-body-to-per-elem. Reason: Expect output dof be multiple of mesh element num but instead get mesh element num = {} and output size = {}. {}", full_size_, y_size, ERR_STRING);
		}
		int dim = y_size / full_size_;
		return reduced_size_ * dim;
	}

	Eigen::VectorXd PerBody2PerElem::inverse_eval(const Eigen::VectorXd &y) const
	{
		// Inverse does not exists. Choose average as a good enough alternative.

		Eigen::VectorXd x = Eigen::VectorXd::Zero(inverse_size(y.size()));
		assert(y.size() % full_size_ == 0);
		int dim = y.size() / full_size_;
		for (int e = 0; e < mesh_.n_elements(); e++)
		{
			int id = elem_id_to_compacted_body_id_(e);
			x(Eigen::seq(id, x.size() - 1, reduced_size_)) += y(Eigen::seq(e, y.size() - 1, full_size_));
		}
		for (int i = 0; i < reduced_size_; i++)
		{
			assert(compacted_body_elem_num_(i) != 0);
			x(Eigen::seq(i, x.size() - 1, reduced_size_)) /= compacted_body_elem_num_(i);
		}

		return x;
	}

	Eigen::VectorXd PerBody2PerElem::eval(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd y;
		y.setZero(size(x.size()));

		for (int e = 0; e < mesh_.n_elements(); e++)
		{
			const auto &id = elem_id_to_compacted_body_id_(e);
			y(Eigen::seq(e, y.size() - 1, full_size_)) = x(Eigen::seq(id, x.size() - 1, reduced_size_));
		}

		return y;
	}

	int PerBody2PerElem::size(const int x_size) const
	{
		assert(x_size % reduced_size_ == 0);
		return (x_size / reduced_size_) * full_size_;
	}

	Eigen::VectorXd PerBody2PerElem::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		assert(grad.size() == size(x.size()));
		Eigen::VectorXd grad_body;
		grad_body.setZero(x.size());

		for (int e = 0; e < mesh_.n_elements(); e++)
		{
			int id = elem_id_to_compacted_body_id_(e);
			grad_body(Eigen::seq(id, x.size() - 1, reduced_size_)) += grad(Eigen::seq(e, grad.size() - 1, full_size_));
		}

		return grad_body;
	}

	SliceMap::SliceMap(const int from, const int to, const int total) : from_(from), to_(to), total_(total)
	{
		if (to_ - from_ < 0)
			log_and_throw_adjoint_error("Invalid Slice Map input!");
	}

	int SliceMap::inverse_size(int y_size) const
	{
		if (total_ == -1)
		{
			log_and_throw_adjoint_error("SliceMap with unknown total is impossible to inverse!");
		}
		if (y_size != to_ - from_)
		{
			log_and_throw_adjoint_error("Invalid composition slice. Reason: Output DOF {} and [from, to] = [{}, {}] size mismatch. {}", y_size, from_, to_, ERR_STRING);
		}
		return total_;
	}

	Eigen::VectorXd SliceMap::inverse_eval(const Eigen::VectorXd &y) const
	{
		if (total_ == -1)
		{
			log_and_throw_adjoint_error("SliceMap with unknown total is impossible to inverse!");
		}
		if (y.size() != size(0))
		{
			log_and_throw_adjoint_error("Inverse eval on SliceMap is inconsistent in size!");
		}

		Eigen::VectorXd y_;
		y_.setZero(total_);
		y_.segment(from_, to_ - from_) = y;
		return y_;
	}

	Eigen::VectorXd SliceMap::eval(const Eigen::VectorXd &x) const
	{
		return x.segment(from_, to_ - from_);
	}

	Eigen::VectorXd SliceMap::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd grad_full;
		grad_full.setZero(x.size());
		grad_full.segment(from_, to_ - from_) = grad;
		return grad_full;
	}

	InsertConstantMap::InsertConstantMap(const int size, const double val, const int start_index) : start_index_(start_index)
	{
		if (size <= 0)
			log_and_throw_adjoint_error("Invalid InsertConstantMap input!");
		values_.setConstant(size, val);
	}

	InsertConstantMap::InsertConstantMap(const Eigen::VectorXd &values, const int start_index) : start_index_(start_index), values_(values)
	{
	}

	int InsertConstantMap::size(const int x_size) const
	{
		return x_size + values_.size();
	}

	int InsertConstantMap::inverse_size(int y_size) const
	{
		if (y_size < values_.size())
		{
			log_and_throw_adjoint_error("Invalid composition append-const. Reason: Output DOF {} is smaller than append size {}. {}", y_size, values_.size(), ERR_STRING);
		}
		return y_size - values_.size();
	}

	Eigen::VectorXd InsertConstantMap::inverse_eval(const Eigen::VectorXd &y) const
	{
		if (start_index_ >= 0)
		{
			Eigen::VectorXd x(y.size() - values_.size());
			if (start_index_ > 0)
				x.head(start_index_) = y.head(start_index_);
			x.tail(x.size() - start_index_) = y.tail(y.size() - start_index_ - values_.size());
			return x;
		}
		else
			return y.head(y.size() - values_.size());
	}

	Eigen::VectorXd InsertConstantMap::eval(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd y;
		y.setZero(size(x.size()));
		if (start_index_ >= 0)
		{
			if (start_index_ > 0)
				y.head(start_index_) = x.head(start_index_);
			y.segment(start_index_, values_.size()) = values_;
			y.tail(y.size() - start_index_ - values_.size()) = x.tail(x.size() - start_index_);
		}
		else
			y << x, values_;

		return y;
	}

	Eigen::VectorXd InsertConstantMap::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		assert(x.size() == grad.size() - values_.size());
		Eigen::VectorXd reduced_grad(grad.size() - values_.size());
		if (start_index_ >= 0)
		{
			if (start_index_ > 0)
				reduced_grad.head(start_index_) = grad.head(start_index_);
			reduced_grad.tail(reduced_grad.size() - start_index_) = grad.tail(grad.size() - start_index_ - values_.size());
		}
		else
			reduced_grad = grad.head(grad.size() - values_.size());

		return reduced_grad;
	}

	LinearFilter::LinearFilter(const mesh::Mesh &mesh, const double radius)
	{
		std::vector<Eigen::Triplet<double>> tt_adjacency_list;

		Eigen::MatrixXd barycenters;
		if (mesh.is_volume())
			mesh.cell_barycenters(barycenters);
		else
			mesh.face_barycenters(barycenters);

		RowVectorNd min, max;
		mesh.bounding_box(min, max);
		// TODO: more efficient way
		for (int i = 0; i < barycenters.rows(); i++)
		{
			auto center_i = barycenters.row(i);
			for (int j = 0; j <= i; j++)
			{
				auto center_j = barycenters.row(j);
				double dist = 0;
				dist = (center_i - center_j).norm();
				if (dist < radius)
				{
					tt_adjacency_list.emplace_back(i, j, radius - dist);
					if (i != j)
						tt_adjacency_list.emplace_back(j, i, radius - dist);
				}
			}
		}
		tt_radius_adjacency.resize(barycenters.rows(), barycenters.rows());
		tt_radius_adjacency.setFromTriplets(tt_adjacency_list.begin(), tt_adjacency_list.end());

		tt_radius_adjacency_row_sum.setZero(tt_radius_adjacency.rows());
		for (int i = 0; i < tt_radius_adjacency.rows(); i++)
			tt_radius_adjacency_row_sum(i) = tt_radius_adjacency.row(i).sum();
	}

	int LinearFilter::inverse_size(int y_size) const
	{
		check_non_empty(y_size, "LinearFilter::inverse_size");
		if (y_size != tt_radius_adjacency.rows())
		{
			log_and_throw_adjoint_error("Invalid composition linear-filter. Reason: Output DOF {} and mesh element count {} mismatch. {}", y_size, tt_radius_adjacency.rows(), ERR_STRING);
		}
		return y_size;
	}

	int LinearFilter::size(const int x_size) const
	{
		return x_size;
	}

	Eigen::VectorXd LinearFilter::eval(const Eigen::VectorXd &x) const
	{
		assert(x.size() == tt_radius_adjacency.rows());
		return (tt_radius_adjacency * x).array() / tt_radius_adjacency_row_sum.array();
	}

	Eigen::VectorXd LinearFilter::inverse_eval(const Eigen::VectorXd &y) const
	{
		// No inverse exists. Choose identity as reasonable alternative.
		return y;
	}

	Eigen::VectorXd LinearFilter::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		assert(x.size() == tt_radius_adjacency.rows());
		return (tt_radius_adjacency * grad).array() / tt_radius_adjacency_row_sum.array();
	}

	ScalarVelocityParametrization::ScalarVelocityParametrization(const double start_val, const double dt) : start_val_(start_val), dt_(dt) {}

	int ScalarVelocityParametrization::inverse_size(int y_size) const
	{
		check_non_empty(y_size, "ScalarVelocityParametrization::inverse_size");
		return y_size;
	}

	int ScalarVelocityParametrization::size(const int x_size) const
	{
		return x_size;
	}

	Eigen::VectorXd ScalarVelocityParametrization::inverse_eval(const Eigen::VectorXd &y) const
	{
		Eigen::VectorXd x;
		x.setZero(size(y.size()));
		x(0) = (y(0) - start_val_) / dt_;
		for (int i = 1; i < x.size(); ++i)
			x(i) = (y(i) - y(i - 1)) / dt_;
		return x;
	}

	Eigen::VectorXd ScalarVelocityParametrization::eval(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd y;
		y.setZero(size(x.size()));
		y(0) = start_val_ + dt_ * x(0);
		for (int i = 1; i < x.size(); ++i)
			y(i) = y(i - 1) + dt_ * x(i);
		return y;
	}

	Eigen::VectorXd ScalarVelocityParametrization::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		assert(x.size() == grad.size());

		Eigen::MatrixXd hess;
		hess.setZero(x.size(), size(x.size()));
		for (int i = 0; i < hess.rows(); ++i)
			for (int j = 0; j <= i; ++j)
				hess(i, j) = dt_;

		return hess.transpose() * grad;
	}

} // namespace polyfem::solver
