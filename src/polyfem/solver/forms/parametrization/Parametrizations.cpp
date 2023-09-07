#include "Parametrizations.hpp"
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/Common.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

namespace polyfem::solver
{
	std::vector<std::shared_ptr<Parametrization>> ParametrizationFactory::build(const json &params, const int full_size)
	{
		return std::vector<std::shared_ptr<Parametrization>>();
	}

	ExponentialMap::ExponentialMap(const int from, const int to)
		: from_(from), to_(to)
	{
		assert(from_ < to_ || from_ < 0);
	}

	Eigen::VectorXd ExponentialMap::inverse_eval(const Eigen::VectorXd &y)
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
		assert(from_ < to_ || from_ < 0);
		assert(scale_ != 0);
	}

	Eigen::VectorXd Scaling::inverse_eval(const Eigen::VectorXd &y)
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

	Eigen::VectorXd PowerMap::inverse_eval(const Eigen::VectorXd &y)
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

	Eigen::VectorXd ENu2LambdaMu::inverse_eval(const Eigen::VectorXd &y)
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
		std::map<int, int> body_id_map;
		for (int e = 0; e < mesh.n_elements(); e++)
		{
			const int body_id = mesh.get_body_id(e);
			if (!body_id_map.count(body_id))
			{
				body_id_map[body_id] = reduced_size_;
				reduced_size_++;
			}
		}
		logger().info("{} objects found!", reduced_size_);

		node_id_to_body_id_.resize(full_size_);
		node_id_to_body_id_.setConstant(-1);
		for (int e = 0; e < bases.size(); e++)
		{
			const int body_id = mesh_.get_body_id(e);
			const int id = body_id_map.at(body_id);
			for (const auto &bs : bases[e].bases)
			{
				for (const auto &g : bs.global())
				{
					if (node_id_to_body_id_(g.index) < 0)
						node_id_to_body_id_(g.index) = id;
					else if (node_id_to_body_id_(g.index) != id)
						log_and_throw_error("Same node on different bodies!");
				}
			}
		}
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
				y(i * dim + d) = x(node_id_to_body_id_(i) * dim + d);
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
				grad_body(node_id_to_body_id_(i) * dim + d) += grad(i * dim + d);
			}
		}

		return grad_body;
	}

	PerBody2PerElem::PerBody2PerElem(const mesh::Mesh &mesh) : mesh_(mesh), full_size_(mesh_.n_elements())
	{
		reduced_size_ = 0;
		for (int e = 0; e < mesh.n_elements(); e++)
		{
			const int body_id = mesh.get_body_id(e);
			if (!body_id_map_.count(body_id))
			{
				body_id_map_[body_id] = {{e, reduced_size_}};
				reduced_size_++;
			}
		}
		logger().info("{} objects found!", reduced_size_);
	}

	Eigen::VectorXd PerBody2PerElem::eval(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd y;
		y.setZero(size(x.size()));

		for (int e = 0; e < mesh_.n_elements(); e++)
		{
			const int body_id = mesh_.get_body_id(e);
			const auto &entry = body_id_map_.at(body_id);
			// y(e) = x(entry[1]);
			y(Eigen::seq(e, y.size() - 1, full_size_)) = x(Eigen::seq(entry[1], x.size() - 1, reduced_size_));
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
			const int body_id = mesh_.get_body_id(e);
			const auto &entry = body_id_map_.at(body_id);
			// grad_body(entry[1]) += grad(e);
			grad_body(Eigen::seq(entry[1], x.size() - 1, reduced_size_)) += grad(Eigen::seq(e, grad.size() - 1, full_size_));
		}

		return grad_body;
	}

	SliceMap::SliceMap(const int from, const int to, const int total) : from_(from), to_(to), total_(total)
	{
		if (to_ - from_ < 0)
			log_and_throw_error("Invalid Slice Map input!");
	}

	Eigen::VectorXd SliceMap::inverse_eval(const Eigen::VectorXd &y)
	{
		if (total_ == -1)
			return y;
		else
		{
			if (y.size() != size(0))
				log_and_throw_error("Inverse eval on SliceMap is inconsistent in size!");
			Eigen::VectorXd y_;
			y_.setZero(total_);
			y_.segment(from_, to_ - from_) = y;
			return y_;
		}
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

	InsertConstantMap::InsertConstantMap(const int size, const double val, const int start_index): start_index_(start_index)
	{
		if (size <= 0)
			log_and_throw_error("Invalid InsertConstantMap input!");
		values_.setConstant(size, val);
	}

	InsertConstantMap::InsertConstantMap(const Eigen::VectorXd &values) : values_(values)
	{
	}

	int InsertConstantMap::size(const int x_size) const
	{
		return x_size + values_.size();
	}

	Eigen::VectorXd InsertConstantMap::inverse_eval(const Eigen::VectorXd &y)
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

	Eigen::VectorXd LinearFilter::eval(const Eigen::VectorXd &x) const
	{
		assert(x.size() == tt_radius_adjacency.rows());
		return (tt_radius_adjacency * x).array() / tt_radius_adjacency_row_sum.array();
	}

	Eigen::VectorXd LinearFilter::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		assert(x.size() == tt_radius_adjacency.rows());
		return (tt_radius_adjacency * grad).array() / tt_radius_adjacency_row_sum.array();
	}

	CustomSymmetric::CustomSymmetric(const json &args)
	{
		for (const auto &entry : args["fixed_entries"])
			fixed_entries.push_back(entry.get<int>());

		for (const auto &pair : args["equal_pairs"])
			equal_pairs.emplace_back(pair[0].get<int>(), pair[1].get<int>());

		for (const auto &pair : args["sum_equal_pairs"])
			sum_equal_pairs.emplace_back(pair[0].get<int>(), pair[1].get<int>());
	}
	int CustomSymmetric::size(const int x_size) const
	{
		return x_size;
	}
	Eigen::VectorXd CustomSymmetric::eval(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd y = x;
		// y(8) = 1.0 - y(3);
		// y(9) = y(4);
		// y(10) = 1.0 - y(1);
		// y(11) = y(2);
		// y(18) = y(15);
		// y(19) = y(14);
		// y(5) = 0.5;
		// y(6) = 0.5;

		for (const auto &pair : equal_pairs)
			y(pair.second) = y(pair.first);

		for (const auto &pair : sum_equal_pairs)
			y(pair.second) = 1.0 - y(pair.first);

		return y;
	}
	Eigen::VectorXd CustomSymmetric::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd grad_new = grad;
		// grad_new(3) -= grad_new(8);
		// grad_new(4) += grad_new(9);
		// grad_new(1) -= grad_new(10);
		// grad_new(2) += grad_new(11);
		// grad_new(15) += grad_new(18);
		// grad_new(14) += grad_new(19);

		// grad_new({5,6,8,9,10,11,18,19}).setZero();

		grad_new(fixed_entries).setZero();

		for (const auto &pair : equal_pairs)
		{
			grad_new(pair.first) += grad_new(pair.second);
			grad_new(pair.second) = 0;
		}

		for (const auto &pair : sum_equal_pairs)
		{
			grad_new(pair.first) -= grad_new(pair.second);
			grad_new(pair.second) = 0;
		}

		return grad_new;
	}
} // namespace polyfem::solver
