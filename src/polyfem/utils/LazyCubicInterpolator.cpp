#include "LazyCubicInterpolator.hpp"
#include <mutex>

namespace polyfem
{
	void LazyCubicInterpolator::bicubic_interpolation(const Eigen::MatrixXd &corner_point, const std::vector<std::string> &keys, const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad) const
	{
		Eigen::MatrixXd corner_val(4, 1);
		Eigen::MatrixXd corner_grad(4, 2);
		Eigen::MatrixXd corner_grad_grad(4, 1);
		for (int i = 0; i < 4; ++i)
		{
			{
				std::shared_lock distance_lock(distance_mutex_);
				corner_val(i) = implicit_function_distance.at(keys[i]);
			}
			{
				std::shared_lock grad_lock(grad_mutex_);
				auto mixed_grads = implicit_function_grads.at(keys[i]);
				corner_grad(i, 0) = mixed_grads(0);
				corner_grad(i, 1) = mixed_grads(1);
				corner_grad_grad(i, 0) = mixed_grads(2);
			}
		}

		Eigen::MatrixXd x(16, 1);
		x << corner_val(0), corner_val(1), corner_val(2), corner_val(3),
			delta_ * corner_grad(0, 0), delta_ * corner_grad(1, 0), delta_ * corner_grad(2, 0), delta_ * corner_grad(3, 0),
			delta_ * corner_grad(0, 1), delta_ * corner_grad(1, 1), delta_ * corner_grad(2, 1), delta_ * corner_grad(3, 1),
			delta_ * delta_ * corner_grad_grad(0, 0), delta_ * delta_ * corner_grad_grad(1, 0), delta_ * delta_ * corner_grad_grad(2, 0), delta_ * delta_ * corner_grad_grad(3, 0);

		Eigen::MatrixXd coeffs = cubic_mat * x;

		auto bar_x = [&corner_point](double x_) { return (x_ - corner_point(0, 0)) / (corner_point(1, 0) - corner_point(0, 0)); };
		auto bar_y = [&corner_point](double y_) { return (y_ - corner_point(0, 1)) / (corner_point(2, 1) - corner_point(0, 1)); };

		val = 0;
		grad.setZero(2, 1);
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
			{
				val += coeffs(i + j * 4) * pow(bar_x(point(0)), i) * pow(bar_y(point(1)), j);
				grad(0) += i == 0 ? 0 : (coeffs(i + j * 4) * i * pow(bar_x(point(0)), i - 1) * pow(bar_y(point(1)), j));
				grad(1) += j == 0 ? 0 : coeffs(i + j * 4) * pow(bar_x(point(0)), i) * j * pow(bar_y(point(1)), j - 1);
			}

		grad(0) /= (corner_point(1, 0) - corner_point(0, 0));
		grad(1) /= (corner_point(2, 1) - corner_point(0, 1));

		assert(!std::isnan(grad(0)) && !std::isnan(grad(1)));
	}

	void LazyCubicInterpolator::tricubic_interpolation(const Eigen::MatrixXd &corner_point, const std::vector<std::string> &keys, const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad) const
	{
		Eigen::MatrixXd corner_val(8, 1);
		Eigen::MatrixXd corner_grad(8, 3);
		Eigen::MatrixXd corner_grad_grad(8, 3);
		Eigen::MatrixXd corner_grad_grad_grad(8, 1);
		for (int i = 0; i < 8; ++i)
		{
			{
				std::shared_lock distance_lock(distance_mutex_);
				corner_val(i) = implicit_function_distance.at(keys[i]);
			}
			{
				std::shared_lock grad_lock(grad_mutex_);
				auto mixed_grads = implicit_function_grads.at(keys[i]);
				corner_grad(i, 0) = mixed_grads(0);
				corner_grad(i, 1) = mixed_grads(1);
				corner_grad(i, 2) = mixed_grads(2);
				corner_grad_grad(i, 0) = mixed_grads(3);
				corner_grad_grad(i, 1) = mixed_grads(4);
				corner_grad_grad(i, 2) = mixed_grads(5);
				corner_grad_grad_grad(i, 0) = mixed_grads(6);
			}
		}
		Eigen::MatrixXd x(64, 1);
		x << corner_val(0), corner_val(1), corner_val(2), corner_val(3), corner_val(4), corner_val(5), corner_val(6), corner_val(7),
			delta_ * corner_grad(0, 0), delta_ * corner_grad(1, 0), delta_ * corner_grad(2, 0), delta_ * corner_grad(3, 0), delta_ * corner_grad(4, 0), delta_ * corner_grad(5, 0), delta_ * corner_grad(6, 0), delta_ * corner_grad(7, 0),
			delta_ * corner_grad(0, 1), delta_ * corner_grad(1, 1), delta_ * corner_grad(2, 1), delta_ * corner_grad(3, 1), delta_ * corner_grad(4, 1), delta_ * corner_grad(5, 1), delta_ * corner_grad(6, 1), delta_ * corner_grad(7, 1),
			delta_ * corner_grad(0, 2), delta_ * corner_grad(1, 2), delta_ * corner_grad(2, 2), delta_ * corner_grad(3, 2), delta_ * corner_grad(4, 2), delta_ * corner_grad(5, 2), delta_ * corner_grad(6, 2), delta_ * corner_grad(7, 2),
			delta_ * delta_ * corner_grad_grad(0, 0), delta_ * delta_ * corner_grad_grad(1, 0), delta_ * delta_ * corner_grad_grad(2, 0), delta_ * delta_ * corner_grad_grad(3, 0), delta_ * delta_ * corner_grad_grad(4, 0), delta_ * delta_ * corner_grad_grad(5, 0), delta_ * delta_ * corner_grad_grad(6, 0), delta_ * delta_ * corner_grad_grad(7, 0),
			delta_ * delta_ * corner_grad_grad(0, 1), delta_ * delta_ * corner_grad_grad(1, 1), delta_ * delta_ * corner_grad_grad(2, 1), delta_ * delta_ * corner_grad_grad(3, 1), delta_ * delta_ * corner_grad_grad(4, 1), delta_ * delta_ * corner_grad_grad(5, 1), delta_ * delta_ * corner_grad_grad(6, 1), delta_ * delta_ * corner_grad_grad(7, 1),
			delta_ * delta_ * corner_grad_grad(0, 2), delta_ * delta_ * corner_grad_grad(1, 2), delta_ * delta_ * corner_grad_grad(2, 2), delta_ * delta_ * corner_grad_grad(3, 2), delta_ * delta_ * corner_grad_grad(4, 2), delta_ * delta_ * corner_grad_grad(5, 2), delta_ * delta_ * corner_grad_grad(6, 2), delta_ * delta_ * corner_grad_grad(7, 2),
			delta_ * delta_ * delta_ * corner_grad_grad_grad(0, 0), delta_ * delta_ * delta_ * corner_grad_grad_grad(1, 0), delta_ * delta_ * delta_ * corner_grad_grad_grad(2, 0), delta_ * delta_ * delta_ * corner_grad_grad_grad(3, 0), delta_ * delta_ * delta_ * corner_grad_grad_grad(4, 0), delta_ * delta_ * delta_ * corner_grad_grad_grad(5, 0), delta_ * delta_ * delta_ * corner_grad_grad_grad(6, 0), delta_ * delta_ * delta_ * corner_grad_grad_grad(7, 0);

		Eigen::MatrixXd coeffs = cubic_mat * x;

		auto bar_x = [&corner_point](double x_) { return (x_ - corner_point(0, 0)) / (corner_point(1, 0) - corner_point(0, 0)); };
		auto bar_y = [&corner_point](double y_) { return (y_ - corner_point(0, 1)) / (corner_point(2, 1) - corner_point(0, 1)); };
		auto bar_z = [&corner_point](double z_) { return (z_ - corner_point(0, 2)) / (corner_point(4, 2) - corner_point(0, 2)); };

		val = 0;
		grad.setZero(3, 1);
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				for (int l = 0; l < 4; ++l)
				{
					val += coeffs(i + j * 4 + l * 16) * pow(bar_x(point(0)), i) * pow(bar_y(point(1)), j) * pow(bar_z(point(2)), l);
					grad(0) += i == 0 ? 0 : (coeffs(i + j * 4 + l * 16) * i * pow(bar_x(point(0)), i - 1) * pow(bar_y(point(1)), j)) * pow(bar_z(point(2)), l);
					grad(1) += j == 0 ? 0 : coeffs(i + j * 4 + l * 16) * pow(bar_x(point(0)), i) * j * pow(bar_y(point(1)), j - 1) * pow(bar_z(point(2)), l);
					grad(2) += l == 0 ? 0 : coeffs(i + j * 4 + l * 16) * pow(bar_x(point(0)), i) * pow(bar_y(point(1)), j) * l * pow(bar_z(point(2)), l - 1);
				}

		grad(0) /= (corner_point(1, 0) - corner_point(0, 0));
		grad(1) /= (corner_point(2, 1) - corner_point(0, 1));
		grad(2) /= (corner_point(4, 2) - corner_point(0, 2));

		assert(!std::isnan(grad(0)) && !std::isnan(grad(1)) && !std::isnan(grad(2)));
	}

	void LazyCubicInterpolator::cache_grid(std::function<void(const Eigen::MatrixXd &, double &)> compute_distance, const Eigen::MatrixXd &point)
	{
		Eigen::MatrixXi keys;
		build_corner_keys(point, keys);
		std::vector<std::string> keys_string;

		auto safe_compute_distance = [this, compute_distance](const Eigen::MatrixXd &clamped_point, const std::string &key_string) {
			std::unique_lock lock(distance_mutex_);
			if (implicit_function_distance.count(key_string) == 0)
				compute_distance(clamped_point, implicit_function_distance[key_string]);
		};
		auto safe_distance = [this, safe_compute_distance](const Eigen::VectorXi &key) {
			Eigen::MatrixXd point;
			std::string key_string;
			double distance;
			setup_key(key, key_string, point);
			safe_compute_distance(point, key_string);
			{
				std::shared_lock lock(distance_mutex_);
				distance = implicit_function_distance[key_string];
			}
			return distance;
		};
		auto centered_fd = [this, safe_distance](const Eigen::VectorXi &key, const int k) {
			Eigen::MatrixXi key_plus, key_minus;
			double distance_plus, distance_minus;
			key_plus = key;
			key_plus(k) += 1;
			distance_plus = safe_distance(key_plus);
			key_minus = key;
			key_minus(k) -= 1;
			distance_minus = safe_distance(key_minus);
			return (1. / 2. / delta_) * (distance_plus - distance_minus);
		};
		auto centered_mixed_fd = [this, centered_fd](const Eigen::VectorXi &key, const int k1, const int k2) {
			Eigen::VectorXi key_plus, key_minus;
			key_plus = key;
			key_plus(k1) += 1;
			key_minus = key;
			key_minus(k1) -= 1;
			return (1. / 2. / delta_) * (centered_fd(key_plus, k2) - centered_fd(key_minus, k2));
		};
		auto centered_mixed_fd_3d = [this, centered_mixed_fd](const Eigen::VectorXi &key) {
			Eigen::VectorXi key_plus, key_minus;
			key_plus = key;
			key_plus(0) += 1;
			key_minus = key;
			key_minus(0) -= 1;
			return (1. / 2. / delta_) * (centered_mixed_fd(key_plus, 1, 2) - centered_mixed_fd(key_minus, 1, 2));
		};
		auto compute_grad = [this, centered_fd, centered_mixed_fd, centered_mixed_fd_3d](const Eigen::VectorXi &key) {
			Eigen::VectorXd mixed_grads(dim_ == 2 ? 3 : 7);
			if (dim_ == 2)
			{
				mixed_grads(0) = centered_fd(key, 0);
				mixed_grads(1) = centered_fd(key, 1);
				mixed_grads(2) = centered_mixed_fd(key, 0, 1);
			}
			else if (dim_ == 3)
			{
				mixed_grads(0) = centered_fd(key, 0);
				mixed_grads(1) = centered_fd(key, 1);
				mixed_grads(2) = centered_fd(key, 2);
				mixed_grads(3) = centered_mixed_fd(key, 0, 1);
				mixed_grads(4) = centered_mixed_fd(key, 0, 2);
				mixed_grads(5) = centered_mixed_fd(key, 1, 2);
				mixed_grads(6) = centered_mixed_fd_3d(key);
			}
			return mixed_grads;
		};
		auto safe_compute_grad = [this, compute_grad](const Eigen::VectorXi &key) {
			std::string key_string;
			Eigen::MatrixXd point;
			setup_key(key, key_string, point);
			{
				std::unique_lock lock(grad_mutex_);
				if (implicit_function_grads.count(key_string) == 0)
				{
					Eigen::MatrixXd grad = compute_grad(key);
					implicit_function_grads[key_string] = grad;
				}
			}
		};
		Eigen::MatrixXd corner_point(keys.rows(), dim_);
		for (int i = 0; i < keys.rows(); ++i)
		{
			std::string key_string;
			Eigen::MatrixXd clamped_point;
			setup_key(keys.row(i), key_string, clamped_point);
			safe_compute_distance(clamped_point, key_string);
			keys_string.push_back(key_string);
			corner_point.row(i) = clamped_point.transpose();
			safe_compute_grad(keys.row(i));
		}
	}

	void LazyCubicInterpolator::evaluate(const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad) const
	{
		Eigen::MatrixXi keys;
		build_corner_keys(point, keys);

		std::vector<std::string> keys_string;
		Eigen::MatrixXd corner_point(keys.rows(), dim_);
		for (int i = 0; i < keys.rows(); ++i)
		{
			std::string key_string;
			Eigen::MatrixXd clamped_point;
			setup_key(keys.row(i), key_string, clamped_point);
			keys_string.push_back(key_string);
			corner_point.row(i) = clamped_point.transpose();
		}

		grad.setZero(dim_, 1);
		if (dim_ == 2)
			bicubic_interpolation(corner_point, keys_string, point, val, grad);
		else if (dim_ == 3)
			tricubic_interpolation(corner_point, keys_string, point, val, grad);

		for (int i = 0; i < dim_; ++i)
			if (std::isnan(grad(i)))
				throw std::runtime_error("Nan found in gradient computation.");
	}

	void LazyCubicInterpolator::lazy_evaluate(std::function<void(const Eigen::MatrixXd &, double &)> compute_distance, const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad)
	{
		cache_grid(compute_distance, point);

		evaluate(point, val, grad);
	}

} // namespace polyfem