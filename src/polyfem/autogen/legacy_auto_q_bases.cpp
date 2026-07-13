#include "legacy_auto_q_bases.hpp"

namespace polyfem {
namespace autogen {

void q_basis_value_1d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	assert(uv.cols() == 1);
	val.resize(uv.rows(), 1);
	Span<const double> sx(uv.col(0).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sy = sx;
	Span<const double> sz = sx;
	Span<double> out(val.data(), static_cast<std::size_t>(val.rows()));
	q_basis_value_1d(q, local_index, sx, sy, sz, out);
}

void q_grad_basis_value_1d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	assert(uv.cols() == 1);
	val.resize(uv.rows(), 1);
	Span<const double> sx(uv.col(0).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sy = sx;
	Span<const double> sz = sx;
	Span<double> gx(val.col(0).data(), static_cast<std::size_t>(val.rows()));
	Span<double> gy = gx;
	Span<double> gz = gx;
	q_grad_basis_value_1d(q, local_index, sx, sy, sz, gx, gy, gz);
}

void q_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	assert(uv.cols() == 2);
	val.resize(uv.rows(), 1);
	Span<const double> sx(uv.col(0).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sy(uv.col(1).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sz = sx;
	Span<double> out(val.data(), static_cast<std::size_t>(val.rows()));
	q_basis_value_2d(q, local_index, sx, sy, sz, out);
}

void q_grad_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	assert(uv.cols() == 2);
	val.resize(uv.rows(), 2);
	Span<const double> sx(uv.col(0).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sy(uv.col(1).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sz = sx;
	Span<double> gx(val.col(0).data(), static_cast<std::size_t>(val.rows()));
	Span<double> gy(val.col(1).data(), static_cast<std::size_t>(val.rows()));
	Span<double> gz = gx;
	q_grad_basis_value_2d(q, local_index, sx, sy, sz, gx, gy, gz);
}

void q_basis_value_3d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	assert(uv.cols() == 3);
	val.resize(uv.rows(), 1);
	Span<const double> sx(uv.col(0).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sy(uv.col(1).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sz(uv.col(2).data(), static_cast<std::size_t>(uv.rows()));
	Span<double> out(val.data(), static_cast<std::size_t>(val.rows()));
	q_basis_value_3d(q, local_index, sx, sy, sz, out);
}

void q_grad_basis_value_3d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	assert(uv.cols() == 3);
	val.resize(uv.rows(), 3);
	Span<const double> sx(uv.col(0).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sy(uv.col(1).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sz(uv.col(2).data(), static_cast<std::size_t>(uv.rows()));
	Span<double> gx(val.col(0).data(), static_cast<std::size_t>(val.rows()));
	Span<double> gy(val.col(1).data(), static_cast<std::size_t>(val.rows()));
	Span<double> gz(val.col(2).data(), static_cast<std::size_t>(val.rows()));
	q_grad_basis_value_3d(q, local_index, sx, sy, sz, gx, gy, gz);
}

}}
