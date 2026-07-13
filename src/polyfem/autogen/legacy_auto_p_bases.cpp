#include "legacy_auto_p_bases.hpp"

namespace polyfem {
namespace autogen {

void p_basis_value_2d(const bool bernstein, const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	assert(uv.cols() == 2);
	val.resize(uv.rows(), 1);
	Span<const double> sx(uv.col(0).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sy(uv.col(1).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sz = sx;
	Span<double> out(val.data(), static_cast<std::size_t>(val.rows()));
	p_basis_value_2d(bernstein, p, local_index, sx, sy, sz, out);
}

void p_grad_basis_value_2d(const bool bernstein, const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	assert(uv.cols() == 2);
	val.resize(uv.rows(), 2);
	Span<const double> sx(uv.col(0).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sy(uv.col(1).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sz = sx;
	Span<double> gx(val.col(0).data(), static_cast<std::size_t>(val.rows()));
	Span<double> gy(val.col(1).data(), static_cast<std::size_t>(val.rows()));
	Span<double> gz = gx;
	p_grad_basis_value_2d(bernstein, p, local_index, sx, sy, sz, gx, gy, gz);
}

void p_basis_value_3d(const bool bernstein, const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	assert(uv.cols() == 3);
	val.resize(uv.rows(), 1);
	Span<const double> sx(uv.col(0).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sy(uv.col(1).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sz(uv.col(2).data(), static_cast<std::size_t>(uv.rows()));
	Span<double> out(val.data(), static_cast<std::size_t>(val.rows()));
	p_basis_value_3d(bernstein, p, local_index, sx, sy, sz, out);
}

void p_grad_basis_value_3d(const bool bernstein, const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	assert(uv.cols() == 3);
	val.resize(uv.rows(), 3);
	Span<const double> sx(uv.col(0).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sy(uv.col(1).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sz(uv.col(2).data(), static_cast<std::size_t>(uv.rows()));
	Span<double> gx(val.col(0).data(), static_cast<std::size_t>(val.rows()));
	Span<double> gy(val.col(1).data(), static_cast<std::size_t>(val.rows()));
	Span<double> gz(val.col(2).data(), static_cast<std::size_t>(val.rows()));
	p_grad_basis_value_3d(bernstein, p, local_index, sx, sy, sz, gx, gy, gz);
}

}}
