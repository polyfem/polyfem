#include "legacy_auto_pyramid_bases.hpp"

namespace polyfem {
namespace autogen {

void pyramid_basis_value_3d(const int pyramid, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	assert(uv.cols() == 3);
	val.resize(uv.rows(), 1);
	Span<const double> sx(uv.col(0).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sy(uv.col(1).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sz(uv.col(2).data(), static_cast<std::size_t>(uv.rows()));
	Span<double> out(val.data(), static_cast<std::size_t>(val.rows()));
	pyramid_basis_value_3d(pyramid, local_index, sx, sy, sz, out);
}

void pyramid_grad_basis_value_3d(const int pyramid, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	assert(uv.cols() == 3);
	val.resize(uv.rows(), 3);
	Span<const double> sx(uv.col(0).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sy(uv.col(1).data(), static_cast<std::size_t>(uv.rows()));
	Span<const double> sz(uv.col(2).data(), static_cast<std::size_t>(uv.rows()));
	Span<double> gx(val.col(0).data(), static_cast<std::size_t>(val.rows()));
	Span<double> gy(val.col(1).data(), static_cast<std::size_t>(val.rows()));
	Span<double> gz(val.col(2).data(), static_cast<std::size_t>(val.rows()));
	pyramid_grad_basis_value_3d(pyramid, local_index, sx, sy, sz, gx, gy, gz);
}

}}
