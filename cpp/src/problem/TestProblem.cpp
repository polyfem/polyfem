////////////////////////////////////////////////////////////////////////////////
#include "TestProblem.hpp"
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

namespace poly_fem {
namespace {

// -----------------------------------------------------------------------------

template<typename T> inline T pow2(T x) { return x*x; }

// -----------------------------------------------------------------------------

template<typename T>
T reentrant_corner(T x, T y, double omega) {
	const double alpha = M_PI/omega;
	const T r = sqrt(x*x+y*y);
	const T theta = atan2(y, x);
	return pow(r, alpha)*sin(alpha*theta);
}

template<typename T>
std::array<T, 2> linear_elasticity_mode_1(T x, T y, double nu, double E, double lambda, double Q) {
	const double kappa = 3.0 - 4.0 * nu;
	const double G = E  / (2.0 * (1.0 + nu));
	const T r = sqrt(x*x+y*y);
	const T theta = atan2(y, x);
	return {{
		1.0 / (2.0*G) * pow(r, lambda) * ((kappa - Q*(lambda + 1)) * cos(lambda * theta) - lambda * cos((lambda - 2)*theta)),
        1.0 / (2.0*G) * pow(r, lambda) * ((kappa + Q*(lambda + 1)) * sin(lambda * theta) + lambda * sin((lambda - 2)*theta)),
	}};
}

template<typename T>
std::array<T, 2> linear_elasticity_mode_2(T x, T y, double nu, double E, double lambda, double Q) {
	const double kappa = 3.0 - 4.0 * nu;
	const double G = E  / (2.0 * (1.0 + nu));
	const T r = sqrt(x*x+y*y);
	const T theta = atan2(y, x);
	return {{
		1.0 / (2.0*G) * pow(r, lambda) * ((kappa - Q*(lambda + 1)) * sin(lambda * theta) - lambda * sin((lambda - 2)*theta)),
        1.0 / (2.0*G) * pow(r, lambda) * ((kappa + Q*(lambda + 1)) * cos(lambda * theta) + lambda * cos((lambda - 2)*theta)),
	}};
}

template<typename T>
T peak(T x, T y, double xc, double yc, double alpha) {
	return exp( -alpha * (pow2(x-xc) + pow2(y-yc)) );
}

template<typename T>
T boundary_line_singularity(T x, T y, double alpha) {
	return pow(x, alpha);
}

template<typename T>
T wave_front(T x, T y, double xc, double yc, double r0, double alpha) {
    const T r = sqrt( pow2(x - xc) + pow2(y - yc) );
    const T one(1);
    return atan2(alpha * (r - r0), one);
}

template<typename T>
T interior_line_singularity(T x, T y, double alpha, double beta) {
	if (x <= beta * (y + 1)) {
		return cos(M_PI*y/2);
	} else {
		return cos(M_PI*y/2) + pow(x - beta*(y+1), alpha);
	}
}

// -----------------------------------------------------------------------------

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

#define PARAM(x) (params_[#x].get<double>())

////////////////////////////////////////////////////////////////////////////////

TestProblem::TestProblem(const std::string &name)
	: ProblemWithSolution(name)
{
	params_ = {
		{"type", "reentrant_corner"},
		{"omega", 7.0 * M_PI / 4.0},
		{"is_scalar", true}
	};
}

template<typename T>
T TestProblem::eval_impl(const T &pt) const {
	T res(is_scalar() ? 1 : 2);
	if (params_["type"] == "reentrant_corner") {
		res(0) = reentrant_corner(pt(0), pt(1), PARAM(omega));
	} else if (params_["type"] == "linear_elasticity_mode_1") {
		auto uv = linear_elasticity_mode_1(pt(0), pt(1), PARAM(nu), PARAM(E), PARAM(lambda), PARAM(Q));
		res(0) = uv[0];
		res(1) = uv[1];
	} else if (params_["type"] == "linear_elasticity_mode_2") {
		auto uv = linear_elasticity_mode_2(pt(0), pt(1), PARAM(nu), PARAM(E), PARAM(lambda), PARAM(Q));
		res(0) = uv[0];
		res(1) = uv[1];
	} else if (params_["type"] == "peak") {
		res(0) = peak(pt(0), pt(1), PARAM(xc), PARAM(yc), PARAM(alpha));
	} else if (params_["type"] == "boundary_line_singularity") {
		res(0) = boundary_line_singularity(pt(0), pt(1), PARAM(alpha));
	} else if (params_["type"] == "wave_front") {
		res(0) = wave_front(pt(0), pt(1), PARAM(xc), PARAM(yc), PARAM(r0), PARAM(alpha));
	} else if (params_["type"] == "interior_line_singularity") {
		res(0) = interior_line_singularity(pt(0), pt(1), PARAM(alpha), PARAM(beta));
	}
	return res;
}

void TestProblem::set_parameters(const json &params) {
	// j_original.merge_patch(j_patch);
	assert(!params.is_null());
	params_.merge_patch(params);
}

} // namespace poly_fem
