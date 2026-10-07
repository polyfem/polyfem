#include "../P1TriCGV.hpp"
namespace miso {
	RealVector<3> P1TriCGV::CL_m0(const RealVector<3> &p0x, const RealVector<3> &p0y, const RealVector<3> &p1x, const RealVector<3> &p1y) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*p0x[0] + (0.5)*p1x[0];
		const Real _t1 = (0.5)*p0y[0] + (0.5)*p1y[0];
		#pragma endregion
		#pragma region Expressions
		return {
			-(-p0x[0] + p0x[1])*(-p0y[0] + p0y[2]) + (-p0x[0] + p0x[2])*(-p0y[0] + p0y[1]),
			-(-_t0 + (0.5)*p0x[1] + (0.5)*p1x[1])*(-_t1 + (0.5)*p0y[2] + (0.5)*p1y[2]) + (-_t0 + (0.5)*p0x[2] + (0.5)*p1x[2])*(-_t1 + (0.5)*p0y[1] + (0.5)*p1y[1]),
			-(-p1x[0] + p1x[1])*(-p1y[0] + p1y[2]) + (-p1x[0] + p1x[2])*(-p1y[0] + p1y[1]),
		};
		#pragma endregion
	}
}
