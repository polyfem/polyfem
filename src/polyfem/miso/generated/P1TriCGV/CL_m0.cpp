#include "../P1TriCGV.hpp"
namespace miso {
	RealVector<3> P1TriCGV::CL_m0(const RealVector<3> &rx, const RealVector<3> &ry, const RealVector<3> &u0x, const RealVector<3> &u0y, const RealVector<3> &u1x, const RealVector<3> &u1y) {
		#pragma region Temporaries
		const Real _t0 = rx[0]*u0y[1];
		const Real _t1 = rx[1]*u0y[2];
		const Real _t2 = rx[2]*u0y[0];
		const Real _t3 = ry[0]*u0x[2];
		const Real _t4 = ry[1]*u0x[0];
		const Real _t5 = ry[2]*u0x[1];
		const Real _t6 = u0x[0]*u0y[1];
		const Real _t7 = u0x[1]*u0y[2];
		const Real _t8 = u0x[2]*u0y[0];
		const Real _t9 = rx[0]*ry[1] - rx[0]*ry[2] - rx[1]*ry[0] + rx[1]*ry[2] + rx[2]*ry[0] - rx[2]*ry[1];
		const Real _t10 = rx[0]*u1y[1];
		const Real _t11 = rx[1]*u1y[2];
		const Real _t12 = rx[2]*u1y[0];
		const Real _t13 = ry[0]*u1x[2];
		const Real _t14 = ry[1]*u1x[0];
		const Real _t15 = ry[2]*u1x[1];
		const Real _t16 = u1x[0]*u1y[1];
		const Real _t17 = u1x[1]*u1y[2];
		const Real _t18 = u1x[2]*u1y[0];
		#pragma endregion
		#pragma region Expressions
		return {
			-_t0 - _t1 - _t2 - _t3 - _t4 - _t5 - _t6 - _t7 - _t8 - _t9 + rx[0]*u0y[2] + rx[1]*u0y[0] + rx[2]*u0y[1] + ry[0]*u0x[1] + ry[1]*u0x[2] + ry[2]*u0x[0] + u0x[0]*u0y[2] + u0x[1]*u0y[0] + u0x[2]*u0y[1],
			-0.5*_t0 - 0.5*_t1 - 0.5*_t10 - 0.5*_t11 - 0.5*_t12 - 0.5*_t13 - 0.5*_t14 - 0.5*_t15 - 0.25*_t16 - 0.25*_t17 - 0.25*_t18 - 0.5*_t2 - 0.5*_t3 - 0.5*_t4 - 0.5*_t5 - 0.25*_t6 - 0.25*_t7 - 0.25*_t8 - _t9 + (0.5)*rx[0]*u0y[2] + (0.5)*rx[0]*u1y[2] + (0.5)*rx[1]*u0y[0] + (0.5)*rx[1]*u1y[0] + (0.5)*rx[2]*u0y[1] + (0.5)*rx[2]*u1y[1] + (0.5)*ry[0]*u0x[1] + (0.5)*ry[0]*u1x[1] + (0.5)*ry[1]*u0x[2] + (0.5)*ry[1]*u1x[2] + (0.5)*ry[2]*u0x[0] + (0.5)*ry[2]*u1x[0] + (0.25)*u0x[0]*u0y[2] - 0.25*u0x[0]*u1y[1] + (0.25)*u0x[0]*u1y[2] + (0.25)*u0x[1]*u0y[0] + (0.25)*u0x[1]*u1y[0] - 0.25*u0x[1]*u1y[2] + (0.25)*u0x[2]*u0y[1] - 0.25*u0x[2]*u1y[0] + (0.25)*u0x[2]*u1y[1] + (0.25)*u0y[0]*u1x[1] - 0.25*u0y[0]*u1x[2] - 0.25*u0y[1]*u1x[0] + (0.25)*u0y[1]*u1x[2] + (0.25)*u0y[2]*u1x[0] - 0.25*u0y[2]*u1x[1] + (0.25)*u1x[0]*u1y[2] + (0.25)*u1x[1]*u1y[0] + (0.25)*u1x[2]*u1y[1],
			-_t10 - _t11 - _t12 - _t13 - _t14 - _t15 - _t16 - _t17 - _t18 - _t9 + rx[0]*u1y[2] + rx[1]*u1y[0] + rx[2]*u1y[1] + ry[0]*u1x[1] + ry[1]*u1x[2] + ry[2]*u1x[0] + u1x[0]*u1y[2] + u1x[1]*u1y[0] + u1x[2]*u1y[1],
		};
		#pragma endregion
	}
}
