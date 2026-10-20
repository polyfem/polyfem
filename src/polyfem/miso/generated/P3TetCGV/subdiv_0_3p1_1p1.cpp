#include "../P3TetCGV.hpp"
namespace miso {
	std::array<RealVector<8>, 16> P3TetCGV::subdiv_0_3p1_1p1(const RealVector<8> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[0];
		const Real _t1 = (0.5)*_b[1];
		const Real _t2 = _t0 + _t1;
		const Real _t3 = (0.5)*_b[2];
		const Real _t4 = _t0 + _t3;
		const Real _t5 = (0.25)*_b[0] + (0.25)*_b[1];
		const Real _t6 = (0.25)*_b[2] + (0.25)*_b[3];
		const Real _t7 = _t5 + _t6;
		const Real _t8 = (0.5)*_b[4];
		const Real _t9 = _t0 + _t8;
		const Real _t10 = (0.25)*_b[4] + (0.25)*_b[5];
		const Real _t11 = _t10 + _t5;
		const Real _t12 = (0.5)*_b[6];
		const Real _t13 = _t0 + _t12;
		const Real _t14 = (0.25)*_b[6] + (0.25)*_b[7];
		const Real _t15 = _t14 + _t5;
		const Real _t16 = (0.5)*_b[3];
		const Real _t17 = _t1 + _t16;
		const Real _t18 = (0.5)*_b[5];
		const Real _t19 = _t1 + _t18;
		const Real _t20 = (0.5)*_b[7];
		const Real _t21 = _t1 + _t20;
		const Real _t22 = _t12 + _t3;
		const Real _t23 = _t14 + _t6;
		const Real _t24 = _t12 + _t8;
		const Real _t25 = _t10 + _t14;
		const Real _t26 = _t12 + _t20;
		const Real _t27 = _t16 + _t20;
		const Real _t28 = _t18 + _t20;
		const Real _t29 = _t3 + _t8;
		const Real _t30 = _t10 + _t6;
		const Real _t31 = _t18 + _t8;
		const Real _t32 = _t16 + _t18;
		const Real _t33 = _t16 + _t3;
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				_t2,
				_t4,
				_t7,
				_t9,
				_t11,
				_t13,
				_t15,
			},
			{
				_t2,
				_b[1],
				_t7,
				_t17,
				_t11,
				_t19,
				_t15,
				_t21,
			},
			{
				_t13,
				_t15,
				_t22,
				_t23,
				_t24,
				_t25,
				_b[6],
				_t26,
			},
			{
				_t15,
				_t21,
				_t23,
				_t27,
				_t25,
				_t28,
				_t26,
				_b[7],
			},
			{
				_t9,
				_t11,
				_t29,
				_t30,
				_b[4],
				_t31,
				_t24,
				_t25,
			},
			{
				_t11,
				_t19,
				_t30,
				_t32,
				_t31,
				_b[5],
				_t25,
				_t28,
			},
			{
				_t4,
				_t7,
				_b[2],
				_t33,
				_t29,
				_t30,
				_t22,
				_t23,
			},
			{
				_t7,
				_t17,
				_t33,
				_b[3],
				_t30,
				_t32,
				_t23,
				_t27,
			},
			{
				_t13,
				_t15,
				_t22,
				_t23,
				_t4,
				_t7,
				_t9,
				_t11,
			},
			{
				_t15,
				_t21,
				_t23,
				_t27,
				_t7,
				_t17,
				_t11,
				_t19,
			},
			{
				_t13,
				_t15,
				_t22,
				_t23,
				_t24,
				_t25,
				_t9,
				_t11,
			},
			{
				_t15,
				_t21,
				_t23,
				_t27,
				_t25,
				_t28,
				_t11,
				_t19,
			},
			{
				_t9,
				_t11,
				_t29,
				_t30,
				_t22,
				_t23,
				_t4,
				_t7,
			},
			{
				_t11,
				_t19,
				_t30,
				_t32,
				_t23,
				_t27,
				_t7,
				_t17,
			},
			{
				_t9,
				_t11,
				_t29,
				_t30,
				_t22,
				_t23,
				_t24,
				_t25,
			},
			{
				_t11,
				_t19,
				_t30,
				_t32,
				_t23,
				_t27,
				_t25,
				_t28,
			},
		}};
		#pragma endregion
	}
}
