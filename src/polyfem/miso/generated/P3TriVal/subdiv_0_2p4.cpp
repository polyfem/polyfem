#include "../P3TriVal.hpp"
namespace miso {
	std::array<RealVector<15>, 4> P3TriVal::subdiv_0_2p4(const RealVector<15> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[0];
		const Real _t1 = (0.5)*_b[1];
		const Real _t2 = (0.25)*_b[0];
		const Real _t3 = (0.25)*_b[2];
		const Real _t4 = (0.125)*_b[0];
		const Real _t5 = (0.375)*_b[2];
		const Real _t6 = (0.125)*_b[3];
		const Real _t7 = (0.0625)*_b[0];
		const Real _t8 = (0.25)*_b[1];
		const Real _t9 = (0.25)*_b[3];
		const Real _t10 = (0.0625)*_b[4];
		const Real _t11 = _t10 + _t5 + _t7 + _t8 + _t9;
		const Real _t12 = (0.5)*_b[5];
		const Real _t13 = (0.25)*_b[5];
		const Real _t14 = (0.25)*_b[6];
		const Real _t15 = _t14 + _t8;
		const Real _t16 = (0.125)*_b[2];
		const Real _t17 = (0.125)*_b[5];
		const Real _t18 = (0.125)*_b[7];
		const Real _t19 = _t17 + _t18;
		const Real _t20 = (0.1875)*_b[6];
		const Real _t21 = _t20 + _t7;
		const Real _t22 = (0.0625)*_b[8];
		const Real _t23 = (0.1875)*_b[7];
		const Real _t24 = (0.0625)*_b[3] + _t23;
		const Real _t25 = _t22 + _t24;
		const Real _t26 = (0.0625)*_b[5];
		const Real _t27 = (0.1875)*_b[2] + _t26;
		const Real _t28 = (0.1875)*_b[1] + _t21 + _t25 + _t27;
		const Real _t29 = (0.25)*_b[9];
		const Real _t30 = (0.125)*_b[9];
		const Real _t31 = (0.125)*_b[1];
		const Real _t32 = (0.125)*_b[10];
		const Real _t33 = _t14 + _t31 + _t32;
		const Real _t34 = (0.0625)*_b[11];
		const Real _t35 = (0.0625)*_b[2];
		const Real _t36 = (0.0625)*_b[9];
		const Real _t37 = _t35 + _t36;
		const Real _t38 = _t34 + _t37;
		const Real _t39 = _t19 + _t33 + _t38 + _t7;
		const Real _t40 = (0.125)*_b[12];
		const Real _t41 = (0.375)*_b[9];
		const Real _t42 = (0.0625)*_b[13];
		const Real _t43 = (0.1875)*_b[10];
		const Real _t44 = (0.0625)*_b[12] + _t43;
		const Real _t45 = _t42 + _t44;
		const Real _t46 = (0.0625)*_b[1];
		const Real _t47 = (0.1875)*_b[9] + _t46;
		const Real _t48 = (0.1875)*_b[5] + _t21 + _t45 + _t47;
		const Real _t49 = (0.25)*_b[12];
		const Real _t50 = (0.0625)*_b[14];
		const Real _t51 = _t13 + _t41 + _t49 + _t50 + _t7;
		const Real _t52 = _t20 + _t26;
		const Real _t53 = (0.1875)*_b[12] + _t42 + _t43 + _t47 + _t50 + _t52;
		const Real _t54 = (0.125)*_b[13];
		const Real _t55 = (0.125)*_b[6];
		const Real _t56 = _t54 + _t55;
		const Real _t57 = (0.25)*_b[10];
		const Real _t58 = _t18 + _t40 + _t57;
		const Real _t59 = _t38 + _t50 + _t56 + _t58;
		const Real _t60 = (0.1875)*_b[11];
		const Real _t61 = (0.1875)*_b[13] + _t25 + _t44 + _t50 + _t60;
		const Real _t62 = (0.375)*_b[11];
		const Real _t63 = (0.25)*_b[13];
		const Real _t64 = (0.25)*_b[8];
		const Real _t65 = _t10 + _t50 + _t62 + _t63 + _t64;
		const Real _t66 = (0.125)*_b[14];
		const Real _t67 = _t49 + _t57;
		const Real _t68 = (0.125)*_b[11];
		const Real _t69 = (0.125)*_b[8];
		const Real _t70 = (0.5)*_b[12];
		const Real _t71 = (0.25)*_b[14];
		const Real _t72 = (0.25)*_b[11];
		const Real _t73 = (0.5)*_b[13];
		const Real _t74 = (0.5)*_b[14];
		const Real _t75 = (0.125)*_b[4];
		const Real _t76 = (0.5)*_b[3];
		const Real _t77 = (0.25)*_b[4];
		const Real _t78 = (0.5)*_b[4];
		const Real _t79 = (0.1875)*_b[3] + _t10 + _t20 + _t22 + _t23 + _t27 + _t46;
		const Real _t80 = (0.25)*_b[7];
		const Real _t81 = _t80 + _t9;
		const Real _t82 = _t55 + _t69;
		const Real _t83 = (0.5)*_b[8];
		const Real _t84 = _t32 + _t6 + _t80;
		const Real _t85 = _t10 + _t38 + _t82 + _t84;
		const Real _t86 = (0.1875)*_b[8] + _t10 + _t24 + _t45 + _t60;
		const Real _t87 = _t34 + _t46 + _t52;
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				_t0 + _t1,
				_t1 + _t2 + _t3,
				(0.375)*_b[1] + _t4 + _t5 + _t6,
				_t11,
				_t0 + _t12,
				_t13 + _t15 + _t2,
				_t15 + _t16 + _t19 + _t4,
				_t28,
				_t12 + _t2 + _t29,
				_t13 + _t30 + _t33 + _t4,
				_t39,
				(0.375)*_b[5] + _t4 + _t40 + _t41,
				_t48,
				_t51,
			},
			{
				_t51,
				_t53,
				_t59,
				_t61,
				_t65,
				(0.375)*_b[12] + _t17 + _t41 + _t66,
				_t30 + _t56 + _t66 + _t67,
				_t58 + _t63 + _t66 + _t68,
				(0.375)*_b[13] + _t62 + _t66 + _t69,
				_t29 + _t70 + _t71,
				_t63 + _t67 + _t71,
				_t71 + _t72 + _t73,
				_t70 + _t74,
				_t73 + _t74,
				_b[14],
			},
			{
				_t11,
				(0.375)*_b[3] + _t31 + _t5 + _t75,
				_t3 + _t76 + _t77,
				_t76 + _t78,
				_b[4],
				_t79,
				_t16 + _t75 + _t81 + _t82,
				_t64 + _t77 + _t81,
				_t78 + _t83,
				_t85,
				_t64 + _t68 + _t75 + _t84,
				_t72 + _t77 + _t83,
				_t86,
				(0.375)*_b[8] + _t54 + _t62 + _t75,
				_t65,
			},
			{
				_t65,
				_t61,
				_t59,
				_t53,
				_t51,
				_t86,
				_t25 + _t37 + _t45 + _t55 + _t68,
				_t18 + _t30 + _t35 + _t45 + _t87,
				_t48,
				_t85,
				_t16 + _t25 + _t32 + _t36 + _t87,
				_t39,
				_t79,
				_t28,
				_t11,
			},
		}};
		#pragma endregion
	}
}
