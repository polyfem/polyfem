#include "../P2TriCGV.hpp"
namespace miso {
	RealVector<18> P2TriCGV::CL_m0(const RealVector<6> &p0x, const RealVector<6> &p0y, const RealVector<6> &p1x, const RealVector<6> &p1y) {
		#pragma region Temporaries
		const Real _t0 = 3*p0x[0];
		const Real _t1 = -4*p0x[1];
		const Real _t2 = 3*p0y[0];
		const Real _t3 = -4*p0y[3];
		const Real _t4 = -4*p0x[3];
		const Real _t5 = -4*p0y[1];
		const Real _t6 = (1.5)*p0x[0] + (1.5)*p1x[0];
		const Real _t7 = 2*p0x[1];
		const Real _t8 = -_t7;
		const Real _t9 = 2*p1x[1];
		const Real _t10 = -_t9;
		const Real _t11 = _t10 + _t8;
		const Real _t12 = (0.5)*p0x[2];
		const Real _t13 = (0.5)*p1x[2];
		const Real _t14 = _t12 + _t13;
		const Real _t15 = (1.5)*p0y[0] + (1.5)*p1y[0];
		const Real _t16 = 2*p0y[3];
		const Real _t17 = -_t16;
		const Real _t18 = 2*p1y[3];
		const Real _t19 = -_t18;
		const Real _t20 = _t17 + _t19;
		const Real _t21 = (0.5)*p0y[5];
		const Real _t22 = (0.5)*p1y[5];
		const Real _t23 = _t21 + _t22;
		const Real _t24 = 2*p0x[3];
		const Real _t25 = -_t24;
		const Real _t26 = 2*p1x[3];
		const Real _t27 = -_t26;
		const Real _t28 = _t25 + _t27;
		const Real _t29 = (0.5)*p0x[5];
		const Real _t30 = (0.5)*p1x[5];
		const Real _t31 = _t29 + _t30;
		const Real _t32 = 2*p0y[1];
		const Real _t33 = -_t32;
		const Real _t34 = 2*p1y[1];
		const Real _t35 = -_t34;
		const Real _t36 = _t33 + _t35;
		const Real _t37 = (0.5)*p0y[2];
		const Real _t38 = (0.5)*p1y[2];
		const Real _t39 = _t37 + _t38;
		const Real _t40 = 3*p1x[0];
		const Real _t41 = -4*p1x[1];
		const Real _t42 = 3*p1y[0];
		const Real _t43 = -4*p1y[3];
		const Real _t44 = -4*p1x[3];
		const Real _t45 = -4*p1y[1];
		const Real _t46 = p0x[0] - p0x[2];
		const Real _t47 = p0y[0] - 2*p0y[4];
		const Real _t48 = _t17 + p0y[5];
		const Real _t49 = p0y[0] - p0y[2];
		const Real _t50 = p0x[0] - 2*p0x[4];
		const Real _t51 = _t25 + p0x[5];
		const Real _t52 = (0.5)*p0x[0] + (0.5)*p1x[0];
		const Real _t53 = -_t12 - _t13 + _t52;
		const Real _t54 = (0.5)*p0y[0] + (0.5)*p1y[0];
		const Real _t55 = _t54 - p0y[4] - p1y[4];
		const Real _t56 = -p0y[3] - p1y[3];
		const Real _t57 = _t23 + _t56;
		const Real _t58 = -_t37 - _t38 + _t54;
		const Real _t59 = _t52 - p0x[4] - p1x[4];
		const Real _t60 = -p0x[3] - p1x[3];
		const Real _t61 = _t31 + _t60;
		const Real _t62 = p1x[0] - p1x[2];
		const Real _t63 = p1y[0] - 2*p1y[4];
		const Real _t64 = _t19 + p1y[5];
		const Real _t65 = p1y[0] - p1y[2];
		const Real _t66 = p1x[0] - 2*p1x[4];
		const Real _t67 = _t27 + p1x[5];
		const Real _t68 = 4*p0y[4];
		const Real _t69 = p0y[0] - p0y[5];
		const Real _t70 = 4*p0x[4];
		const Real _t71 = p0x[0] - p0x[5];
		const Real _t72 = -_t21 - _t22 + _t54;
		const Real _t73 = 2*p0y[4];
		const Real _t74 = 2*p1y[4];
		const Real _t75 = _t73 + _t74;
		const Real _t76 = -_t29 - _t30 + _t52;
		const Real _t77 = 2*p0x[4];
		const Real _t78 = 2*p1x[4];
		const Real _t79 = _t77 + _t78;
		const Real _t80 = 4*p1y[4];
		const Real _t81 = p1y[0] - p1y[5];
		const Real _t82 = 4*p1x[4];
		const Real _t83 = p1x[0] - p1x[5];
		const Real _t84 = _t33 + p0y[2];
		const Real _t85 = _t8 + p0x[2];
		const Real _t86 = -p0y[1] - p1y[1];
		const Real _t87 = _t39 + _t86;
		const Real _t88 = -p0x[1] - p1x[1];
		const Real _t89 = _t14 + _t88;
		const Real _t90 = _t35 + p1y[2];
		const Real _t91 = _t10 + p1x[2];
		const Real _t92 = _t77 + p0x[0];
		const Real _t93 = _t73 + p0y[0];
		const Real _t94 = _t52 + p0x[4] + p1x[4];
		const Real _t95 = _t54 + p0y[4] + p1y[4];
		const Real _t96 = _t78 + p1x[0];
		const Real _t97 = _t74 + p1y[0];
		#pragma endregion
		#pragma region Expressions
		return {
			-(-_t0 - _t1 - p0x[2])*(-_t2 - _t3 - p0y[5]) + (-_t0 - _t4 - p0x[5])*(-_t2 - _t5 - p0y[2]),
			-(-_t11 - _t14 - _t6)*(-_t15 - _t20 - _t23) + (-_t15 - _t36 - _t39)*(-_t28 - _t31 - _t6),
			-(-_t40 - _t41 - p1x[2])*(-_t42 - _t43 - p1y[5]) + (-_t40 - _t44 - p1x[5])*(-_t42 - _t45 - p1y[2]),
			_t46*(-_t32 - _t47 - _t48) - _t49*(-_t50 - _t51 - _t7),
			_t53*(-_t55 - _t57 - p0y[1] - p1y[1]) - _t58*(-_t59 - _t61 - p0x[1] - p1x[1]),
			_t62*(-_t34 - _t63 - _t64) - _t65*(-_t66 - _t67 - _t9),
			(_t1 + _t70 + _t71)*(_t5 + p0y[0] + 3*p0y[2]) - (_t1 + p0x[0] + 3*p0x[2])*(_t5 + _t68 + _t69),
			(_t11 + _t76 + _t79)*(_t36 + _t54 + (1.5)*p0y[2] + (1.5)*p1y[2]) - (_t36 + _t72 + _t75)*(_t11 + _t52 + (1.5)*p0x[2] + (1.5)*p1x[2]),
			(_t41 + _t82 + _t83)*(_t45 + p1y[0] + 3*p1y[2]) - (_t41 + p1x[0] + 3*p1x[2])*(_t45 + _t80 + _t81),
			_t69*(-_t24 - _t50 - _t85) - _t71*(-_t16 - _t47 - _t84),
			_t72*(-_t59 - _t89 - p0x[3] - p1x[3]) - _t76*(-_t55 - _t87 - p0y[3] - p1y[3]),
			_t81*(-_t26 - _t66 - _t91) - _t83*(-_t18 - _t63 - _t90),
			(_t17 + _t84 + _t93)*(_t51 + _t8 + _t92) - (_t25 + _t85 + _t92)*(_t33 + _t48 + _t93),
			(_t56 + _t87 + _t95)*(_t61 + _t88 + _t94) - (_t57 + _t86 + _t95)*(_t60 + _t89 + _t94),
			(_t10 + _t67 + _t96)*(_t19 + _t90 + _t97) - (_t27 + _t91 + _t96)*(_t35 + _t64 + _t97),
			(_t3 + _t49 + _t68)*(_t4 + p0x[0] + 3*p0x[5]) - (_t3 + p0y[0] + 3*p0y[5])*(_t4 + _t46 + _t70),
			(_t20 + _t58 + _t75)*(_t28 + _t52 + (1.5)*p0x[5] + (1.5)*p1x[5]) - (_t28 + _t53 + _t79)*(_t20 + _t54 + (1.5)*p0y[5] + (1.5)*p1y[5]),
			(_t43 + _t65 + _t80)*(_t44 + p1x[0] + 3*p1x[5]) - (_t43 + p1y[0] + 3*p1y[5])*(_t44 + _t62 + _t82),
		};
		#pragma endregion
	}
}
