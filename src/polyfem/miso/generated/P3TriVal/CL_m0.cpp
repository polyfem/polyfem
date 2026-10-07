#include "../P3TriVal.hpp"
namespace miso {
	RealVector<15> P3TriVal::CL_m0(const RealVector<10> &px, const RealVector<10> &py) {
		#pragma region Temporaries
		const Real _t0 = (5.5)*px[0];
		const Real _t1 = -px[3];
		const Real _t2 = (5.5)*py[0];
		const Real _t3 = -py[9];
		const Real _t4 = -px[9];
		const Real _t5 = -py[3];
		const Real _t6 = (1.84375)*px[0];
		const Real _t7 = (0.28125)*px[1];
		const Real _t8 = (0.40625)*px[3];
		const Real _t9 = (1.84375)*py[0];
		const Real _t10 = (3.375)*py[7];
		const Real _t11 = (1.125)*py[8];
		const Real _t12 = -5.0625*py[5];
		const Real _t13 = _t12 + _t3;
		const Real _t14 = (0.28125)*py[2];
		const Real _t15 = -_t14;
		const Real _t16 = (0.28125)*py[6];
		const Real _t17 = _t15 + _t16;
		const Real _t18 = (0.28125)*py[1];
		const Real _t19 = (0.40625)*py[3];
		const Real _t20 = (3.375)*px[7];
		const Real _t21 = (1.125)*px[8];
		const Real _t22 = -5.0625*px[5];
		const Real _t23 = _t22 + _t4;
		const Real _t24 = (0.28125)*px[2];
		const Real _t25 = -_t24;
		const Real _t26 = (0.28125)*px[6];
		const Real _t27 = _t25 + _t26;
		const Real _t28 = (0.125)*px[0];
		const Real _t29 = (3.375)*px[1];
		const Real _t30 = (3.375)*px[2];
		const Real _t31 = (0.125)*px[3];
		const Real _t32 = -4.5*py[1];
		const Real _t33 = -2.25*py[8];
		const Real _t34 = (1.125)*py[4];
		const Real _t35 = (1.125)*py[2];
		const Real _t36 = -_t35;
		const Real _t37 = (1.125)*py[6];
		const Real _t38 = _t36 + _t37;
		const Real _t39 = (0.125)*py[0];
		const Real _t40 = (6.75)*py[5];
		const Real _t41 = _t39 + _t40;
		const Real _t42 = (3.375)*py[1];
		const Real _t43 = (3.375)*py[2];
		const Real _t44 = (0.125)*py[3];
		const Real _t45 = -4.5*px[1];
		const Real _t46 = -2.25*px[8];
		const Real _t47 = (1.125)*px[4];
		const Real _t48 = (1.125)*px[2];
		const Real _t49 = -_t48;
		const Real _t50 = (1.125)*px[6];
		const Real _t51 = _t49 + _t50;
		const Real _t52 = (6.75)*px[5];
		const Real _t53 = _t28 + _t52;
		const Real _t54 = (0.40625)*px[0];
		const Real _t55 = (1.84375)*px[3];
		const Real _t56 = -0.40625*py[0];
		const Real _t57 = (1.125)*py[7];
		const Real _t58 = (0.28125)*py[4];
		const Real _t59 = (4.21875)*py[2];
		const Real _t60 = (0.40625)*py[0];
		const Real _t61 = (1.84375)*py[3];
		const Real _t62 = -0.40625*px[0];
		const Real _t63 = (1.125)*px[7];
		const Real _t64 = (0.28125)*px[4];
		const Real _t65 = (4.21875)*px[2];
		const Real _t66 = _t45 + px[0] + 9*px[2];
		const Real _t67 = (4.5)*py[8];
		const Real _t68 = _t32 + py[0] + 9*py[2];
		const Real _t69 = (4.5)*px[8];
		const Real _t70 = (0.40625)*px[9];
		const Real _t71 = (0.28125)*py[7];
		const Real _t72 = -_t71;
		const Real _t73 = (0.28125)*py[8];
		const Real _t74 = _t72 + _t73;
		const Real _t75 = _t12 + _t5;
		const Real _t76 = (0.40625)*py[9];
		const Real _t77 = (0.28125)*px[7];
		const Real _t78 = -_t77;
		const Real _t79 = (0.28125)*px[8];
		const Real _t80 = _t78 + _t79;
		const Real _t81 = _t1 + _t22;
		const Real _t82 = -_t70;
		const Real _t83 = -_t47;
		const Real _t84 = -2.25*px[1];
		const Real _t85 = (1.6875)*px[5];
		const Real _t86 = _t28 + _t85;
		const Real _t87 = _t24 - _t26;
		const Real _t88 = (1.125)*py[1];
		const Real _t89 = -_t88;
		const Real _t90 = -_t19;
		const Real _t91 = -2.25*py[4];
		const Real _t92 = (1.6875)*py[5];
		const Real _t93 = _t39 + _t92;
		const Real _t94 = _t71 - _t73;
		const Real _t95 = (1.125)*px[1];
		const Real _t96 = -_t95;
		const Real _t97 = -_t8;
		const Real _t98 = -2.25*px[4];
		const Real _t99 = _t77 - _t79;
		const Real _t100 = -_t76;
		const Real _t101 = -_t34;
		const Real _t102 = -2.25*py[1];
		const Real _t103 = _t14 - _t16;
		const Real _t104 = _t21 + _t54;
		const Real _t105 = -2.25*py[6];
		const Real _t106 = _t56 + _t92;
		const Real _t107 = _t11 + _t60;
		const Real _t108 = -2.25*px[6];
		const Real _t109 = _t62 + _t85;
		const Real _t110 = _t22 - px[0];
		const Real _t111 = _t110 + _t29 + _t47 - _t65;
		const Real _t112 = _t12 - py[0];
		const Real _t113 = _t112 + _t34 + _t42 - _t59;
		const Real _t114 = (3.375)*px[4];
		const Real _t115 = (0.125)*px[9];
		const Real _t116 = -4.5*py[4];
		const Real _t117 = -_t57;
		const Real _t118 = (3.375)*py[4];
		const Real _t119 = (0.125)*py[9];
		const Real _t120 = -4.5*px[4];
		const Real _t121 = -_t63;
		const Real _t122 = (4.5)*px[6];
		const Real _t123 = _t48 + _t52 + _t63 + _t84 + _t98 + px[0];
		const Real _t124 = _t102 + _t35 + _t40 + _t57 + _t91 + py[0];
		const Real _t125 = (4.5)*py[6];
		const Real _t126 = (1.84375)*px[9];
		const Real _t127 = (4.21875)*py[7];
		const Real _t128 = (1.84375)*py[9];
		const Real _t129 = (4.21875)*px[7];
		const Real _t130 = _t110 + _t114 - _t129 + _t95;
		const Real _t131 = _t112 + _t118 - _t127 + _t88;
		const Real _t132 = _t120 + px[0] + 9*px[7];
		const Real _t133 = _t116 + py[0] + 9*py[7];
		#pragma endregion
		#pragma region Expressions
		return {
			-(-_t0 - _t1 + 9*px[1] - 4.5*px[2])*(-_t2 - _t3 + 9*py[4] - 4.5*py[7]) + (-_t0 - _t4 + 9*px[4] - 4.5*px[7])*(-_t2 - _t5 + 9*py[1] - 4.5*py[2]),
			-(-_t6 + _t7 - _t8 + (1.96875)*px[2])*(-_t10 - _t11 - _t13 - _t17 - _t9 - 3.9375*py[1] + (4.21875)*py[4]) + (_t18 - _t19 - _t9 + (1.96875)*py[2])*(-_t20 - _t21 - _t23 - _t27 - _t6 - 3.9375*px[1] + (4.21875)*px[4]),
			-(_t28 - _t29 + _t30 - _t31)*(_t32 + _t33 + _t34 + _t38 + _t41 - 2.25*py[7] + py[9]) + (_t39 - _t42 + _t43 - _t44)*(_t45 + _t46 + _t47 + _t51 + _t53 - 2.25*px[7] + px[9]),
			(_t15 + _t60 + _t61 - 1.96875*py[1])*(-_t23 - _t62 - _t63 - _t64 - _t65 - 1.6875*px[1] + (4.21875)*px[6] - 3.375*px[8]) - (_t25 + _t54 + _t55 - 1.96875*px[1])*(-_t13 - _t56 - _t57 - _t58 - _t59 - 1.6875*py[1] + (4.21875)*py[6] - 3.375*py[8]),
			-(-_t66 + (5.5)*px[3])*(-_t3 - _t67 - _t68 + 9*py[6]) + (-_t68 + (5.5)*py[3])*(-_t4 - _t66 - _t69 + 9*px[6]),
			(-_t6 + _t64 - _t70 + (1.96875)*px[7])*(-_t37 - _t43 - _t74 - _t75 - _t9 + (4.21875)*py[1] - 3.9375*py[4]) - (_t58 - _t76 - _t9 + (1.96875)*py[7])*(-_t30 - _t50 - _t6 - _t80 - _t81 + (4.21875)*px[1] - 3.9375*px[4]),
			-(_t100 + _t101 + _t102 + _t103 + _t93 + (1.40625)*py[7] + (0.5625)*py[8])*(_t86 + _t96 + _t97 + _t98 + _t99 + (1.40625)*px[2] + (0.5625)*px[6]) + (_t82 + _t83 + _t84 + _t86 + _t87 + (1.40625)*px[7] + (0.5625)*px[8])*(_t89 + _t90 + _t91 + _t93 + _t94 + (1.40625)*py[2] + (0.5625)*py[6]),
			-(_t100 + _t107 + _t38 + _t89 - 0.84375*py[4] + (0.84375)*py[7])*(-_t108 - _t109 - _t31 - _t49 - _t80 - 1.40625*px[1] - 0.5625*px[4]) + (_t104 + _t51 + _t82 + _t96 - 0.84375*px[4] + (0.84375)*px[7])*(-_t105 - _t106 - _t36 - _t44 - _t74 - 1.40625*py[1] - 0.5625*py[4]),
			-(_t111 + _t55 + _t99 + (3.9375)*px[6])*(_t100 + _t113 + _t71 + (4.21875)*py[6] + (1.6875)*py[8]) + (_t113 + _t61 + _t94 + (3.9375)*py[6])*(_t111 + _t77 + _t82 + (4.21875)*px[6] + (1.6875)*px[8]),
			(-_t114 - _t115 + _t20 + _t28)*(_t105 + _t11 + _t116 + _t117 + _t41 + _t88 - 2.25*py[2] + py[3]) - (_t10 - _t118 - _t119 + _t39)*(_t108 + _t120 + _t121 + _t21 + _t53 + _t95 - 2.25*px[2] + px[3]),
			-(-_t106 - _t117 - _t119 - _t17 - _t33 - 0.5625*py[1] - 1.40625*py[4])*(_t104 + _t121 + _t50 + _t83 + _t97 - 0.84375*px[1] + (0.84375)*px[2]) + (-_t109 - _t115 - _t121 - _t27 - _t46 - 0.5625*px[1] - 1.40625*px[4])*(_t101 + _t107 + _t117 + _t37 + _t90 - 0.84375*py[1] + (0.84375)*py[2]),
			(-_t115 - _t123 + _t50 + _t69)*(_t11 - _t124 + _t125 - _t44) - (-_t119 - _t124 + _t37 + _t67)*(_t122 - _t123 + _t21 - _t31),
			(_t126 + _t54 + _t78 - 1.96875*px[4])*(-_t127 - _t18 - _t35 - _t56 - _t75 - 1.6875*py[4] - 3.375*py[6] + (4.21875)*py[8]) - (_t128 + _t60 + _t72 - 1.96875*py[4])*(-_t129 - _t48 - _t62 - _t7 - _t81 - 1.6875*px[4] - 3.375*px[6] + (4.21875)*px[8]),
			-(_t103 + _t128 + _t131 + (3.9375)*py[8])*(_t130 + _t24 + _t97 + (1.6875)*px[6] + (4.21875)*px[8]) + (_t126 + _t130 + _t87 + (3.9375)*px[8])*(_t131 + _t14 + _t90 + (1.6875)*py[6] + (4.21875)*py[8]),
			(-_t132 + (5.5)*px[9])*(-_t125 - _t133 - _t5 + 9*py[8]) - (-_t133 + (5.5)*py[9])*(-_t1 - _t122 - _t132 + 9*px[8]),
		};
		#pragma endregion
	}
}
