#include "../P2TetVal.hpp"
namespace miso {
	std::array<RealVector<20>, 8> P2TetVal::subdiv_0_3p3(const RealVector<20> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[0];
		const Real _t1 = (0.5)*_b[1];
		const Real _t2 = (0.25)*_b[0];
		const Real _t3 = (0.25)*_b[2];
		const Real _t4 = (0.125)*_b[0];
		const Real _t5 = (0.125)*_b[3];
		const Real _t6 = (0.375)*_b[1] + (0.375)*_b[2] + _t4 + _t5;
		const Real _t7 = (0.5)*_b[4];
		const Real _t8 = (0.25)*_b[1];
		const Real _t9 = _t2 + _t8;
		const Real _t10 = (0.25)*_b[4];
		const Real _t11 = (0.25)*_b[5];
		const Real _t12 = _t10 + _t11;
		const Real _t13 = (0.125)*_b[4];
		const Real _t14 = _t13 + _t4;
		const Real _t15 = (0.125)*_b[2];
		const Real _t16 = (0.125)*_b[6];
		const Real _t17 = _t15 + _t16;
		const Real _t18 = _t11 + _t17;
		const Real _t19 = _t14 + _t18 + _t8;
		const Real _t20 = (0.25)*_b[7];
		const Real _t21 = (0.125)*_b[7];
		const Real _t22 = _t21 + _t4;
		const Real _t23 = (0.125)*_b[1];
		const Real _t24 = (0.125)*_b[8];
		const Real _t25 = _t23 + _t24;
		const Real _t26 = _t12 + _t22 + _t25;
		const Real _t27 = (0.125)*_b[9];
		const Real _t28 = (0.375)*_b[4] + (0.375)*_b[7] + _t27 + _t4;
		const Real _t29 = (0.5)*_b[10];
		const Real _t30 = (0.25)*_b[10];
		const Real _t31 = (0.25)*_b[11];
		const Real _t32 = _t30 + _t31;
		const Real _t33 = (0.125)*_b[10];
		const Real _t34 = _t31 + _t33;
		const Real _t35 = (0.125)*_b[12];
		const Real _t36 = _t15 + _t35;
		const Real _t37 = _t34 + _t36;
		const Real _t38 = _t37 + _t4 + _t8;
		const Real _t39 = (0.25)*_b[13];
		const Real _t40 = _t10 + _t39;
		const Real _t41 = (0.125)*_b[11] + (0.125)*_b[13] + (0.125)*_b[14] + (0.125)*_b[5];
		const Real _t42 = _t23 + _t33 + _t41;
		const Real _t43 = _t14 + _t42;
		const Real _t44 = (0.125)*_b[15];
		const Real _t45 = _t33 + _t44;
		const Real _t46 = _t22 + _t40 + _t45;
		const Real _t47 = (0.25)*_b[16];
		const Real _t48 = (0.125)*_b[16];
		const Real _t49 = (0.125)*_b[17];
		const Real _t50 = _t23 + _t49;
		const Real _t51 = _t48 + _t50;
		const Real _t52 = _t32 + _t4 + _t51;
		const Real _t53 = (0.125)*_b[18];
		const Real _t54 = _t39 + _t53;
		const Real _t55 = _t48 + _t54;
		const Real _t56 = _t14 + _t30 + _t55;
		const Real _t57 = (0.125)*_b[19];
		const Real _t58 = (0.375)*_b[10] + (0.375)*_b[16] + _t4 + _t57;
		const Real _t59 = _t47 + _t57;
		const Real _t60 = _t34 + _t50 + _t59;
		const Real _t61 = (0.25)*_b[17];
		const Real _t62 = _t31 + _t61;
		const Real _t63 = _t48 + _t57;
		const Real _t64 = _t36 + _t62 + _t63;
		const Real _t65 = (0.375)*_b[12] + (0.375)*_b[17] + _t5 + _t57;
		const Real _t66 = _t13 + _t33 + _t54 + _t59;
		const Real _t67 = _t41 + _t49 + _t53 + _t63;
		const Real _t68 = (0.25)*_b[14];
		const Real _t69 = _t61 + _t68;
		const Real _t70 = _t16 + _t35;
		const Real _t71 = _t53 + _t70;
		const Real _t72 = _t57 + _t69 + _t71;
		const Real _t73 = (0.25)*_b[18];
		const Real _t74 = _t39 + _t73;
		const Real _t75 = _t21 + _t44;
		const Real _t76 = _t63 + _t74 + _t75;
		const Real _t77 = _t24 + _t44;
		const Real _t78 = _t49 + _t68;
		const Real _t79 = _t77 + _t78;
		const Real _t80 = _t57 + _t73 + _t79;
		const Real _t81 = (0.375)*_b[15] + (0.375)*_b[18] + _t27 + _t57;
		const Real _t82 = (0.5)*_b[16];
		const Real _t83 = (0.25)*_b[19];
		const Real _t84 = _t47 + _t83;
		const Real _t85 = (0.25)*_b[12];
		const Real _t86 = (0.5)*_b[17];
		const Real _t87 = (0.25)*_b[15];
		const Real _t88 = (0.5)*_b[18];
		const Real _t89 = (0.5)*_b[19];
		const Real _t90 = _t11 + _t13;
		const Real _t91 = _t20 + _t27;
		const Real _t92 = _t25 + _t90 + _t91;
		const Real _t93 = (0.25)*_b[8];
		const Real _t94 = _t11 + _t93;
		const Real _t95 = _t21 + _t27;
		const Real _t96 = _t17 + _t94 + _t95;
		const Real _t97 = (0.375)*_b[6] + (0.375)*_b[8] + _t27 + _t5;
		const Real _t98 = (0.5)*_b[7];
		const Real _t99 = (0.25)*_b[9];
		const Real _t100 = _t20 + _t99;
		const Real _t101 = (0.25)*_b[6];
		const Real _t102 = (0.5)*_b[8];
		const Real _t103 = (0.5)*_b[9];
		const Real _t104 = _t39 + _t45;
		const Real _t105 = _t104 + _t13 + _t91;
		const Real _t106 = _t24 + _t27;
		const Real _t107 = _t106 + _t41 + _t75;
		const Real _t108 = _t68 + _t93;
		const Real _t109 = _t108 + _t27 + _t44 + _t70;
		const Real _t110 = _t39 + _t87;
		const Real _t111 = (0.5)*_b[15];
		const Real _t112 = _t48 + _t53;
		const Real _t113 = _t110 + _t112 + _t95;
		const Real _t114 = _t53 + _t78;
		const Real _t115 = _t106 + _t114 + _t87;
		const Real _t116 = (0.5)*_b[2];
		const Real _t117 = (0.25)*_b[3];
		const Real _t118 = (0.5)*_b[3];
		const Real _t119 = _t16 + _t5;
		const Real _t120 = _t23 + _t3;
		const Real _t121 = _t119 + _t120 + _t90;
		const Real _t122 = _t117 + _t3;
		const Real _t123 = _t101 + _t11;
		const Real _t124 = (0.5)*_b[6];
		const Real _t125 = _t15 + _t5;
		const Real _t126 = _t123 + _t125 + _t21 + _t24;
		const Real _t127 = _t35 + _t5;
		const Real _t128 = _t120 + _t127 + _t34;
		const Real _t129 = _t31 + _t85;
		const Real _t130 = (0.5)*_b[12];
		const Real _t131 = _t17 + _t41;
		const Real _t132 = _t127 + _t131;
		const Real _t133 = _t101 + _t68;
		const Real _t134 = _t127 + _t133 + _t77;
		const Real _t135 = _t125 + _t129 + _t48 + _t49;
		const Real _t136 = _t114 + _t119 + _t85;
		const Real _t137 = _t112 + _t42;
		const Real _t138 = _t112 + _t17 + _t31 + _t68;
		const Real _t139 = _t17 + _t42;
		const Real _t140 = _t104 + _t11 + _t25;
		const Real _t141 = _t13 + _t21;
		const Real _t142 = _t112 + _t41 + _t77;
		const Real _t143 = _t131 + _t77;
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				_t0 + _t1,
				_t1 + _t2 + _t3,
				_t6,
				_t0 + _t7,
				_t12 + _t9,
				_t19,
				_t2 + _t20 + _t7,
				_t26,
				_t28,
				_t0 + _t29,
				_t32 + _t9,
				_t38,
				_t2 + _t30 + _t40,
				_t43,
				_t46,
				_t2 + _t29 + _t47,
				_t52,
				_t56,
				_t58,
			},
			{
				_t58,
				_t60,
				_t64,
				_t65,
				_t66,
				_t67,
				_t72,
				_t76,
				_t80,
				_t81,
				_t30 + _t82 + _t83,
				_t62 + _t84,
				_t83 + _t85 + _t86,
				_t74 + _t84,
				_t69 + _t73 + _t83,
				_t83 + _t87 + _t88,
				_t82 + _t89,
				_t86 + _t89,
				_t88 + _t89,
				_b[19],
			},
			{
				_t28,
				_t92,
				_t96,
				_t97,
				_t10 + _t98 + _t99,
				_t100 + _t94,
				_t101 + _t102 + _t99,
				_t103 + _t98,
				_t102 + _t103,
				_b[9],
				_t105,
				_t107,
				_t109,
				_t100 + _t110,
				_t108 + _t87 + _t99,
				_t103 + _t111,
				_t113,
				_t115,
				_t111 + _t73 + _t99,
				_t81,
			},
			{
				_t6,
				_t116 + _t117 + _t8,
				_t116 + _t118,
				_b[3],
				_t121,
				_t122 + _t123,
				_t118 + _t124,
				_t126,
				_t117 + _t124 + _t93,
				_t97,
				_t128,
				_t122 + _t129,
				_t118 + _t130,
				_t132,
				_t117 + _t133 + _t85,
				_t134,
				_t135,
				_t117 + _t130 + _t61,
				_t136,
				_t65,
			},
			{
				_t58,
				_t60,
				_t64,
				_t65,
				_t52,
				_t37 + _t51,
				_t135,
				_t38,
				_t128,
				_t6,
				_t56,
				_t137,
				_t138,
				_t43,
				_t139,
				_t19,
				_t46,
				_t140,
				_t26,
				_t28,
			},
			{
				_t58,
				_t60,
				_t64,
				_t65,
				_t66,
				_t67,
				_t72,
				_t76,
				_t80,
				_t81,
				_t56,
				_t137,
				_t138,
				_t141 + _t45 + _t55,
				_t142,
				_t113,
				_t46,
				_t140,
				_t105,
				_t28,
			},
			{
				_t28,
				_t92,
				_t96,
				_t97,
				_t140,
				_t143,
				_t134,
				_t138,
				_t136,
				_t65,
				_t26,
				_t141 + _t18 + _t25,
				_t126,
				_t139,
				_t132,
				_t135,
				_t19,
				_t121,
				_t128,
				_t6,
			},
			{
				_t28,
				_t92,
				_t96,
				_t97,
				_t140,
				_t143,
				_t134,
				_t138,
				_t136,
				_t65,
				_t105,
				_t107,
				_t109,
				_t142,
				_t71 + _t79,
				_t72,
				_t113,
				_t115,
				_t80,
				_t81,
			},
		}};
		#pragma endregion
	}
}
