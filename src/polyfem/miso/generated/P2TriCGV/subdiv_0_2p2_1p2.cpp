#include "../P2TriCGV.hpp"
namespace miso {
	std::array<RealVector<18>, 8> P2TriCGV::subdiv_0_2p2_1p2(const RealVector<18> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[0];
		const Real _t1 = (0.5)*_b[1];
		const Real _t2 = (0.25)*_b[0];
		const Real _t3 = (0.25)*_b[2];
		const Real _t4 = _t1 + _t2 + _t3;
		const Real _t5 = (0.5)*_b[3];
		const Real _t6 = (0.25)*_b[1];
		const Real _t7 = _t2 + _t6;
		const Real _t8 = (0.25)*_b[3];
		const Real _t9 = (0.25)*_b[4];
		const Real _t10 = _t8 + _t9;
		const Real _t11 = (0.125)*_b[0];
		const Real _t12 = (0.125)*_b[3];
		const Real _t13 = _t11 + _t12;
		const Real _t14 = (0.125)*_b[2];
		const Real _t15 = (0.125)*_b[5];
		const Real _t16 = _t14 + _t15;
		const Real _t17 = _t13 + _t16 + _t6 + _t9;
		const Real _t18 = (0.25)*_b[6];
		const Real _t19 = _t18 + _t2 + _t5;
		const Real _t20 = (0.125)*_b[1];
		const Real _t21 = _t11 + _t20;
		const Real _t22 = (0.125)*_b[6];
		const Real _t23 = (0.125)*_b[7];
		const Real _t24 = _t22 + _t23;
		const Real _t25 = _t10 + _t21 + _t24;
		const Real _t26 = _t12 + _t15 + _t9;
		const Real _t27 = (0.0625)*_b[0] + (0.0625)*_b[2];
		const Real _t28 = _t20 + _t27;
		const Real _t29 = (0.0625)*_b[6] + (0.0625)*_b[8] + _t23;
		const Real _t30 = _t26 + _t28 + _t29;
		const Real _t31 = (0.5)*_b[9];
		const Real _t32 = (0.25)*_b[10];
		const Real _t33 = (0.25)*_b[9];
		const Real _t34 = _t32 + _t33;
		const Real _t35 = (0.125)*_b[11];
		const Real _t36 = (0.125)*_b[9];
		const Real _t37 = _t32 + _t35 + _t36;
		const Real _t38 = _t11 + _t14 + _t37 + _t6;
		const Real _t39 = (0.25)*_b[12];
		const Real _t40 = _t33 + _t39 + _t8;
		const Real _t41 = _t2 + _t40;
		const Real _t42 = (0.125)*_b[10] + (0.125)*_b[13] + (0.125)*_b[4];
		const Real _t43 = _t20 + _t42;
		const Real _t44 = (0.125)*_b[12];
		const Real _t45 = _t36 + _t44;
		const Real _t46 = _t13 + _t43 + _t45;
		const Real _t47 = (0.0625)*_b[11] + (0.0625)*_b[12] + (0.0625)*_b[14] + (0.0625)*_b[3] + (0.0625)*_b[5] + (0.0625)*_b[9];
		const Real _t48 = _t27 + _t43 + _t47;
		const Real _t49 = (0.25)*_b[15];
		const Real _t50 = _t2 + _t31 + _t49;
		const Real _t51 = (0.125)*_b[15];
		const Real _t52 = (0.125)*_b[16];
		const Real _t53 = _t51 + _t52;
		const Real _t54 = _t21 + _t34 + _t53;
		const Real _t55 = (0.0625)*_b[15] + (0.0625)*_b[17] + _t52;
		const Real _t56 = _t28 + _t37 + _t55;
		const Real _t57 = (0.5)*_b[2];
		const Real _t58 = _t3 + _t6;
		const Real _t59 = (0.25)*_b[5];
		const Real _t60 = _t59 + _t9;
		const Real _t61 = (0.5)*_b[5];
		const Real _t62 = _t14 + _t20;
		const Real _t63 = (0.125)*_b[8];
		const Real _t64 = _t23 + _t63;
		const Real _t65 = _t60 + _t62 + _t64;
		const Real _t66 = (0.25)*_b[8];
		const Real _t67 = _t3 + _t61 + _t66;
		const Real _t68 = (0.25)*_b[11];
		const Real _t69 = _t32 + _t68;
		const Real _t70 = (0.5)*_b[11];
		const Real _t71 = (0.125)*_b[14];
		const Real _t72 = _t35 + _t71;
		const Real _t73 = _t16 + _t43 + _t72;
		const Real _t74 = (0.25)*_b[14];
		const Real _t75 = _t59 + _t68 + _t74;
		const Real _t76 = _t3 + _t75;
		const Real _t77 = (0.125)*_b[17];
		const Real _t78 = _t52 + _t77;
		const Real _t79 = _t62 + _t69 + _t78;
		const Real _t80 = (0.25)*_b[17];
		const Real _t81 = _t3 + _t70 + _t80;
		const Real _t82 = _t40 + _t49;
		const Real _t83 = _t12 + _t42 + _t45;
		const Real _t84 = _t53 + _t83;
		const Real _t85 = _t42 + _t47;
		const Real _t86 = _t55 + _t85;
		const Real _t87 = (0.5)*_b[12];
		const Real _t88 = _t18 + _t49 + _t87;
		const Real _t89 = (0.25)*_b[13];
		const Real _t90 = _t39 + _t89;
		const Real _t91 = _t24 + _t53 + _t90;
		const Real _t92 = _t44 + _t71 + _t89;
		const Real _t93 = _t29 + _t55 + _t92;
		const Real _t94 = (0.5)*_b[15];
		const Real _t95 = (0.25)*_b[16];
		const Real _t96 = _t49 + _t95;
		const Real _t97 = _t51 + _t77 + _t95;
		const Real _t98 = _t37 + _t97;
		const Real _t99 = _t92 + _t97;
		const Real _t100 = (0.5)*_b[16];
		const Real _t101 = _t100 + _t49 + _t80;
		const Real _t102 = _t15 + _t42 + _t72;
		const Real _t103 = _t102 + _t78;
		const Real _t104 = _t75 + _t80;
		const Real _t105 = _t74 + _t89;
		const Real _t106 = _t105 + _t64 + _t78;
		const Real _t107 = (0.5)*_b[14];
		const Real _t108 = _t107 + _t66 + _t80;
		const Real _t109 = _t80 + _t95;
		const Real _t110 = (0.5)*_b[17];
		const Real _t111 = (0.5)*_b[6];
		const Real _t112 = (0.25)*_b[7];
		const Real _t113 = _t112 + _t18;
		const Real _t114 = _t112 + _t22 + _t63;
		const Real _t115 = _t114 + _t26;
		const Real _t116 = (0.5)*_b[7];
		const Real _t117 = _t116 + _t18 + _t66;
		const Real _t118 = _t18 + _t40;
		const Real _t119 = _t24 + _t83;
		const Real _t120 = _t29 + _t85;
		const Real _t121 = _t114 + _t92;
		const Real _t122 = _t112 + _t66;
		const Real _t123 = (0.5)*_b[8];
		const Real _t124 = _t102 + _t64;
		const Real _t125 = _t66 + _t75;
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				_t0 + _t1,
				_t4,
				_t0 + _t5,
				_t10 + _t7,
				_t17,
				_t19,
				_t25,
				_t30,
				_t0 + _t31,
				_t34 + _t7,
				_t38,
				_t41,
				_t46,
				_t48,
				_t50,
				_t54,
				_t56,
			},
			{
				_t4,
				_t1 + _t57,
				_b[2],
				_t17,
				_t58 + _t60,
				_t57 + _t61,
				_t30,
				_t65,
				_t67,
				_t38,
				_t58 + _t69,
				_t57 + _t70,
				_t48,
				_t73,
				_t76,
				_t56,
				_t79,
				_t81,
			},
			{
				_t50,
				_t54,
				_t56,
				_t82,
				_t84,
				_t86,
				_t88,
				_t91,
				_t93,
				_t31 + _t94,
				_t34 + _t96,
				_t98,
				_t87 + _t94,
				_t90 + _t96,
				_t99,
				_b[15],
				_t100 + _t94,
				_t101,
			},
			{
				_t56,
				_t79,
				_t81,
				_t86,
				_t103,
				_t104,
				_t93,
				_t106,
				_t108,
				_t98,
				_t109 + _t69,
				_t110 + _t70,
				_t99,
				_t105 + _t109,
				_t107 + _t110,
				_t101,
				_t100 + _t110,
				_b[17],
			},
			{
				_t19,
				_t25,
				_t30,
				_t111 + _t5,
				_t10 + _t113,
				_t115,
				_b[6],
				_t111 + _t116,
				_t117,
				_t118,
				_t119,
				_t120,
				_t111 + _t87,
				_t113 + _t90,
				_t121,
				_t88,
				_t91,
				_t93,
			},
			{
				_t30,
				_t65,
				_t67,
				_t115,
				_t122 + _t60,
				_t123 + _t61,
				_t117,
				_t116 + _t123,
				_b[8],
				_t120,
				_t124,
				_t125,
				_t121,
				_t105 + _t122,
				_t107 + _t123,
				_t93,
				_t106,
				_t108,
			},
			{
				_t88,
				_t91,
				_t93,
				_t82,
				_t84,
				_t86,
				_t50,
				_t54,
				_t56,
				_t118,
				_t119,
				_t120,
				_t41,
				_t46,
				_t48,
				_t19,
				_t25,
				_t30,
			},
			{
				_t93,
				_t106,
				_t108,
				_t86,
				_t103,
				_t104,
				_t56,
				_t79,
				_t81,
				_t120,
				_t124,
				_t125,
				_t48,
				_t73,
				_t76,
				_t30,
				_t65,
				_t67,
			},
		}};
		#pragma endregion
	}
}
