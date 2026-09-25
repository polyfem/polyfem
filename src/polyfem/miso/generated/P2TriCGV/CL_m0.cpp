#include "../P2TriCGV.hpp"
namespace miso {
	RealVector<18> P2TriCGV::CL_m0(const RealVector<6> &rx, const RealVector<6> &ry, const RealVector<6> &u0x, const RealVector<6> &u0y, const RealVector<6> &u1x, const RealVector<6> &u1y) {
		#pragma region Temporaries
		const Real _t0 = -4*u0x[1];
		const Real _t1 = 3*rx[0];
		const Real _t2 = _t1 + 3*u0x[0];
		const Real _t3 = -4*rx[1];
		const Real _t4 = _t3 + rx[2];
		const Real _t5 = -4*u0y[3];
		const Real _t6 = 3*ry[0];
		const Real _t7 = _t6 + 3*u0y[0];
		const Real _t8 = -4*ry[3];
		const Real _t9 = _t8 + ry[5];
		const Real _t10 = -4*rx[3];
		const Real _t11 = _t10 - 4*u0x[3];
		const Real _t12 = rx[5] + u0x[5];
		const Real _t13 = -4*ry[1];
		const Real _t14 = _t13 - 4*u0y[1];
		const Real _t15 = ry[2] + u0y[2];
		const Real _t16 = _t1 + (1.5)*u0x[0] + (1.5)*u1x[0];
		const Real _t17 = 2*u0x[1];
		const Real _t18 = -_t17;
		const Real _t19 = 2*u1x[1];
		const Real _t20 = -_t19;
		const Real _t21 = _t18 + _t20;
		const Real _t22 = (0.5)*u0x[2];
		const Real _t23 = (0.5)*u1x[2];
		const Real _t24 = _t22 + _t23;
		const Real _t25 = _t6 + (1.5)*u0y[0] + (1.5)*u1y[0];
		const Real _t26 = 2*u0y[3];
		const Real _t27 = -_t26;
		const Real _t28 = 2*u1y[3];
		const Real _t29 = -_t28;
		const Real _t30 = _t27 + _t29;
		const Real _t31 = (0.5)*u0y[5];
		const Real _t32 = (0.5)*u1y[5];
		const Real _t33 = _t31 + _t32;
		const Real _t34 = 2*u0x[3];
		const Real _t35 = -_t34;
		const Real _t36 = 2*u1x[3];
		const Real _t37 = -_t36;
		const Real _t38 = _t10 + _t35 + _t37;
		const Real _t39 = (0.5)*u0x[5];
		const Real _t40 = (0.5)*u1x[5];
		const Real _t41 = _t39 + _t40 + rx[5];
		const Real _t42 = 2*u0y[1];
		const Real _t43 = -_t42;
		const Real _t44 = 2*u1y[1];
		const Real _t45 = -_t44;
		const Real _t46 = _t13 + _t43 + _t45;
		const Real _t47 = (0.5)*u0y[2];
		const Real _t48 = (0.5)*u1y[2];
		const Real _t49 = _t47 + _t48 + ry[2];
		const Real _t50 = -4*u1x[1];
		const Real _t51 = _t1 + 3*u1x[0];
		const Real _t52 = -4*u1y[3];
		const Real _t53 = _t6 + 3*u1y[0];
		const Real _t54 = _t10 - 4*u1x[3];
		const Real _t55 = rx[5] + u1x[5];
		const Real _t56 = _t13 - 4*u1y[1];
		const Real _t57 = ry[2] + u1y[2];
		const Real _t58 = rx[0] - rx[2];
		const Real _t59 = _t58 + u0x[0] - u0x[2];
		const Real _t60 = -2*ry[4];
		const Real _t61 = ry[0] + u0y[0];
		const Real _t62 = _t60 + _t61 - 2*u0y[4];
		const Real _t63 = 2*ry[1];
		const Real _t64 = 2*ry[3];
		const Real _t65 = -_t64;
		const Real _t66 = _t65 + ry[5];
		const Real _t67 = _t63 + _t66;
		const Real _t68 = _t27 + u0y[5];
		const Real _t69 = ry[0] - ry[2];
		const Real _t70 = _t69 + u0y[0] - u0y[2];
		const Real _t71 = 2*rx[3];
		const Real _t72 = -_t71;
		const Real _t73 = -2*rx[4];
		const Real _t74 = 2*rx[1];
		const Real _t75 = _t72 + _t73 + _t74;
		const Real _t76 = rx[0] + u0x[0];
		const Real _t77 = _t76 - 2*u0x[4];
		const Real _t78 = _t12 + _t35;
		const Real _t79 = (0.5)*u0x[0] + (0.5)*u1x[0];
		const Real _t80 = -_t22 - _t23 + _t58 + _t79;
		const Real _t81 = _t60 + _t67;
		const Real _t82 = (0.5)*u0y[0] + (0.5)*u1y[0];
		const Real _t83 = _t82 + ry[0];
		const Real _t84 = _t83 - u0y[4] - u1y[4];
		const Real _t85 = -u0y[3] - u1y[3];
		const Real _t86 = _t33 + _t85;
		const Real _t87 = -_t47 - _t48 + _t69 + _t82;
		const Real _t88 = _t79 + rx[0];
		const Real _t89 = _t88 - u0x[4] - u1x[4];
		const Real _t90 = -u0x[3] - u1x[3];
		const Real _t91 = _t41 + _t90;
		const Real _t92 = _t58 + u1x[0] - u1x[2];
		const Real _t93 = -2*u1y[4];
		const Real _t94 = ry[0] + u1y[0];
		const Real _t95 = _t29 + _t94;
		const Real _t96 = _t95 + u1y[5];
		const Real _t97 = _t69 + u1y[0] - u1y[2];
		const Real _t98 = rx[0] + u1x[0];
		const Real _t99 = _t98 - 2*u1x[4];
		const Real _t100 = _t37 + _t55;
		const Real _t101 = 3*rx[2];
		const Real _t102 = _t0 + _t3;
		const Real _t103 = ry[0] - ry[5];
		const Real _t104 = _t103 + u0y[0] - u0y[5];
		const Real _t105 = 4*ry[4];
		const Real _t106 = _t105 + 4*u0y[4];
		const Real _t107 = ry[0] + 3*ry[2];
		const Real _t108 = rx[0] - rx[5];
		const Real _t109 = _t108 + u0x[0] - u0x[5];
		const Real _t110 = 4*rx[4];
		const Real _t111 = _t110 + 4*u0x[4];
		const Real _t112 = _t21 + _t3;
		const Real _t113 = _t103 - _t31 - _t32 + _t82;
		const Real _t114 = 2*u0y[4];
		const Real _t115 = 2*u1y[4];
		const Real _t116 = _t105 + _t114 + _t115;
		const Real _t117 = _t108 - _t39 - _t40 + _t79;
		const Real _t118 = 2*u0x[4];
		const Real _t119 = 2*u1x[4];
		const Real _t120 = _t110 + _t118 + _t119;
		const Real _t121 = _t3 + _t50;
		const Real _t122 = _t103 + u1y[0] - u1y[5];
		const Real _t123 = _t105 + 4*u1y[4];
		const Real _t124 = _t108 + u1x[0] - u1x[5];
		const Real _t125 = _t110 + 4*u1x[4];
		const Real _t126 = -_t63;
		const Real _t127 = _t126 + _t64;
		const Real _t128 = _t15 + _t43;
		const Real _t129 = _t71 + _t73 + rx[2];
		const Real _t130 = -_t74;
		const Real _t131 = _t130 + _t18;
		const Real _t132 = _t131 + u0x[2];
		const Real _t133 = _t127 + _t60;
		const Real _t134 = -u0y[1] - u1y[1];
		const Real _t135 = _t134 + _t49;
		const Real _t136 = _t130 - u0x[1] - u1x[1];
		const Real _t137 = _t136 + _t24;
		const Real _t138 = _t45 + _t57;
		const Real _t139 = _t130 + _t20;
		const Real _t140 = _t139 + u1x[2];
		const Real _t141 = _t72 + 2*rx[4];
		const Real _t142 = _t118 + _t141 + _t76;
		const Real _t143 = _t114 + _t61;
		const Real _t144 = _t126 + 2*ry[4];
		const Real _t145 = _t144 + _t66;
		const Real _t146 = _t144 + _t65;
		const Real _t147 = _t141 + _t88 + u0x[4] + u1x[4];
		const Real _t148 = _t83 + u0y[4] + u1y[4];
		const Real _t149 = _t119 + _t141 + _t98;
		const Real _t150 = 3*rx[5];
		const Real _t151 = _t5 + _t8;
		const Real _t152 = 3*ry[5];
		const Real _t153 = _t30 + _t8;
		const Real _t154 = _t52 + _t8;
		#pragma endregion
		#pragma region Expressions
		return {
			(-_t11 - _t12 - _t2)*(-_t14 - _t15 - _t7) - (-_t0 - _t2 - _t4 - u0x[2])*(-_t5 - _t7 - _t9 - u0y[5]),
			(-_t16 - _t38 - _t41)*(-_t25 - _t46 - _t49) - (-_t16 - _t21 - _t24 - _t4)*(-_t25 - _t30 - _t33 - _t9),
			(-_t51 - _t54 - _t55)*(-_t53 - _t56 - _t57) - (-_t4 - _t50 - _t51 - u1x[2])*(-_t52 - _t53 - _t9 - u1y[5]),
			_t59*(-_t42 - _t62 - _t67 - _t68) - _t70*(-_t17 - _t75 - _t77 - _t78),
			_t80*(-_t81 - _t84 - _t86 - u0y[1] - u1y[1]) - _t87*(-_t75 - _t89 - _t91 - u0x[1] - u1x[1]),
			_t92*(-_t44 - _t81 - _t93 - _t96) - _t97*(-_t100 - _t19 - _t75 - _t99),
			(_t102 + _t109 + _t111)*(_t107 + _t14 + u0y[0] + 3*u0y[2]) - (_t104 + _t106 + _t14)*(_t101 + _t102 + _t76 + 3*u0x[2]),
			(_t112 + _t117 + _t120)*(_t107 + _t46 + _t82 + (1.5)*u0y[2] + (1.5)*u1y[2]) - (_t113 + _t116 + _t46)*(_t101 + _t112 + _t88 + (1.5)*u0x[2] + (1.5)*u1x[2]),
			(_t121 + _t124 + _t125)*(_t107 + _t56 + u1y[0] + 3*u1y[2]) - (_t122 + _t123 + _t56)*(_t101 + _t121 + _t98 + 3*u1x[2]),
			_t104*(-_t129 - _t132 - _t34 - _t77) - _t109*(-_t127 - _t128 - _t26 - _t62),
			_t113*(-_t129 - _t137 - _t89 - u0x[3] - u1x[3]) - _t117*(-_t133 - _t135 - _t84 - u0y[3] - u1y[3]),
			_t122*(-_t129 - _t140 - _t36 - _t99) - _t124*(-_t133 - _t138 - _t28 - _t93 - _t94),
			(_t131 + _t142 + _t78)*(_t128 + _t143 + _t146 + _t27) - (_t132 + _t142 + _t35 + rx[2])*(_t143 + _t145 + _t43 + _t68),
			(_t136 + _t147 + _t91)*(_t135 + _t146 + _t148 + _t85) - (_t134 + _t145 + _t148 + _t86)*(_t137 + _t147 + _t90 + rx[2]),
			(_t100 + _t139 + _t149)*(_t115 + _t138 + _t146 + _t95) - (_t115 + _t145 + _t45 + _t96)*(_t140 + _t149 + _t37 + rx[2]),
			(_t106 + _t151 + _t70)*(_t11 + _t150 + _t76 + 3*u0x[5]) - (_t11 + _t111 + _t59)*(_t151 + _t152 + _t61 + 3*u0y[5]),
			(_t116 + _t153 + _t87)*(_t150 + _t38 + _t88 + (1.5)*u0x[5] + (1.5)*u1x[5]) - (_t120 + _t38 + _t80)*(_t152 + _t153 + _t83 + (1.5)*u0y[5] + (1.5)*u1y[5]),
			(_t123 + _t154 + _t97)*(_t150 + _t54 + _t98 + 3*u1x[5]) - (_t125 + _t54 + _t92)*(_t152 + _t154 + _t94 + 3*u1y[5]),
		};
		#pragma endregion
	}
}
