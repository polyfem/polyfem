#include "../P3TriVal.hpp"
namespace miso {
	RealVector<15> P3TriVal::CL_m0(const RealVector<10> &rx, const RealVector<10> &ry, const RealVector<10> &ux, const RealVector<10> &uy) {
		#pragma region Temporaries
		const Real _t0 = (5.5)*rx[0] + (5.5)*ux[0];
		const Real _t1 = -rx[3] - ux[3];
		const Real _t2 = (5.5)*ry[0] + (5.5)*uy[0];
		const Real _t3 = -ry[9] - uy[9];
		const Real _t4 = -rx[9] - ux[9];
		const Real _t5 = -ry[3] - uy[3];
		const Real _t6 = (0.28125)*rx[1];
		const Real _t7 = (0.28125)*ux[1];
		const Real _t8 = (0.40625)*rx[3];
		const Real _t9 = (0.40625)*ux[3];
		const Real _t10 = (1.84375)*rx[0] + (1.84375)*ux[0];
		const Real _t11 = (1.84375)*ry[0] + (1.84375)*uy[0];
		const Real _t12 = (3.375)*ry[7] + (3.375)*uy[7];
		const Real _t13 = (1.125)*ry[8];
		const Real _t14 = (1.125)*uy[8];
		const Real _t15 = _t13 + _t14;
		const Real _t16 = -5.0625*ry[5] - 5.0625*uy[5];
		const Real _t17 = _t16 + _t3;
		const Real _t18 = (0.28125)*ry[6];
		const Real _t19 = (0.28125)*uy[6];
		const Real _t20 = (0.28125)*ry[2];
		const Real _t21 = (0.28125)*uy[2];
		const Real _t22 = -_t20 - _t21;
		const Real _t23 = _t18 + _t19 + _t22;
		const Real _t24 = (0.28125)*ry[1];
		const Real _t25 = (0.28125)*uy[1];
		const Real _t26 = (0.40625)*ry[3];
		const Real _t27 = (0.40625)*uy[3];
		const Real _t28 = (3.375)*rx[7] + (3.375)*ux[7];
		const Real _t29 = (1.125)*rx[8];
		const Real _t30 = (1.125)*ux[8];
		const Real _t31 = _t29 + _t30;
		const Real _t32 = -5.0625*rx[5] - 5.0625*ux[5];
		const Real _t33 = _t32 + _t4;
		const Real _t34 = (0.28125)*rx[6];
		const Real _t35 = (0.28125)*ux[6];
		const Real _t36 = (0.28125)*rx[2];
		const Real _t37 = (0.28125)*ux[2];
		const Real _t38 = -_t36 - _t37;
		const Real _t39 = _t34 + _t35 + _t38;
		const Real _t40 = (3.375)*rx[1];
		const Real _t41 = (3.375)*ux[1];
		const Real _t42 = (0.125)*rx[3];
		const Real _t43 = (0.125)*ux[3];
		const Real _t44 = (0.125)*rx[0] + (0.125)*ux[0];
		const Real _t45 = (3.375)*rx[2] + (3.375)*ux[2];
		const Real _t46 = -4.5*ry[1] - 4.5*uy[1];
		const Real _t47 = (1.125)*ry[6];
		const Real _t48 = (1.125)*uy[6];
		const Real _t49 = _t47 + _t48;
		const Real _t50 = (1.125)*ry[2];
		const Real _t51 = (1.125)*uy[2];
		const Real _t52 = -_t50 - _t51;
		const Real _t53 = _t49 + _t52;
		const Real _t54 = (1.125)*ry[4];
		const Real _t55 = (1.125)*uy[4];
		const Real _t56 = _t54 + _t55;
		const Real _t57 = -2.25*ry[8] - 2.25*uy[8];
		const Real _t58 = (0.125)*ry[0] + (0.125)*uy[0];
		const Real _t59 = (6.75)*ry[5] + (6.75)*uy[5];
		const Real _t60 = _t58 + _t59;
		const Real _t61 = (3.375)*ry[1];
		const Real _t62 = (3.375)*uy[1];
		const Real _t63 = (0.125)*ry[3];
		const Real _t64 = (0.125)*uy[3];
		const Real _t65 = (3.375)*ry[2] + (3.375)*uy[2];
		const Real _t66 = -4.5*rx[1] - 4.5*ux[1];
		const Real _t67 = (1.125)*rx[6];
		const Real _t68 = (1.125)*ux[6];
		const Real _t69 = _t67 + _t68;
		const Real _t70 = (1.125)*rx[2];
		const Real _t71 = (1.125)*ux[2];
		const Real _t72 = -_t70 - _t71;
		const Real _t73 = _t69 + _t72;
		const Real _t74 = (1.125)*rx[4];
		const Real _t75 = (1.125)*ux[4];
		const Real _t76 = _t74 + _t75;
		const Real _t77 = -2.25*rx[8] - 2.25*ux[8];
		const Real _t78 = (6.75)*rx[5] + (6.75)*ux[5];
		const Real _t79 = _t44 + _t78;
		const Real _t80 = (0.40625)*rx[0] + (0.40625)*ux[0];
		const Real _t81 = (1.84375)*rx[3] + (1.84375)*ux[3];
		const Real _t82 = (0.28125)*ry[4];
		const Real _t83 = (0.28125)*uy[4];
		const Real _t84 = (4.21875)*ry[2];
		const Real _t85 = (4.21875)*uy[2];
		const Real _t86 = -0.40625*ry[0] - 0.40625*uy[0];
		const Real _t87 = (1.125)*ry[7];
		const Real _t88 = (1.125)*uy[7];
		const Real _t89 = _t87 + _t88;
		const Real _t90 = (0.40625)*ry[0] + (0.40625)*uy[0];
		const Real _t91 = (1.84375)*ry[3] + (1.84375)*uy[3];
		const Real _t92 = (0.28125)*rx[4];
		const Real _t93 = (0.28125)*ux[4];
		const Real _t94 = (4.21875)*rx[2];
		const Real _t95 = (4.21875)*ux[2];
		const Real _t96 = -0.40625*rx[0] - 0.40625*ux[0];
		const Real _t97 = (1.125)*rx[7];
		const Real _t98 = (1.125)*ux[7];
		const Real _t99 = _t97 + _t98;
		const Real _t100 = rx[0] + ux[0];
		const Real _t101 = _t100 + _t66 + 9*rx[2] + 9*ux[2];
		const Real _t102 = (4.5)*ry[8];
		const Real _t103 = (4.5)*uy[8];
		const Real _t104 = ry[0] + uy[0];
		const Real _t105 = _t104 + _t46 + 9*ry[2] + 9*uy[2];
		const Real _t106 = (4.5)*rx[8];
		const Real _t107 = (4.5)*ux[8];
		const Real _t108 = (0.40625)*rx[9];
		const Real _t109 = (0.40625)*ux[9];
		const Real _t110 = _t16 + _t5;
		const Real _t111 = (0.28125)*ry[8];
		const Real _t112 = (0.28125)*uy[8];
		const Real _t113 = (0.28125)*ry[7];
		const Real _t114 = (0.28125)*uy[7];
		const Real _t115 = -_t113 - _t114;
		const Real _t116 = _t111 + _t112 + _t115;
		const Real _t117 = (0.40625)*ry[9];
		const Real _t118 = (0.40625)*uy[9];
		const Real _t119 = _t1 + _t32;
		const Real _t120 = (0.28125)*rx[8];
		const Real _t121 = (0.28125)*ux[8];
		const Real _t122 = (0.28125)*rx[7];
		const Real _t123 = (0.28125)*ux[7];
		const Real _t124 = -_t122 - _t123;
		const Real _t125 = _t120 + _t121 + _t124;
		const Real _t126 = -_t108 - _t109;
		const Real _t127 = -_t74 - _t75;
		const Real _t128 = -2.25*rx[1] - 2.25*ux[1];
		const Real _t129 = (1.6875)*rx[5] + (1.6875)*ux[5];
		const Real _t130 = _t129 + _t44;
		const Real _t131 = _t36 + _t37;
		const Real _t132 = _t131 - _t34 - _t35;
		const Real _t133 = (1.125)*ry[1];
		const Real _t134 = (1.125)*uy[1];
		const Real _t135 = -_t133 - _t134;
		const Real _t136 = -_t26 - _t27;
		const Real _t137 = -2.25*ry[4] - 2.25*uy[4];
		const Real _t138 = (1.6875)*ry[5] + (1.6875)*uy[5];
		const Real _t139 = _t138 + _t58;
		const Real _t140 = _t113 + _t114;
		const Real _t141 = -_t111 - _t112 + _t140;
		const Real _t142 = (1.125)*rx[1];
		const Real _t143 = (1.125)*ux[1];
		const Real _t144 = -_t142 - _t143;
		const Real _t145 = -_t8 - _t9;
		const Real _t146 = -2.25*rx[4] - 2.25*ux[4];
		const Real _t147 = _t122 + _t123;
		const Real _t148 = -_t120 - _t121 + _t147;
		const Real _t149 = -_t117 - _t118;
		const Real _t150 = -_t54 - _t55;
		const Real _t151 = -2.25*ry[1] - 2.25*uy[1];
		const Real _t152 = _t20 + _t21;
		const Real _t153 = _t152 - _t18 - _t19;
		const Real _t154 = _t31 + _t80;
		const Real _t155 = -2.25*ry[6] - 2.25*uy[6];
		const Real _t156 = _t138 + _t86;
		const Real _t157 = _t63 + _t64;
		const Real _t158 = _t15 + _t90;
		const Real _t159 = -2.25*rx[6] - 2.25*ux[6];
		const Real _t160 = _t129 + _t96;
		const Real _t161 = _t42 + _t43;
		const Real _t162 = _t32 - rx[0] - ux[0];
		const Real _t163 = _t162 + _t40 + _t41 + _t76 - _t94 - _t95;
		const Real _t164 = _t16 - ry[0] - uy[0];
		const Real _t165 = _t164 + _t56 + _t61 + _t62 - _t84 - _t85;
		const Real _t166 = (3.375)*rx[4];
		const Real _t167 = (3.375)*ux[4];
		const Real _t168 = (0.125)*rx[9];
		const Real _t169 = (0.125)*ux[9];
		const Real _t170 = -4.5*ry[4] - 4.5*uy[4];
		const Real _t171 = -_t87 - _t88;
		const Real _t172 = _t133 + _t134;
		const Real _t173 = (3.375)*ry[4];
		const Real _t174 = (3.375)*uy[4];
		const Real _t175 = (0.125)*ry[9];
		const Real _t176 = (0.125)*uy[9];
		const Real _t177 = -4.5*rx[4] - 4.5*ux[4];
		const Real _t178 = -_t97 - _t98;
		const Real _t179 = _t142 + _t143;
		const Real _t180 = _t175 + _t176;
		const Real _t181 = _t168 + _t169;
		const Real _t182 = (4.5)*rx[6];
		const Real _t183 = (4.5)*ux[6];
		const Real _t184 = _t70 + _t71;
		const Real _t185 = _t100 + _t128 + _t146 + _t184 + _t78 + _t99;
		const Real _t186 = _t50 + _t51;
		const Real _t187 = _t104 + _t137 + _t151 + _t186 + _t59 + _t89;
		const Real _t188 = (4.5)*ry[6];
		const Real _t189 = (4.5)*uy[6];
		const Real _t190 = (1.84375)*rx[9] + (1.84375)*ux[9];
		const Real _t191 = (4.21875)*ry[7];
		const Real _t192 = (4.21875)*uy[7];
		const Real _t193 = (1.84375)*ry[9] + (1.84375)*uy[9];
		const Real _t194 = (4.21875)*rx[7];
		const Real _t195 = (4.21875)*ux[7];
		const Real _t196 = _t162 + _t166 + _t167 + _t179 - _t194 - _t195;
		const Real _t197 = _t164 + _t172 + _t173 + _t174 - _t191 - _t192;
		const Real _t198 = _t100 + _t177 + 9*rx[7] + 9*ux[7];
		const Real _t199 = _t104 + _t170 + 9*ry[7] + 9*uy[7];
		#pragma endregion
		#pragma region Expressions
		return {
			-(-_t0 - _t1 + 9*rx[1] - 4.5*rx[2] + 9*ux[1] - 4.5*ux[2])*(-_t2 - _t3 + 9*ry[4] - 4.5*ry[7] + 9*uy[4] - 4.5*uy[7]) + (-_t0 - _t4 + 9*rx[4] - 4.5*rx[7] + 9*ux[4] - 4.5*ux[7])*(-_t2 - _t5 + 9*ry[1] - 4.5*ry[2] + 9*uy[1] - 4.5*uy[2]),
			-(-_t10 + _t6 + _t7 - _t8 - _t9 + (1.96875)*rx[2] + (1.96875)*ux[2])*(-_t11 - _t12 - _t15 - _t17 - _t23 - 3.9375*ry[1] + (4.21875)*ry[4] - 3.9375*uy[1] + (4.21875)*uy[4]) + (-_t11 + _t24 + _t25 - _t26 - _t27 + (1.96875)*ry[2] + (1.96875)*uy[2])*(-_t10 - _t28 - _t31 - _t33 - _t39 - 3.9375*rx[1] + (4.21875)*rx[4] - 3.9375*ux[1] + (4.21875)*ux[4]),
			-(-_t40 - _t41 - _t42 - _t43 + _t44 + _t45)*(_t46 + _t53 + _t56 + _t57 + _t60 - 2.25*ry[7] + ry[9] - 2.25*uy[7] + uy[9]) + (_t58 - _t61 - _t62 - _t63 - _t64 + _t65)*(_t66 + _t73 + _t76 + _t77 + _t79 - 2.25*rx[7] + rx[9] - 2.25*ux[7] + ux[9]),
			(_t22 + _t90 + _t91 - 1.96875*ry[1] - 1.96875*uy[1])*(-_t33 - _t92 - _t93 - _t94 - _t95 - _t96 - _t99 - 1.6875*rx[1] + (4.21875)*rx[6] - 3.375*rx[8] - 1.6875*ux[1] + (4.21875)*ux[6] - 3.375*ux[8]) - (_t38 + _t80 + _t81 - 1.96875*rx[1] - 1.96875*ux[1])*(-_t17 - _t82 - _t83 - _t84 - _t85 - _t86 - _t89 - 1.6875*ry[1] + (4.21875)*ry[6] - 3.375*ry[8] - 1.6875*uy[1] + (4.21875)*uy[6] - 3.375*uy[8]),
			-(-_t101 + (5.5)*rx[3] + (5.5)*ux[3])*(-_t102 - _t103 - _t105 - _t3 + 9*ry[6] + 9*uy[6]) + (-_t105 + (5.5)*ry[3] + (5.5)*uy[3])*(-_t101 - _t106 - _t107 - _t4 + 9*rx[6] + 9*ux[6]),
			(-_t10 - _t108 - _t109 + _t92 + _t93 + (1.96875)*rx[7] + (1.96875)*ux[7])*(-_t11 - _t110 - _t116 - _t49 - _t65 + (4.21875)*ry[1] - 3.9375*ry[4] + (4.21875)*uy[1] - 3.9375*uy[4]) - (-_t11 - _t117 - _t118 + _t82 + _t83 + (1.96875)*ry[7] + (1.96875)*uy[7])*(-_t10 - _t119 - _t125 - _t45 - _t69 + (4.21875)*rx[1] - 3.9375*rx[4] + (4.21875)*ux[1] - 3.9375*ux[4]),
			(_t126 + _t127 + _t128 + _t130 + _t132 + (1.40625)*rx[7] + (0.5625)*rx[8] + (1.40625)*ux[7] + (0.5625)*ux[8])*(_t135 + _t136 + _t137 + _t139 + _t141 + (1.40625)*ry[2] + (0.5625)*ry[6] + (1.40625)*uy[2] + (0.5625)*uy[6]) - (_t130 + _t144 + _t145 + _t146 + _t148 + (1.40625)*rx[2] + (0.5625)*rx[6] + (1.40625)*ux[2] + (0.5625)*ux[6])*(_t139 + _t149 + _t150 + _t151 + _t153 + (1.40625)*ry[7] + (0.5625)*ry[8] + (1.40625)*uy[7] + (0.5625)*uy[8]),
			(_t126 + _t144 + _t154 + _t73 - 0.84375*rx[4] + (0.84375)*rx[7] - 0.84375*ux[4] + (0.84375)*ux[7])*(-_t116 - _t155 - _t156 - _t157 - _t52 - 1.40625*ry[1] - 0.5625*ry[4] - 1.40625*uy[1] - 0.5625*uy[4]) - (_t135 + _t149 + _t158 + _t53 - 0.84375*ry[4] + (0.84375)*ry[7] - 0.84375*uy[4] + (0.84375)*uy[7])*(-_t125 - _t159 - _t160 - _t161 - _t72 - 1.40625*rx[1] - 0.5625*rx[4] - 1.40625*ux[1] - 0.5625*ux[4]),
			(_t141 + _t165 + _t91 + (3.9375)*ry[6] + (3.9375)*uy[6])*(_t126 + _t147 + _t163 + (4.21875)*rx[6] + (1.6875)*rx[8] + (4.21875)*ux[6] + (1.6875)*ux[8]) - (_t148 + _t163 + _t81 + (3.9375)*rx[6] + (3.9375)*ux[6])*(_t140 + _t149 + _t165 + (4.21875)*ry[6] + (1.6875)*ry[8] + (4.21875)*uy[6] + (1.6875)*uy[8]),
			(-_t166 - _t167 - _t168 - _t169 + _t28 + _t44)*(_t15 + _t155 + _t170 + _t171 + _t172 + _t60 - 2.25*ry[2] + ry[3] - 2.25*uy[2] + uy[3]) - (_t12 - _t173 - _t174 - _t175 - _t176 + _t58)*(_t159 + _t177 + _t178 + _t179 + _t31 + _t79 - 2.25*rx[2] + rx[3] - 2.25*ux[2] + ux[3]),
			-(-_t156 - _t171 - _t180 - _t23 - _t57 - 0.5625*ry[1] - 1.40625*ry[4] - 0.5625*uy[1] - 1.40625*uy[4])*(_t127 + _t145 + _t154 + _t178 + _t69 - 0.84375*rx[1] + (0.84375)*rx[2] - 0.84375*ux[1] + (0.84375)*ux[2]) + (-_t160 - _t178 - _t181 - _t39 - _t77 - 0.5625*rx[1] - 1.40625*rx[4] - 0.5625*ux[1] - 1.40625*ux[4])*(_t136 + _t150 + _t158 + _t171 + _t49 - 0.84375*ry[1] + (0.84375)*ry[2] - 0.84375*uy[1] + (0.84375)*uy[2]),
			-(-_t161 + _t182 + _t183 - _t185 + _t29 + _t30)*(_t102 + _t103 - _t180 - _t187 + _t47 + _t48) + (_t106 + _t107 - _t181 - _t185 + _t67 + _t68)*(_t13 + _t14 - _t157 - _t187 + _t188 + _t189),
			-(_t115 + _t193 + _t90 - 1.96875*ry[4] - 1.96875*uy[4])*(-_t119 - _t184 - _t194 - _t195 - _t6 - _t7 - _t96 - 1.6875*rx[4] - 3.375*rx[6] + (4.21875)*rx[8] - 1.6875*ux[4] - 3.375*ux[6] + (4.21875)*ux[8]) + (_t124 + _t190 + _t80 - 1.96875*rx[4] - 1.96875*ux[4])*(-_t110 - _t186 - _t191 - _t192 - _t24 - _t25 - _t86 - 1.6875*ry[4] - 3.375*ry[6] + (4.21875)*ry[8] - 1.6875*uy[4] - 3.375*uy[6] + (4.21875)*uy[8]),
			(_t132 + _t190 + _t196 + (3.9375)*rx[8] + (3.9375)*ux[8])*(_t136 + _t152 + _t197 + (1.6875)*ry[6] + (4.21875)*ry[8] + (1.6875)*uy[6] + (4.21875)*uy[8]) - (_t153 + _t193 + _t197 + (3.9375)*ry[8] + (3.9375)*uy[8])*(_t131 + _t145 + _t196 + (1.6875)*rx[6] + (4.21875)*rx[8] + (1.6875)*ux[6] + (4.21875)*ux[8]),
			(-_t198 + (5.5)*rx[9] + (5.5)*ux[9])*(-_t188 - _t189 - _t199 - _t5 + 9*ry[8] + 9*uy[8]) - (-_t199 + (5.5)*ry[9] + (5.5)*uy[9])*(-_t1 - _t182 - _t183 - _t198 + 9*rx[8] + 9*ux[8]),
		};
		#pragma endregion
	}
}
