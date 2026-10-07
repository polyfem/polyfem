#include "../P4TriVal.hpp"
namespace miso {
	std::array<RealVector<28>, 4> P4TriVal::subdiv_0_2p6(const RealVector<28> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[0];
		const Real _t1 = (0.5)*_b[1];
		const Real _t2 = (0.25)*_b[0];
		const Real _t3 = (0.125)*_b[0];
		const Real _t4 = (0.375)*_b[2];
		const Real _t5 = (0.125)*_b[3];
		const Real _t6 = (0.0625)*_b[0];
		const Real _t7 = (0.25)*_b[1];
		const Real _t8 = (0.25)*_b[3];
		const Real _t9 = (0.0625)*_b[4];
		const Real _t10 = (0.03125)*_b[0];
		const Real _t11 = (0.3125)*_b[3];
		const Real _t12 = (0.15625)*_b[4];
		const Real _t13 = (0.03125)*_b[5];
		const Real _t14 = (0.015625)*_b[0];
		const Real _t15 = (0.09375)*_b[1];
		const Real _t16 = (0.09375)*_b[5];
		const Real _t17 = (0.015625)*_b[6];
		const Real _t18 = (0.234375)*_b[2] + (0.234375)*_b[4] + _t11 + _t14 + _t15 + _t16 + _t17;
		const Real _t19 = (0.5)*_b[7];
		const Real _t20 = (0.25)*_b[7];
		const Real _t21 = (0.25)*_b[8];
		const Real _t22 = _t21 + _t7;
		const Real _t23 = (0.125)*_b[7];
		const Real _t24 = (0.125)*_b[9];
		const Real _t25 = _t23 + _t24;
		const Real _t26 = (0.0625)*_b[3];
		const Real _t27 = (0.1875)*_b[8];
		const Real _t28 = _t27 + _t6;
		const Real _t29 = (0.1875)*_b[9];
		const Real _t30 = (0.1875)*_b[2] + _t29;
		const Real _t31 = (0.0625)*_b[10];
		const Real _t32 = (0.0625)*_b[7];
		const Real _t33 = _t31 + _t32;
		const Real _t34 = (0.125)*_b[1];
		const Real _t35 = (0.03125)*_b[4];
		const Real _t36 = (0.125)*_b[8];
		const Real _t37 = _t10 + _t36;
		const Real _t38 = (0.03125)*_b[7];
		const Real _t39 = (0.125)*_b[10];
		const Real _t40 = (0.03125)*_b[11];
		const Real _t41 = _t38 + _t39 + _t40;
		const Real _t42 = (0.15625)*_b[2];
		const Real _t43 = (0.078125)*_b[8];
		const Real _t44 = _t14 + _t43;
		const Real _t45 = (0.015625)*_b[12];
		const Real _t46 = (0.078125)*_b[11];
		const Real _t47 = (0.015625)*_b[5] + _t46;
		const Real _t48 = _t45 + _t47;
		const Real _t49 = (0.015625)*_b[7];
		const Real _t50 = (0.15625)*_b[10];
		const Real _t51 = (0.15625)*_b[9];
		const Real _t52 = (0.15625)*_b[3] + _t49 + _t50 + _t51;
		const Real _t53 = (0.078125)*_b[1] + (0.078125)*_b[4] + _t42 + _t44 + _t48 + _t52;
		const Real _t54 = (0.125)*_b[14];
		const Real _t55 = _t21 + _t34 + _t54;
		const Real _t56 = (0.0625)*_b[13];
		const Real _t57 = (0.0625)*_b[2];
		const Real _t58 = (0.0625)*_b[15];
		const Real _t59 = (0.03125)*_b[13];
		const Real _t60 = (0.09375)*_b[14];
		const Real _t61 = (0.03125)*_b[3];
		const Real _t62 = (0.09375)*_b[15];
		const Real _t63 = _t61 + _t62;
		const Real _t64 = _t10 + _t27;
		const Real _t65 = (0.09375)*_b[2] + _t29;
		const Real _t66 = (0.0625)*_b[1];
		const Real _t67 = (0.015625)*_b[17];
		const Real _t68 = (0.015625)*_b[4];
		const Real _t69 = _t67 + _t68;
		const Real _t70 = _t14 + _t36;
		const Real _t71 = (0.0625)*_b[14];
		const Real _t72 = (0.0625)*_b[16];
		const Real _t73 = (0.015625)*_b[13];
		const Real _t74 = _t62 + _t73;
		const Real _t75 = _t71 + _t72 + _t74;
		const Real _t76 = _t26 + _t75;
		const Real _t77 = _t41 + _t65 + _t66 + _t69 + _t70 + _t76;
		const Real _t78 = (0.375)*_b[13];
		const Real _t79 = (0.125)*_b[18];
		const Real _t80 = (0.0625)*_b[18];
		const Real _t81 = (0.1875)*_b[14];
		const Real _t82 = (0.1875)*_b[13] + _t81;
		const Real _t83 = (0.0625)*_b[19];
		const Real _t84 = _t66 + _t83;
		const Real _t85 = (0.09375)*_b[7];
		const Real _t86 = (0.03125)*_b[2];
		const Real _t87 = (0.09375)*_b[9];
		const Real _t88 = (0.03125)*_b[18];
		const Real _t89 = _t62 + _t88;
		const Real _t90 = (0.09375)*_b[13] + _t81;
		const Real _t91 = (0.015625)*_b[18];
		const Real _t92 = (0.015625)*_b[3];
		const Real _t93 = (0.140625)*_b[15];
		const Real _t94 = _t91 + _t92 + _t93;
		const Real _t95 = (0.015625)*_b[21];
		const Real _t96 = (0.046875)*_b[10] + (0.046875)*_b[16] + _t95;
		const Real _t97 = _t94 + _t96;
		const Real _t98 = (0.046875)*_b[13] + (0.140625)*_b[14];
		const Real _t99 = (0.046875)*_b[19] + (0.046875)*_b[20];
		const Real _t100 = (0.046875)*_b[2] + (0.140625)*_b[9] + _t99;
		const Real _t101 = (0.046875)*_b[1] + (0.046875)*_b[7] + (0.140625)*_b[8] + _t100 + _t14 + _t97 + _t98;
		const Real _t102 = (0.25)*_b[18];
		const Real _t103 = (0.0625)*_b[22];
		const Real _t104 = (0.03125)*_b[22];
		const Real _t105 = (0.125)*_b[19];
		const Real _t106 = (0.03125)*_b[1];
		const Real _t107 = (0.03125)*_b[23];
		const Real _t108 = _t105 + _t106 + _t107;
		const Real _t109 = (0.015625)*_b[24];
		const Real _t110 = (0.015625)*_b[22];
		const Real _t111 = _t110 + _t62;
		const Real _t112 = _t109 + _t111;
		const Real _t113 = (0.0625)*_b[9];
		const Real _t114 = (0.0625)*_b[20];
		const Real _t115 = (0.015625)*_b[2];
		const Real _t116 = _t113 + _t114 + _t115 + _t80;
		const Real _t117 = _t108 + _t112 + _t116 + _t32 + _t70 + _t90;
		const Real _t118 = (0.3125)*_b[18];
		const Real _t119 = (0.15625)*_b[22];
		const Real _t120 = (0.03125)*_b[25];
		const Real _t121 = (0.15625)*_b[13];
		const Real _t122 = (0.015625)*_b[26];
		const Real _t123 = (0.078125)*_b[23];
		const Real _t124 = (0.015625)*_b[25] + _t123;
		const Real _t125 = _t122 + _t124;
		const Real _t126 = (0.015625)*_b[1];
		const Real _t127 = (0.15625)*_b[14];
		const Real _t128 = (0.15625)*_b[19];
		const Real _t129 = (0.15625)*_b[18] + _t126 + _t127 + _t128;
		const Real _t130 = (0.078125)*_b[22] + (0.078125)*_b[7] + _t121 + _t125 + _t129 + _t44;
		const Real _t131 = (0.09375)*_b[25];
		const Real _t132 = (0.015625)*_b[27];
		const Real _t133 = (0.234375)*_b[13] + (0.234375)*_b[22] + _t118 + _t131 + _t132 + _t14 + _t85;
		const Real _t134 = _t43 + _t49;
		const Real _t135 = (0.078125)*_b[13] + (0.078125)*_b[25] + _t119 + _t122 + _t123 + _t129 + _t132 + _t134;
		const Real _t136 = (0.0625)*_b[25];
		const Real _t137 = (0.03125)*_b[26];
		const Real _t138 = (0.03125)*_b[8];
		const Real _t139 = _t137 + _t138 + _t54;
		const Real _t140 = (0.1875)*_b[19];
		const Real _t141 = (0.09375)*_b[22] + _t140;
		const Real _t142 = (0.125)*_b[23];
		const Real _t143 = _t132 + _t142;
		const Real _t144 = _t109 + _t116 + _t136 + _t139 + _t141 + _t143 + _t74;
		const Real _t145 = (0.140625)*_b[19] + (0.046875)*_b[22];
		const Real _t146 = (0.046875)*_b[14] + (0.046875)*_b[9];
		const Real _t147 = (0.140625)*_b[20] + (0.046875)*_b[24] + _t146;
		const Real _t148 = (0.140625)*_b[23] + (0.046875)*_b[25] + (0.046875)*_b[26] + _t132 + _t145 + _t147 + _t97;
		const Real _t149 = (0.0625)*_b[26];
		const Real _t150 = (0.125)*_b[16];
		const Real _t151 = _t120 + _t150 + _t40;
		const Real _t152 = (0.1875)*_b[20];
		const Real _t153 = (0.09375)*_b[24] + _t152;
		const Real _t154 = (0.0625)*_b[21];
		const Real _t155 = _t154 + _t31 + _t83;
		const Real _t156 = _t111 + _t143 + _t149 + _t151 + _t153 + _t155 + _t69;
		const Real _t157 = (0.15625)*_b[24];
		const Real _t158 = (0.15625)*_b[16];
		const Real _t159 = (0.15625)*_b[20];
		const Real _t160 = (0.15625)*_b[21] + _t158 + _t159;
		const Real _t161 = (0.078125)*_b[17] + (0.078125)*_b[26] + _t124 + _t132 + _t157 + _t160 + _t48;
		const Real _t162 = (0.09375)*_b[12];
		const Real _t163 = (0.3125)*_b[21];
		const Real _t164 = (0.09375)*_b[26];
		const Real _t165 = (0.234375)*_b[17] + (0.234375)*_b[24] + _t132 + _t162 + _t163 + _t164 + _t17;
		const Real _t166 = (0.03125)*_b[27];
		const Real _t167 = (0.125)*_b[25];
		const Real _t168 = (0.1875)*_b[22] + _t140;
		const Real _t169 = _t142 + _t166;
		const Real _t170 = (0.03125)*_b[24];
		const Real _t171 = (0.09375)*_b[20];
		const Real _t172 = _t149 + _t71;
		const Real _t173 = (0.1875)*_b[23];
		const Real _t174 = _t166 + _t173;
		const Real _t175 = (0.09375)*_b[19];
		const Real _t176 = _t136 + _t72;
		const Real _t177 = (0.03125)*_b[21];
		const Real _t178 = _t177 + _t62;
		const Real _t179 = (0.125)*_b[21];
		const Real _t180 = (0.125)*_b[26];
		const Real _t181 = (0.03125)*_b[17];
		const Real _t182 = (0.1875)*_b[24] + _t152;
		const Real _t183 = (0.03125)*_b[12];
		const Real _t184 = (0.15625)*_b[17];
		const Real _t185 = (0.375)*_b[22];
		const Real _t186 = (0.25)*_b[25];
		const Real _t187 = (0.0625)*_b[27];
		const Real _t188 = _t173 + _t187;
		const Real _t189 = (0.0625)*_b[24];
		const Real _t190 = _t105 + _t180;
		const Real _t191 = (0.25)*_b[23];
		const Real _t192 = (0.125)*_b[20];
		const Real _t193 = _t167 + _t191 + _t192;
		const Real _t194 = (0.0625)*_b[17];
		const Real _t195 = (0.25)*_b[21];
		const Real _t196 = (0.375)*_b[24];
		const Real _t197 = (0.25)*_b[26];
		const Real _t198 = (0.125)*_b[27];
		const Real _t199 = _t186 + _t191;
		const Real _t200 = (0.5)*_b[25];
		const Real _t201 = (0.25)*_b[27];
		const Real _t202 = (0.5)*_b[26];
		const Real _t203 = (0.5)*_b[27];
		const Real _t204 = (0.03125)*_b[6];
		const Real _t205 = (0.375)*_b[4];
		const Real _t206 = (0.25)*_b[5];
		const Real _t207 = (0.0625)*_b[6];
		const Real _t208 = (0.125)*_b[6];
		const Real _t209 = (0.5)*_b[5];
		const Real _t210 = (0.25)*_b[6];
		const Real _t211 = (0.5)*_b[6];
		const Real _t212 = (0.078125)*_b[2] + (0.078125)*_b[5] + _t12 + _t126 + _t17 + _t43 + _t45 + _t46 + _t52;
		const Real _t213 = (0.125)*_b[5];
		const Real _t214 = (0.1875)*_b[10];
		const Real _t215 = (0.1875)*_b[4] + _t214;
		const Real _t216 = (0.125)*_b[11];
		const Real _t217 = _t204 + _t216;
		const Real _t218 = _t138 + _t183 + _t24;
		const Real _t219 = (0.1875)*_b[11];
		const Real _t220 = _t207 + _t219;
		const Real _t221 = (0.0625)*_b[12];
		const Real _t222 = _t113 + _t221;
		const Real _t223 = (0.25)*_b[11];
		const Real _t224 = _t206 + _t223;
		const Real _t225 = (0.125)*_b[12];
		const Real _t226 = _t225 + _t39;
		const Real _t227 = (0.25)*_b[12];
		const Real _t228 = (0.5)*_b[12];
		const Real _t229 = (0.0625)*_b[5];
		const Real _t230 = (0.09375)*_b[4] + _t214;
		const Real _t231 = _t17 + _t216;
		const Real _t232 = _t115 + _t218 + _t229 + _t230 + _t231 + _t67 + _t76;
		const Real _t233 = (0.09375)*_b[16];
		const Real _t234 = _t204 + _t219;
		const Real _t235 = _t150 + _t213 + _t223;
		const Real _t236 = (0.140625)*_b[16] + (0.046875)*_b[17] + _t146 + _t94;
		const Real _t237 = (0.140625)*_b[10] + (0.046875)*_b[4] + _t95 + _t99;
		const Real _t238 = (0.140625)*_b[11] + (0.046875)*_b[12] + (0.046875)*_b[5] + _t17 + _t236 + _t237;
		const Real _t239 = (0.09375)*_b[10];
		const Real _t240 = _t114 + _t229;
		const Real _t241 = (0.1875)*_b[16];
		const Real _t242 = (0.09375)*_b[17] + _t241;
		const Real _t243 = (0.1875)*_b[17] + _t241;
		const Real _t244 = (0.375)*_b[17];
		const Real _t245 = _t107 + _t13 + _t192;
		const Real _t246 = _t112 + _t155 + _t221 + _t231 + _t242 + _t245 + _t68;
		const Real _t247 = (0.078125)*_b[12] + (0.078125)*_b[24] + _t125 + _t160 + _t17 + _t184 + _t47;
		const Real _t248 = (0.046875)*_b[21];
		const Real _t249 = (0.109375)*_b[16];
		const Real _t250 = (0.109375)*_b[19];
		const Real _t251 = _t40 + _t69;
		const Real _t252 = (0.109375)*_b[20];
		const Real _t253 = _t115 + _t125;
		const Real _t254 = (0.109375)*_b[14] + _t93;
		const Real _t255 = _t138 + _t73;
		const Real _t256 = (0.046875)*_b[18] + _t92 + _t96;
		const Real _t257 = _t126 + _t134;
		const Real _t258 = _t109 + _t257;
		const Real _t259 = (0.109375)*_b[10];
		const Real _t260 = _t107 + _t110;
		const Real _t261 = _t109 + _t260;
		const Real _t262 = _t115 + _t255;
		const Real _t263 = (0.109375)*_b[9] + _t93;
		const Real _t264 = (0.046875)*_b[3] + _t91;
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				_t0 + _t1,
				(0.25)*_b[2] + _t1 + _t2,
				(0.375)*_b[1] + _t3 + _t4 + _t5,
				_t4 + _t6 + _t7 + _t8 + _t9,
				(0.15625)*_b[1] + (0.3125)*_b[2] + _t10 + _t11 + _t12 + _t13,
				_t18,
				_t0 + _t19,
				_t2 + _t20 + _t22,
				(0.125)*_b[2] + _t22 + _t25 + _t3,
				(0.1875)*_b[1] + _t26 + _t28 + _t30 + _t33,
				_t30 + _t34 + _t35 + _t37 + _t41 + _t5,
				_t53,
				(0.25)*_b[13] + _t19 + _t2,
				(0.125)*_b[13] + _t20 + _t3 + _t55,
				_t25 + _t55 + _t56 + _t57 + _t58 + _t6,
				(0.03125)*_b[16] + _t15 + _t33 + _t59 + _t60 + _t63 + _t64 + _t65,
				_t77,
				(0.375)*_b[7] + _t3 + _t78 + _t79,
				(0.1875)*_b[7] + _t28 + _t80 + _t82 + _t84,
				(0.03125)*_b[20] + _t64 + _t84 + _t85 + _t86 + _t87 + _t89 + _t90,
				_t101,
				_t102 + _t103 + _t20 + _t6 + _t78,
				_t104 + _t108 + _t23 + _t37 + _t79 + _t82,
				_t117,
				(0.3125)*_b[13] + (0.15625)*_b[7] + _t10 + _t118 + _t119 + _t120,
				_t130,
				_t133,
			},
			{
				_t133,
				_t135,
				_t144,
				_t148,
				_t156,
				_t161,
				_t165,
				(0.3125)*_b[22] + (0.15625)*_b[25] + _t118 + _t121 + _t166 + _t38,
				_t139 + _t167 + _t168 + _t169 + _t59 + _t79,
				(0.03125)*_b[9] + _t131 + _t141 + _t170 + _t171 + _t172 + _t174 + _t89,
				(0.03125)*_b[10] + _t104 + _t153 + _t164 + _t174 + _t175 + _t176 + _t178,
				_t151 + _t169 + _t179 + _t180 + _t181 + _t182,
				(0.3125)*_b[24] + (0.15625)*_b[26] + _t163 + _t166 + _t183 + _t184,
				_t102 + _t185 + _t186 + _t187 + _t56,
				(0.1875)*_b[25] + _t168 + _t172 + _t188 + _t80,
				_t103 + _t187 + _t189 + _t190 + _t193 + _t58,
				(0.1875)*_b[26] + _t154 + _t176 + _t182 + _t188,
				_t187 + _t194 + _t195 + _t196 + _t197,
				(0.375)*_b[25] + _t185 + _t198 + _t79,
				(0.125)*_b[22] + _t190 + _t198 + _t199,
				(0.125)*_b[24] + _t193 + _t197 + _t198,
				(0.375)*_b[26] + _t179 + _t196 + _t198,
				(0.25)*_b[22] + _t200 + _t201,
				_t197 + _t199 + _t201,
				(0.25)*_b[24] + _t201 + _t202,
				_t200 + _t203,
				_t202 + _t203,
				_b[27],
			},
			{
				_t18,
				(0.3125)*_b[4] + (0.15625)*_b[5] + _t106 + _t11 + _t204 + _t42,
				_t205 + _t206 + _t207 + _t57 + _t8,
				(0.375)*_b[5] + _t205 + _t208 + _t5,
				(0.25)*_b[4] + _t209 + _t210,
				_t209 + _t211,
				_b[6],
				_t212,
				_t213 + _t215 + _t217 + _t218 + _t5 + _t86,
				(0.1875)*_b[5] + _t215 + _t220 + _t222 + _t26,
				(0.125)*_b[4] + _t208 + _t224 + _t226,
				_t210 + _t224 + _t227,
				_t211 + _t228,
				_t232,
				(0.03125)*_b[14] + _t16 + _t181 + _t222 + _t230 + _t233 + _t234 + _t63,
				_t194 + _t207 + _t226 + _t235 + _t58 + _t9,
				(0.125)*_b[17] + _t208 + _t227 + _t235,
				(0.25)*_b[17] + _t210 + _t228,
				_t238,
				(0.03125)*_b[19] + _t162 + _t178 + _t234 + _t239 + _t240 + _t242 + _t35,
				(0.1875)*_b[12] + _t154 + _t220 + _t240 + _t243,
				(0.375)*_b[12] + _t179 + _t208 + _t244,
				_t246,
				_t170 + _t179 + _t217 + _t225 + _t243 + _t245,
				_t189 + _t195 + _t207 + _t227 + _t244,
				_t247,
				(0.15625)*_b[12] + (0.3125)*_b[17] + _t137 + _t157 + _t163 + _t204,
				_t165,
			},
			{
				_t165,
				_t161,
				_t156,
				_t148,
				_t144,
				_t135,
				_t133,
				_t247,
				(0.09375)*_b[21] + _t111 + _t125 + _t158 + _t159 + _t189 + _t194 + _t31 + _t48 + _t68 + _t83,
				(0.078125)*_b[10] + _t104 + _t125 + _t147 + _t248 + _t249 + _t250 + _t251 + _t94,
				(0.078125)*_b[9] + _t145 + _t170 + _t252 + _t253 + _t254 + _t255 + _t256,
				(0.09375)*_b[18] + _t103 + _t113 + _t114 + _t127 + _t128 + _t253 + _t258 + _t56 + _t62,
				_t130,
				_t246,
				(0.078125)*_b[19] + _t236 + _t248 + _t252 + _t259 + _t261 + _t35 + _t48,
				(0.15625)*_b[15] + _t171 + _t175 + _t177 + _t233 + _t239 + _t251 + _t261 + _t262 + _t60 + _t61 + _t87 + _t88,
				(0.078125)*_b[20] + _t250 + _t256 + _t258 + _t260 + _t263 + _t86 + _t98,
				_t117,
				_t238,
				(0.078125)*_b[14] + _t181 + _t237 + _t249 + _t262 + _t263 + _t264 + _t48,
				(0.078125)*_b[16] + _t100 + _t251 + _t254 + _t257 + _t259 + _t264 + _t59 + _t95,
				_t101,
				_t232,
				(0.09375)*_b[3] + _t257 + _t48 + _t50 + _t51 + _t57 + _t67 + _t75 + _t9,
				_t77,
				_t212,
				_t53,
				_t18,
			},
		}};
		#pragma endregion
	}
}
