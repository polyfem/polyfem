#include "../P3TriCGV.hpp"
namespace miso {
	std::array<RealVector<45>, 8> P3TriCGV::subdiv_0_2p4_1p2(const RealVector<45> &_b) {
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
		const Real _t19 = (0.125)*_b[6];
		const Real _t20 = _t11 + _t19;
		const Real _t21 = (0.125)*_b[1];
		const Real _t22 = (0.125)*_b[7];
		const Real _t23 = _t21 + _t22;
		const Real _t24 = (0.0625)*_b[0];
		const Real _t25 = (0.0625)*_b[2];
		const Real _t26 = _t24 + _t25;
		const Real _t27 = (0.0625)*_b[6];
		const Real _t28 = (0.0625)*_b[8];
		const Real _t29 = _t27 + _t28;
		const Real _t30 = _t12 + _t15 + _t23 + _t26 + _t29 + _t9;
		const Real _t31 = (0.375)*_b[6];
		const Real _t32 = (0.125)*_b[9];
		const Real _t33 = (0.1875)*_b[3];
		const Real _t34 = (0.1875)*_b[4];
		const Real _t35 = (0.0625)*_b[1];
		const Real _t36 = _t24 + _t35;
		const Real _t37 = (0.1875)*_b[6];
		const Real _t38 = (0.1875)*_b[7];
		const Real _t39 = _t37 + _t38;
		const Real _t40 = (0.0625)*_b[10];
		const Real _t41 = (0.0625)*_b[9];
		const Real _t42 = _t40 + _t41;
		const Real _t43 = (0.03125)*_b[0];
		const Real _t44 = (0.03125)*_b[2];
		const Real _t45 = _t35 + _t43 + _t44;
		const Real _t46 = (0.09375)*_b[6];
		const Real _t47 = (0.09375)*_b[8];
		const Real _t48 = _t38 + _t46 + _t47;
		const Real _t49 = (0.03125)*_b[9];
		const Real _t50 = (0.09375)*_b[3] + _t49;
		const Real _t51 = (0.03125)*_b[11];
		const Real _t52 = (0.09375)*_b[5] + _t51;
		const Real _t53 = _t34 + _t40 + _t45 + _t48 + _t50 + _t52;
		const Real _t54 = (0.0625)*_b[12];
		const Real _t55 = (0.25)*_b[9];
		const Real _t56 = _t24 + _t31 + _t54 + _t55 + _t8;
		const Real _t57 = (0.125)*_b[4];
		const Real _t58 = (0.125)*_b[10];
		const Real _t59 = _t32 + _t58;
		const Real _t60 = (0.03125)*_b[1];
		const Real _t61 = _t43 + _t60;
		const Real _t62 = (0.03125)*_b[12];
		const Real _t63 = (0.03125)*_b[13];
		const Real _t64 = _t62 + _t63;
		const Real _t65 = _t12 + _t39 + _t57 + _t59 + _t61 + _t64;
		const Real _t66 = (0.0625)*_b[3];
		const Real _t67 = (0.0625)*_b[5];
		const Real _t68 = _t66 + _t67;
		const Real _t69 = (0.0625)*_b[11];
		const Real _t70 = _t41 + _t58 + _t69;
		const Real _t71 = (0.015625)*_b[0] + (0.015625)*_b[2] + _t60;
		const Real _t72 = (0.015625)*_b[12] + (0.015625)*_b[14] + _t63;
		const Real _t73 = _t48 + _t57 + _t68 + _t70 + _t71 + _t72;
		const Real _t74 = (0.5)*_b[15];
		const Real _t75 = (0.25)*_b[15];
		const Real _t76 = (0.25)*_b[16];
		const Real _t77 = _t75 + _t76;
		const Real _t78 = (0.125)*_b[17];
		const Real _t79 = _t14 + _t78;
		const Real _t80 = (0.125)*_b[15];
		const Real _t81 = _t76 + _t80;
		const Real _t82 = _t11 + _t6 + _t79 + _t81;
		const Real _t83 = (0.25)*_b[18];
		const Real _t84 = _t8 + _t83;
		const Real _t85 = (0.125)*_b[16];
		const Real _t86 = _t80 + _t85;
		const Real _t87 = (0.125)*_b[18];
		const Real _t88 = (0.125)*_b[19];
		const Real _t89 = _t57 + _t88;
		const Real _t90 = _t87 + _t89;
		const Real _t91 = (0.0625)*_b[15];
		const Real _t92 = (0.0625)*_b[17];
		const Real _t93 = _t85 + _t91 + _t92;
		const Real _t94 = (0.0625)*_b[18];
		const Real _t95 = (0.0625)*_b[20];
		const Real _t96 = _t94 + _t95;
		const Real _t97 = _t68 + _t89 + _t96;
		const Real _t98 = _t21 + _t26 + _t93 + _t97;
		const Real _t99 = (0.125)*_b[21];
		const Real _t100 = _t80 + _t99;
		const Real _t101 = (0.0625)*_b[22];
		const Real _t102 = (0.0625)*_b[7];
		const Real _t103 = _t101 + _t102;
		const Real _t104 = (0.0625)*_b[21];
		const Real _t105 = (0.0625)*_b[16];
		const Real _t106 = _t105 + _t91;
		const Real _t107 = _t104 + _t106;
		const Real _t108 = (0.03125)*_b[6];
		const Real _t109 = (0.03125)*_b[8];
		const Real _t110 = _t108 + _t109;
		const Real _t111 = (0.03125)*_b[15];
		const Real _t112 = (0.03125)*_b[17];
		const Real _t113 = (0.03125)*_b[21] + (0.03125)*_b[23];
		const Real _t114 = _t105 + _t111 + _t112 + _t113;
		const Real _t115 = _t103 + _t110 + _t114 + _t45 + _t97;
		const Real _t116 = (0.1875)*_b[18];
		const Real _t117 = _t116 + _t24;
		const Real _t118 = (0.0625)*_b[24];
		const Real _t119 = (0.1875)*_b[21];
		const Real _t120 = _t119 + _t41;
		const Real _t121 = _t118 + _t120;
		const Real _t122 = _t37 + _t91;
		const Real _t123 = _t117 + _t121 + _t122 + _t33;
		const Real _t124 = (0.09375)*_b[18];
		const Real _t125 = _t124 + _t61;
		const Real _t126 = (0.09375)*_b[7];
		const Real _t127 = (0.09375)*_b[4] + _t126;
		const Real _t128 = (0.09375)*_b[19];
		const Real _t129 = (0.03125)*_b[16] + _t128;
		const Real _t130 = _t111 + _t129;
		const Real _t131 = (0.03125)*_b[24];
		const Real _t132 = (0.09375)*_b[21];
		const Real _t133 = (0.03125)*_b[25];
		const Real _t134 = (0.09375)*_b[22];
		const Real _t135 = (0.03125)*_b[10] + _t134;
		const Real _t136 = _t133 + _t135;
		const Real _t137 = _t131 + _t132 + _t136;
		const Real _t138 = _t130 + _t137;
		const Real _t139 = _t125 + _t127 + _t138 + _t46 + _t50;
		const Real _t140 = _t129 + _t136;
		const Real _t141 = _t127 + _t140;
		const Real _t142 = (0.046875)*_b[18];
		const Real _t143 = (0.046875)*_b[20];
		const Real _t144 = _t142 + _t143 + _t71;
		const Real _t145 = (0.015625)*_b[24];
		const Real _t146 = (0.015625)*_b[26];
		const Real _t147 = (0.046875)*_b[21];
		const Real _t148 = (0.046875)*_b[23];
		const Real _t149 = (0.015625)*_b[11] + (0.015625)*_b[9] + _t147 + _t148;
		const Real _t150 = _t145 + _t146 + _t149;
		const Real _t151 = (0.015625)*_b[15] + (0.015625)*_b[17];
		const Real _t152 = (0.046875)*_b[6] + (0.046875)*_b[8] + _t151;
		const Real _t153 = (0.046875)*_b[3] + (0.046875)*_b[5] + _t141 + _t144 + _t150 + _t152;
		const Real _t154 = (0.25)*_b[27];
		const Real _t155 = (0.125)*_b[27];
		const Real _t156 = (0.125)*_b[28];
		const Real _t157 = _t156 + _t21;
		const Real _t158 = (0.0625)*_b[27];
		const Real _t159 = (0.0625)*_b[29];
		const Real _t160 = _t159 + _t78;
		const Real _t161 = _t157 + _t158 + _t160 + _t26 + _t81;
		const Real _t162 = (0.125)*_b[30];
		const Real _t163 = _t162 + _t83;
		const Real _t164 = (0.0625)*_b[28];
		const Real _t165 = _t158 + _t164;
		const Real _t166 = (0.0625)*_b[4];
		const Real _t167 = _t166 + _t66;
		const Real _t168 = (0.0625)*_b[30];
		const Real _t169 = (0.0625)*_b[31];
		const Real _t170 = _t168 + _t169;
		const Real _t171 = _t167 + _t170 + _t87 + _t88;
		const Real _t172 = (0.03125)*_b[27];
		const Real _t173 = (0.03125)*_b[29];
		const Real _t174 = _t164 + _t172 + _t173;
		const Real _t175 = (0.03125)*_b[3];
		const Real _t176 = (0.03125)*_b[5];
		const Real _t177 = (0.03125)*_b[30] + (0.03125)*_b[32] + _t169;
		const Real _t178 = _t166 + _t175 + _t176 + _t177 + _t88 + _t96;
		const Real _t179 = _t174 + _t178 + _t45 + _t93;
		const Real _t180 = (0.0625)*_b[33];
		const Real _t181 = _t158 + _t27;
		const Real _t182 = _t180 + _t181;
		const Real _t183 = _t100 + _t12 + _t163 + _t182 + _t24;
		const Real _t184 = (0.03125)*_b[34];
		const Real _t185 = (0.03125)*_b[28];
		const Real _t186 = (0.03125)*_b[7];
		const Real _t187 = _t185 + _t186;
		const Real _t188 = _t184 + _t187;
		const Real _t189 = _t101 + _t188;
		const Real _t190 = (0.03125)*_b[33];
		const Real _t191 = _t108 + _t172;
		const Real _t192 = _t190 + _t191;
		const Real _t193 = _t189 + _t192;
		const Real _t194 = _t107 + _t171 + _t193 + _t61;
		const Real _t195 = (0.015625)*_b[27];
		const Real _t196 = (0.015625)*_b[29];
		const Real _t197 = (0.015625)*_b[6];
		const Real _t198 = (0.015625)*_b[8];
		const Real _t199 = _t195 + _t196 + _t197 + _t198;
		const Real _t200 = (0.015625)*_b[33] + (0.015625)*_b[35];
		const Real _t201 = _t199 + _t200;
		const Real _t202 = _t189 + _t201;
		const Real _t203 = _t114 + _t178 + _t202 + _t71;
		const Real _t204 = (0.375)*_b[27];
		const Real _t205 = (0.125)*_b[36];
		const Real _t206 = (0.1875)*_b[15];
		const Real _t207 = (0.1875)*_b[16];
		const Real _t208 = (0.1875)*_b[27];
		const Real _t209 = (0.1875)*_b[28];
		const Real _t210 = _t208 + _t209;
		const Real _t211 = (0.0625)*_b[36];
		const Real _t212 = (0.0625)*_b[37];
		const Real _t213 = _t211 + _t212;
		const Real _t214 = (0.09375)*_b[27];
		const Real _t215 = (0.09375)*_b[29];
		const Real _t216 = _t209 + _t214 + _t215;
		const Real _t217 = (0.03125)*_b[36];
		const Real _t218 = (0.09375)*_b[15] + _t217;
		const Real _t219 = (0.03125)*_b[38];
		const Real _t220 = (0.09375)*_b[17] + _t219;
		const Real _t221 = _t207 + _t212 + _t216 + _t218 + _t220 + _t45;
		const Real _t222 = (0.0625)*_b[39];
		const Real _t223 = (0.1875)*_b[30];
		const Real _t224 = _t211 + _t223;
		const Real _t225 = _t222 + _t224;
		const Real _t226 = _t208 + _t66;
		const Real _t227 = _t117 + _t206 + _t225 + _t226;
		const Real _t228 = (0.03125)*_b[40];
		const Real _t229 = (0.09375)*_b[28] + _t228;
		const Real _t230 = (0.09375)*_b[16] + _t128 + _t229;
		const Real _t231 = (0.03125)*_b[4];
		const Real _t232 = _t175 + _t231;
		const Real _t233 = (0.03125)*_b[39];
		const Real _t234 = (0.09375)*_b[30];
		const Real _t235 = (0.09375)*_b[31];
		const Real _t236 = (0.03125)*_b[37] + _t235;
		const Real _t237 = _t234 + _t236;
		const Real _t238 = _t233 + _t237;
		const Real _t239 = _t232 + _t238;
		const Real _t240 = _t125 + _t214 + _t218 + _t230 + _t239;
		const Real _t241 = (0.015625)*_b[39];
		const Real _t242 = (0.015625)*_b[41];
		const Real _t243 = (0.046875)*_b[30];
		const Real _t244 = (0.046875)*_b[32];
		const Real _t245 = (0.015625)*_b[36] + (0.015625)*_b[38] + _t236 + _t243 + _t244;
		const Real _t246 = _t241 + _t242 + _t245;
		const Real _t247 = (0.015625)*_b[3] + (0.015625)*_b[5];
		const Real _t248 = _t231 + _t247;
		const Real _t249 = (0.046875)*_b[27] + (0.046875)*_b[29] + _t248;
		const Real _t250 = (0.046875)*_b[15] + (0.046875)*_b[17] + _t144 + _t230 + _t246 + _t249;
		const Real _t251 = (0.25)*_b[36];
		const Real _t252 = (0.0625)*_b[42];
		const Real _t253 = _t204 + _t24 + _t251 + _t252 + _t75;
		const Real _t254 = (0.125)*_b[37];
		const Real _t255 = _t205 + _t254;
		const Real _t256 = (0.03125)*_b[42];
		const Real _t257 = (0.03125)*_b[43];
		const Real _t258 = _t256 + _t257;
		const Real _t259 = _t210 + _t255 + _t258 + _t61 + _t86;
		const Real _t260 = (0.0625)*_b[38];
		const Real _t261 = _t211 + _t254 + _t260;
		const Real _t262 = (0.015625)*_b[42] + (0.015625)*_b[44] + _t257;
		const Real _t263 = _t216 + _t261 + _t262 + _t71 + _t93;
		const Real _t264 = (0.5)*_b[2];
		const Real _t265 = _t3 + _t6;
		const Real _t266 = (0.25)*_b[5];
		const Real _t267 = _t266 + _t9;
		const Real _t268 = (0.5)*_b[5];
		const Real _t269 = (0.125)*_b[8];
		const Real _t270 = (0.25)*_b[8];
		const Real _t271 = (0.1875)*_b[5];
		const Real _t272 = _t25 + _t35;
		const Real _t273 = (0.1875)*_b[8];
		const Real _t274 = _t273 + _t38;
		const Real _t275 = _t40 + _t69;
		const Real _t276 = (0.125)*_b[11];
		const Real _t277 = (0.375)*_b[8];
		const Real _t278 = _t276 + _t58;
		const Real _t279 = _t44 + _t60;
		const Real _t280 = (0.03125)*_b[14];
		const Real _t281 = _t280 + _t63;
		const Real _t282 = _t15 + _t274 + _t278 + _t279 + _t281 + _t57;
		const Real _t283 = (0.25)*_b[11];
		const Real _t284 = (0.0625)*_b[14];
		const Real _t285 = _t25 + _t266 + _t277 + _t283 + _t284;
		const Real _t286 = (0.25)*_b[17];
		const Real _t287 = _t286 + _t76;
		const Real _t288 = (0.5)*_b[17];
		const Real _t289 = _t78 + _t85;
		const Real _t290 = (0.125)*_b[20];
		const Real _t291 = _t290 + _t89;
		const Real _t292 = (0.25)*_b[20];
		const Real _t293 = _t266 + _t292;
		const Real _t294 = (0.0625)*_b[23];
		const Real _t295 = _t105 + _t92;
		const Real _t296 = _t294 + _t295;
		const Real _t297 = (0.125)*_b[23];
		const Real _t298 = (0.09375)*_b[20];
		const Real _t299 = _t279 + _t298;
		const Real _t300 = (0.03125)*_b[26];
		const Real _t301 = (0.09375)*_b[23];
		const Real _t302 = _t300 + _t301;
		const Real _t303 = _t112 + _t302 + _t47;
		const Real _t304 = _t141 + _t299 + _t303 + _t52;
		const Real _t305 = (0.1875)*_b[20];
		const Real _t306 = _t25 + _t305;
		const Real _t307 = (0.0625)*_b[26];
		const Real _t308 = (0.1875)*_b[23];
		const Real _t309 = _t308 + _t69;
		const Real _t310 = _t307 + _t309;
		const Real _t311 = _t273 + _t92;
		const Real _t312 = _t271 + _t306 + _t310 + _t311;
		const Real _t313 = (0.125)*_b[29];
		const Real _t314 = (0.25)*_b[29];
		const Real _t315 = _t159 + _t164;
		const Real _t316 = _t166 + _t67;
		const Real _t317 = (0.0625)*_b[32];
		const Real _t318 = _t169 + _t317;
		const Real _t319 = _t290 + _t316 + _t318 + _t88;
		const Real _t320 = (0.125)*_b[32];
		const Real _t321 = _t292 + _t320;
		const Real _t322 = (0.03125)*_b[35];
		const Real _t323 = _t109 + _t173;
		const Real _t324 = _t322 + _t323;
		const Real _t325 = _t189 + _t324;
		const Real _t326 = _t279 + _t296 + _t319 + _t325;
		const Real _t327 = (0.0625)*_b[35];
		const Real _t328 = _t28 + _t327;
		const Real _t329 = _t297 + _t328;
		const Real _t330 = _t15 + _t160 + _t25 + _t321 + _t329;
		const Real _t331 = (0.1875)*_b[17];
		const Real _t332 = (0.1875)*_b[29];
		const Real _t333 = _t209 + _t332;
		const Real _t334 = _t212 + _t260;
		const Real _t335 = (0.375)*_b[29];
		const Real _t336 = (0.125)*_b[38];
		const Real _t337 = (0.03125)*_b[41];
		const Real _t338 = (0.09375)*_b[32];
		const Real _t339 = _t236 + _t338;
		const Real _t340 = _t337 + _t339;
		const Real _t341 = _t176 + _t231;
		const Real _t342 = _t215 + _t341;
		const Real _t343 = _t220 + _t230 + _t299 + _t340 + _t342;
		const Real _t344 = (0.0625)*_b[41];
		const Real _t345 = (0.1875)*_b[32];
		const Real _t346 = _t260 + _t345;
		const Real _t347 = _t344 + _t346;
		const Real _t348 = _t332 + _t67;
		const Real _t349 = _t306 + _t331 + _t347 + _t348;
		const Real _t350 = (0.03125)*_b[44];
		const Real _t351 = _t257 + _t350;
		const Real _t352 = _t254 + _t279 + _t289 + _t333 + _t336 + _t351;
		const Real _t353 = (0.25)*_b[38];
		const Real _t354 = (0.0625)*_b[44];
		const Real _t355 = _t25 + _t286 + _t335 + _t353 + _t354;
		const Real _t356 = (0.1875)*_b[36];
		const Real _t357 = _t116 + _t91;
		const Real _t358 = _t222 + _t252;
		const Real _t359 = _t223 + _t226 + _t356 + _t357 + _t358;
		const Real _t360 = (0.09375)*_b[36] + _t111;
		const Real _t361 = _t124 + _t232;
		const Real _t362 = (0.09375)*_b[37] + _t129 + _t229 + _t235;
		const Real _t363 = _t214 + _t233 + _t234 + _t258 + _t360 + _t361 + _t362;
		const Real _t364 = _t142 + _t143;
		const Real _t365 = _t151 + _t364;
		const Real _t366 = (0.046875)*_b[36] + (0.046875)*_b[38] + _t241 + _t242 + _t243 + _t244 + _t249 + _t262 + _t362 + _t365;
		const Real _t367 = (0.25)*_b[30];
		const Real _t368 = _t367 + _t87;
		const Real _t369 = _t205 + _t99;
		const Real _t370 = (0.125)*_b[39];
		const Real _t371 = _t252 + _t370;
		const Real _t372 = _t182 + _t368 + _t369 + _t371;
		const Real _t373 = (0.0625)*_b[19];
		const Real _t374 = _t373 + _t94;
		const Real _t375 = (0.125)*_b[31];
		const Real _t376 = (0.0625)*_b[40];
		const Real _t377 = _t375 + _t376;
		const Real _t378 = _t374 + _t377;
		const Real _t379 = _t104 + _t213;
		const Real _t380 = _t162 + _t193 + _t222 + _t258 + _t378 + _t379;
		const Real _t381 = (0.03125)*_b[18] + (0.03125)*_b[20] + _t373;
		const Real _t382 = _t233 + _t337 + _t377 + _t381;
		const Real _t383 = _t168 + _t317;
		const Real _t384 = _t113 + _t212 + _t217 + _t219 + _t383;
		const Real _t385 = _t202 + _t262 + _t382 + _t384;
		const Real _t386 = (0.1875)*_b[33];
		const Real _t387 = (0.1875)*_b[39] + _t252 + _t386;
		const Real _t388 = _t121 + _t224 + _t387;
		const Real _t389 = (0.09375)*_b[39];
		const Real _t390 = (0.09375)*_b[34];
		const Real _t391 = (0.09375)*_b[40] + _t390;
		const Real _t392 = _t137 + _t217 + _t49;
		const Real _t393 = (0.09375)*_b[33];
		const Real _t394 = _t237 + _t393;
		const Real _t395 = _t258 + _t389 + _t391 + _t392 + _t394;
		const Real _t396 = _t136 + _t391;
		const Real _t397 = (0.046875)*_b[33] + (0.046875)*_b[35];
		const Real _t398 = (0.046875)*_b[39] + (0.046875)*_b[41] + _t150 + _t245 + _t262 + _t396 + _t397;
		const Real _t399 = (0.25)*_b[24];
		const Real _t400 = (0.375)*_b[33];
		const Real _t401 = (0.25)*_b[39];
		const Real _t402 = _t252 + _t399 + _t400 + _t401 + _t54;
		const Real _t403 = (0.125)*_b[24];
		const Real _t404 = (0.125)*_b[25];
		const Real _t405 = _t403 + _t404;
		const Real _t406 = (0.1875)*_b[34];
		const Real _t407 = (0.125)*_b[40];
		const Real _t408 = _t406 + _t407;
		const Real _t409 = _t258 + _t370 + _t386 + _t405 + _t408 + _t64;
		const Real _t410 = _t222 + _t406;
		const Real _t411 = (0.09375)*_b[35];
		const Real _t412 = _t393 + _t411;
		const Real _t413 = _t118 + _t307;
		const Real _t414 = _t262 + _t344 + _t404 + _t407 + _t410 + _t412 + _t413 + _t72;
		const Real _t415 = (0.125)*_b[42];
		const Real _t416 = (0.1875)*_b[37];
		const Real _t417 = (0.0625)*_b[43];
		const Real _t418 = _t252 + _t417;
		const Real _t419 = (0.09375)*_b[38] + _t112;
		const Real _t420 = _t256 + _t350;
		const Real _t421 = _t417 + _t420;
		const Real _t422 = _t105 + _t216 + _t360 + _t416 + _t419 + _t421;
		const Real _t423 = _t370 + _t415;
		const Real _t424 = _t155 + _t251;
		const Real _t425 = _t158 + _t162;
		const Real _t426 = _t164 + _t417;
		const Real _t427 = _t261 + _t383;
		const Real _t428 = _t174 + _t382 + _t421 + _t427;
		const Real _t429 = _t367 + _t401;
		const Real _t430 = (0.125)*_b[33];
		const Real _t431 = _t415 + _t430;
		const Real _t432 = _t375 + _t407;
		const Real _t433 = _t162 + _t432;
		const Real _t434 = (0.0625)*_b[34];
		const Real _t435 = _t101 + _t434;
		const Real _t436 = _t432 + _t435;
		const Real _t437 = _t190 + _t322;
		const Real _t438 = _t222 + _t344 + _t384 + _t421 + _t436 + _t437;
		const Real _t439 = (0.0625)*_b[25];
		const Real _t440 = _t118 + _t439;
		const Real _t441 = (0.1875)*_b[40] + _t406 + _t417;
		const Real _t442 = (0.09375)*_b[41];
		const Real _t443 = _t439 + _t441;
		const Real _t444 = _t131 + _t300;
		const Real _t445 = _t389 + _t412 + _t420 + _t442 + _t443 + _t444;
		const Real _t446 = (0.5)*_b[36];
		const Real _t447 = (0.25)*_b[42];
		const Real _t448 = (0.25)*_b[37];
		const Real _t449 = (0.125)*_b[43];
		const Real _t450 = _t156 + _t448 + _t449;
		const Real _t451 = _t159 + _t336;
		const Real _t452 = _t354 + _t451;
		const Real _t453 = _t158 + _t205 + _t252 + _t450 + _t452;
		const Real _t454 = _t251 + _t447;
		const Real _t455 = _t344 + _t354;
		const Real _t456 = _t432 + _t449;
		const Real _t457 = _t358 + _t427 + _t455 + _t456;
		const Real _t458 = (0.25)*_b[33];
		const Real _t459 = (0.5)*_b[39];
		const Real _t460 = (0.25)*_b[40];
		const Real _t461 = (0.125)*_b[34];
		const Real _t462 = _t449 + _t460 + _t461;
		const Real _t463 = (0.125)*_b[41];
		const Real _t464 = _t180 + _t327;
		const Real _t465 = _t354 + _t371 + _t462 + _t463 + _t464;
		const Real _t466 = (0.5)*_b[42];
		const Real _t467 = (0.25)*_b[43];
		const Real _t468 = _t448 + _t467;
		const Real _t469 = (0.125)*_b[44];
		const Real _t470 = _t336 + _t469;
		const Real _t471 = _t205 + _t415 + _t468 + _t470;
		const Real _t472 = _t460 + _t467;
		const Real _t473 = _t463 + _t469;
		const Real _t474 = _t423 + _t472 + _t473;
		const Real _t475 = (0.5)*_b[43];
		const Real _t476 = (0.25)*_b[44];
		const Real _t477 = _t447 + _t475 + _t476;
		const Real _t478 = _t298 + _t337 + _t338 + _t342 + _t351 + _t362 + _t419;
		const Real _t479 = (0.1875)*_b[38];
		const Real _t480 = _t305 + _t92;
		const Real _t481 = _t345 + _t348 + _t455 + _t479 + _t480;
		const Real _t482 = _t373 + _t95;
		const Real _t483 = _t377 + _t482;
		const Real _t484 = _t294 + _t334;
		const Real _t485 = _t320 + _t325 + _t344 + _t351 + _t483 + _t484;
		const Real _t486 = (0.25)*_b[32];
		const Real _t487 = _t290 + _t486;
		const Real _t488 = _t329 + _t452 + _t463 + _t487;
		const Real _t489 = _t302 + _t51;
		const Real _t490 = _t219 + _t339 + _t411;
		const Real _t491 = _t351 + _t396 + _t442 + _t489 + _t490;
		const Real _t492 = (0.1875)*_b[35];
		const Real _t493 = (0.1875)*_b[41] + _t354 + _t492;
		const Real _t494 = _t310 + _t346 + _t493;
		const Real _t495 = (0.125)*_b[26];
		const Real _t496 = _t404 + _t495;
		const Real _t497 = _t281 + _t351 + _t408 + _t463 + _t492 + _t496;
		const Real _t498 = (0.25)*_b[26];
		const Real _t499 = (0.375)*_b[35];
		const Real _t500 = (0.25)*_b[41];
		const Real _t501 = _t284 + _t354 + _t498 + _t499 + _t500;
		const Real _t502 = _t354 + _t417;
		const Real _t503 = _t254 + _t320;
		const Real _t504 = _t313 + _t353;
		const Real _t505 = _t320 + _t327;
		const Real _t506 = (0.125)*_b[35];
		const Real _t507 = _t486 + _t500;
		const Real _t508 = (0.5)*_b[38];
		const Real _t509 = _t353 + _t476;
		const Real _t510 = (0.25)*_b[35];
		const Real _t511 = (0.5)*_b[41];
		const Real _t512 = (0.5)*_b[44];
		const Real _t513 = (0.125)*_b[12];
		const Real _t514 = (0.1875)*_b[10];
		const Real _t515 = (0.1875)*_b[9];
		const Real _t516 = (0.0625)*_b[13];
		const Real _t517 = _t516 + _t54;
		const Real _t518 = (0.09375)*_b[9] + _t175;
		const Real _t519 = (0.09375)*_b[11] + _t176;
		const Real _t520 = _t280 + _t516 + _t62;
		const Real _t521 = _t166 + _t48 + _t514 + _t518 + _t519 + _t520;
		const Real _t522 = (0.25)*_b[12];
		const Real _t523 = (0.5)*_b[9];
		const Real _t524 = (0.125)*_b[13];
		const Real _t525 = _t513 + _t524;
		const Real _t526 = _t19 + _t55;
		const Real _t527 = (0.25)*_b[10];
		const Real _t528 = _t22 + _t527;
		const Real _t529 = _t276 + _t32;
		const Real _t530 = _t284 + _t524 + _t54;
		const Real _t531 = _t29 + _t528 + _t529 + _t530;
		const Real _t532 = (0.5)*_b[12];
		const Real _t533 = _t522 + _t55;
		const Real _t534 = (0.25)*_b[13];
		const Real _t535 = _t527 + _t534;
		const Real _t536 = (0.125)*_b[14];
		const Real _t537 = _t513 + _t536;
		const Real _t538 = _t529 + _t535 + _t537;
		const Real _t539 = (0.5)*_b[13];
		const Real _t540 = (0.25)*_b[14];
		const Real _t541 = _t522 + _t539 + _t540;
		const Real _t542 = _t118 + _t54;
		const Real _t543 = _t116 + _t119 + _t122 + _t515 + _t542 + _t66;
		const Real _t544 = _t132 + _t64;
		const Real _t545 = _t124 + _t130;
		const Real _t546 = (0.09375)*_b[10] + _t126 + _t133 + _t134 + _t231;
		const Real _t547 = _t131 + _t46 + _t518 + _t544 + _t545 + _t546;
		const Real _t548 = (0.046875)*_b[11] + (0.046875)*_b[9] + _t129 + _t145 + _t146 + _t147 + _t148 + _t152 + _t247 + _t364 + _t546 + _t72;
		const Real _t549 = (0.25)*_b[21];
		const Real _t550 = _t403 + _t549 + _t87;
		const Real _t551 = (0.125)*_b[22];
		const Real _t552 = _t551 + _t99;
		const Real _t553 = _t552 + _t59;
		const Real _t554 = _t374 + _t440;
		const Real _t555 = _t102 + _t27;
		const Real _t556 = _t102 + _t439;
		const Real _t557 = _t104 + _t294 + _t551;
		const Real _t558 = _t381 + _t444 + _t557;
		const Real _t559 = _t110 + _t520 + _t556 + _t558 + _t70;
		const Real _t560 = _t399 + _t549;
		const Real _t561 = _t284 + _t307;
		const Real _t562 = _t404 + _t557;
		const Real _t563 = _t524 + _t542 + _t561 + _t562 + _t70;
		const Real _t564 = (0.5)*_b[24];
		const Real _t565 = (0.25)*_b[25];
		const Real _t566 = _t534 + _t565;
		const Real _t567 = _t403 + _t495;
		const Real _t568 = _t537 + _t566 + _t567;
		const Real _t569 = _t162 + _t32;
		const Real _t570 = _t182 + _t54 + _t550 + _t569;
		const Real _t571 = _t170 + _t42 + _t552;
		const Real _t572 = _t188 + _t192 + _t554 + _t571 + _t64;
		const Real _t573 = _t177 + _t40 + _t49 + _t51;
		const Real _t574 = _t188 + _t439;
		const Real _t575 = _t201 + _t558 + _t573 + _t574 + _t72;
		const Real _t576 = _t180 + _t434;
		const Real _t577 = _t434 + _t437;
		const Real _t578 = _t413 + _t520 + _t562 + _t573 + _t577;
		const Real _t579 = _t461 + _t565;
		const Real _t580 = _t464 + _t530 + _t567 + _t579;
		const Real _t581 = (0.1875)*_b[24] + _t386;
		const Real _t582 = _t120 + _t225 + _t54 + _t581;
		const Real _t583 = (0.09375)*_b[24] + _t233;
		const Real _t584 = _t217 + _t228;
		const Real _t585 = (0.09375)*_b[25] + _t135 + _t390;
		const Real _t586 = _t394 + _t49 + _t544 + _t583 + _t584 + _t585;
		const Real _t587 = _t228 + _t585;
		const Real _t588 = (0.046875)*_b[24] + (0.046875)*_b[26] + _t149 + _t246 + _t397 + _t587 + _t72;
		const Real _t589 = (0.1875)*_b[25] + _t376;
		const Real _t590 = _t406 + _t589;
		const Real _t591 = (0.09375)*_b[26] + _t337;
		const Real _t592 = _t412 + _t520 + _t583 + _t590 + _t591;
		const Real _t593 = (0.1875)*_b[11];
		const Real _t594 = _t284 + _t516;
		const Real _t595 = _t524 + _t536;
		const Real _t596 = _t269 + _t283;
		const Real _t597 = (0.5)*_b[11];
		const Real _t598 = _t283 + _t540;
		const Real _t599 = (0.5)*_b[14];
		const Real _t600 = _t129 + _t298;
		const Real _t601 = _t281 + _t303 + _t519 + _t546 + _t600;
		const Real _t602 = _t305 + _t308 + _t311 + _t561 + _t593 + _t67;
		const Real _t603 = _t297 + _t551;
		const Real _t604 = _t278 + _t603;
		const Real _t605 = _t307 + _t482;
		const Real _t606 = (0.25)*_b[23];
		const Real _t607 = _t290 + _t495 + _t606;
		const Real _t608 = _t498 + _t606;
		const Real _t609 = (0.5)*_b[26];
		const Real _t610 = _t275 + _t318 + _t603;
		const Real _t611 = _t281 + _t324 + _t574 + _t605 + _t610;
		const Real _t612 = _t276 + _t320;
		const Real _t613 = _t159 + _t284 + _t328 + _t607 + _t612;
		const Real _t614 = _t327 + _t434;
		const Real _t615 = _t281 + _t301 + _t490 + _t51 + _t587 + _t591;
		const Real _t616 = (0.1875)*_b[26] + _t492;
		const Real _t617 = _t284 + _t309 + _t347 + _t616;
		const Real _t618 = _t187 + _t228;
		const Real _t619 = _t136 + _t618;
		const Real _t620 = _t150 + _t199 + _t246 + _t381 + _t577 + _t619;
		const Real _t621 = _t180 + _t357 + _t66;
		const Real _t622 = _t184 + _t190;
		const Real _t623 = _t101 + _t186;
		const Real _t624 = _t184 + _t228 + _t623;
		const Real _t625 = _t200 + _t248 + _t365;
		const Real _t626 = _t113 + _t129 + _t174 + _t197 + _t198 + _t246 + _t624 + _t625;
		const Real _t627 = _t102 + _t140 + _t184 + _t185;
		const Real _t628 = _t110 + _t150 + _t177 + _t195 + _t196 + _t625 + _t627;
		const Real _t629 = _t219 + _t340;
		const Real _t630 = _t159 + _t310;
		const Real _t631 = _t112 + _t322 + _t341;
		const Real _t632 = _t480 + _t67;
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
				_t18 + _t2 + _t5,
				_t10 + _t20 + _t23,
				_t30,
				(0.375)*_b[3] + _t11 + _t31 + _t32,
				_t33 + _t34 + _t36 + _t39 + _t42,
				_t53,
				_t56,
				_t65,
				_t73,
				_t0 + _t74,
				_t7 + _t77,
				_t82,
				_t2 + _t75 + _t84,
				_t13 + _t21 + _t86 + _t90,
				_t98,
				_t100 + _t20 + _t84,
				_t103 + _t107 + _t12 + _t27 + _t36 + _t90,
				_t115,
				_t123,
				_t139,
				_t153,
				_t154 + _t2 + _t74,
				_t11 + _t155 + _t157 + _t77,
				_t161,
				_t13 + _t155 + _t163 + _t75,
				_t165 + _t171 + _t36 + _t86,
				_t179,
				_t183,
				_t194,
				_t203,
				(0.375)*_b[15] + _t11 + _t204 + _t205,
				_t206 + _t207 + _t210 + _t213 + _t36,
				_t221,
				_t227,
				_t240,
				_t250,
				_t253,
				_t259,
				_t263,
			},
			{
				_t4,
				_t1 + _t264,
				_b[2],
				_t17,
				_t265 + _t267,
				_t264 + _t268,
				_t30,
				_t14 + _t23 + _t267 + _t269,
				_t268 + _t270 + _t3,
				_t53,
				_t271 + _t272 + _t274 + _t275 + _t34,
				(0.375)*_b[5] + _t14 + _t276 + _t277,
				_t73,
				_t282,
				_t285,
				_t82,
				_t265 + _t287,
				_t264 + _t288,
				_t98,
				_t16 + _t21 + _t289 + _t291,
				_t286 + _t293 + _t3,
				_t115,
				_t103 + _t15 + _t272 + _t28 + _t291 + _t296,
				_t269 + _t293 + _t297 + _t79,
				_t153,
				_t304,
				_t312,
				_t161,
				_t14 + _t157 + _t287 + _t313,
				_t288 + _t3 + _t314,
				_t179,
				_t272 + _t289 + _t315 + _t319,
				_t16 + _t286 + _t313 + _t321,
				_t203,
				_t326,
				_t330,
				_t221,
				_t207 + _t272 + _t331 + _t333 + _t334,
				(0.375)*_b[17] + _t14 + _t335 + _t336,
				_t250,
				_t343,
				_t349,
				_t263,
				_t352,
				_t355,
			},
			{
				_t253,
				_t259,
				_t263,
				_t359,
				_t363,
				_t366,
				_t372,
				_t380,
				_t385,
				_t388,
				_t395,
				_t398,
				_t402,
				_t409,
				_t414,
				(0.375)*_b[36] + _t204 + _t415 + _t80,
				_t106 + _t210 + _t356 + _t416 + _t418,
				_t422,
				_t368 + _t423 + _t424,
				_t255 + _t358 + _t378 + _t425 + _t426,
				_t428,
				_t369 + _t429 + _t431,
				_t180 + _t370 + _t379 + _t418 + _t433 + _t435,
				_t438,
				(0.375)*_b[39] + _t400 + _t403 + _t415,
				_t387 + _t440 + _t441,
				_t445,
				_t154 + _t446 + _t447,
				_t415 + _t424 + _t450,
				_t453,
				_t429 + _t454,
				_t255 + _t423 + _t433 + _t449,
				_t457,
				_t447 + _t458 + _t459,
				_t401 + _t431 + _t462,
				_t465,
				_t446 + _t466,
				_t454 + _t468,
				_t471,
				_t459 + _t466,
				_t401 + _t447 + _t472,
				_t474,
				_b[42],
				_t466 + _t475,
				_t477,
			},
			{
				_t263,
				_t352,
				_t355,
				_t366,
				_t478,
				_t481,
				_t385,
				_t485,
				_t488,
				_t398,
				_t491,
				_t494,
				_t414,
				_t497,
				_t501,
				_t422,
				_t295 + _t333 + _t416 + _t479 + _t502,
				(0.375)*_b[38] + _t335 + _t469 + _t78,
				_t428,
				_t426 + _t451 + _t455 + _t483 + _t503,
				_t473 + _t487 + _t504,
				_t438,
				_t436 + _t463 + _t484 + _t502 + _t505,
				_t297 + _t470 + _t506 + _t507,
				_t445,
				_t307 + _t443 + _t493,
				(0.375)*_b[41] + _t469 + _t495 + _t499,
				_t453,
				_t450 + _t469 + _t504,
				_t314 + _t476 + _t508,
				_t457,
				_t456 + _t463 + _t470 + _t503,
				_t507 + _t509,
				_t465,
				_t462 + _t469 + _t500 + _t506,
				_t476 + _t510 + _t511,
				_t471,
				_t468 + _t509,
				_t508 + _t512,
				_t474,
				_t472 + _t476 + _t500,
				_t511 + _t512,
				_t477,
				_t475 + _t512,
				_b[44],
			},
			{
				_t56,
				_t65,
				_t73,
				(0.375)*_b[9] + _t12 + _t31 + _t513,
				_t167 + _t39 + _t514 + _t515 + _t517,
				_t521,
				_t18 + _t522 + _t523,
				_t525 + _t526 + _t528,
				_t531,
				_t523 + _t532,
				_t533 + _t535,
				_t538,
				_b[12],
				_t532 + _t539,
				_t541,
				_t543,
				_t547,
				_t548,
				_t513 + _t526 + _t550,
				_t517 + _t553 + _t554 + _t555,
				_t559,
				_t533 + _t560,
				_t405 + _t525 + _t553,
				_t563,
				_t532 + _t564,
				_t399 + _t522 + _t566,
				_t568,
				_t570,
				_t572,
				_t575,
				_t430 + _t513 + _t560 + _t569,
				_t405 + _t517 + _t571 + _t576,
				_t578,
				_t458 + _t522 + _t564,
				_t399 + _t430 + _t525 + _t579,
				_t580,
				_t582,
				_t586,
				_t588,
				(0.375)*_b[24] + _t370 + _t400 + _t513,
				_t410 + _t517 + _t581 + _t589,
				_t592,
				_t402,
				_t409,
				_t414,
			},
			{
				_t73,
				_t282,
				_t285,
				_t521,
				_t274 + _t316 + _t514 + _t593 + _t594,
				(0.375)*_b[11] + _t15 + _t277 + _t536,
				_t531,
				_t528 + _t595 + _t596,
				_t270 + _t540 + _t597,
				_t538,
				_t535 + _t598,
				_t597 + _t599,
				_t541,
				_t539 + _t599,
				_b[14],
				_t548,
				_t601,
				_t602,
				_t559,
				_t28 + _t556 + _t594 + _t604 + _t605,
				_t536 + _t596 + _t607,
				_t563,
				_t496 + _t595 + _t604,
				_t598 + _t608,
				_t568,
				_t498 + _t540 + _t566,
				_t599 + _t609,
				_t575,
				_t611,
				_t613,
				_t578,
				_t496 + _t594 + _t610 + _t614,
				_t506 + _t536 + _t608 + _t612,
				_t580,
				_t498 + _t506 + _t579 + _t595,
				_t510 + _t540 + _t609,
				_t588,
				_t615,
				_t617,
				_t592,
				_t344 + _t590 + _t594 + _t616,
				(0.375)*_b[26] + _t463 + _t499 + _t536,
				_t414,
				_t497,
				_t501,
			},
			{
				_t402,
				_t409,
				_t414,
				_t388,
				_t395,
				_t398,
				_t372,
				_t380,
				_t385,
				_t359,
				_t363,
				_t366,
				_t253,
				_t259,
				_t263,
				_t582,
				_t586,
				_t588,
				_t121 + _t181 + _t225 + _t430 + _t87,
				_t191 + _t238 + _t374 + _t392 + _t576 + _t618,
				_t620,
				_t155 + _t225 + _t27 + _t621 + _t99,
				_t104 + _t108 + _t165 + _t239 + _t545 + _t584 + _t622 + _t623,
				_t626,
				_t227,
				_t240,
				_t250,
				_t570,
				_t572,
				_t575,
				_t121 + _t19 + _t425 + _t621,
				_t138 + _t170 + _t172 + _t185 + _t361 + _t49 + _t555 + _t622,
				_t628,
				_t183,
				_t194,
				_t203,
				_t543,
				_t547,
				_t548,
				_t123,
				_t139,
				_t153,
				_t56,
				_t65,
				_t73,
			},
			{
				_t414,
				_t497,
				_t501,
				_t398,
				_t491,
				_t494,
				_t385,
				_t485,
				_t488,
				_t366,
				_t478,
				_t481,
				_t263,
				_t352,
				_t355,
				_t588,
				_t615,
				_t617,
				_t620,
				_t323 + _t482 + _t489 + _t614 + _t619 + _t629,
				_t28 + _t290 + _t347 + _t506 + _t630,
				_t626,
				_t109 + _t294 + _t315 + _t600 + _t624 + _t629 + _t631,
				_t313 + _t329 + _t347 + _t632,
				_t250,
				_t343,
				_t349,
				_t575,
				_t611,
				_t613,
				_t628,
				_t173 + _t28 + _t298 + _t318 + _t489 + _t627 + _t631,
				_t269 + _t505 + _t630 + _t632,
				_t203,
				_t326,
				_t330,
				_t548,
				_t601,
				_t602,
				_t153,
				_t304,
				_t312,
				_t73,
				_t282,
				_t285,
			},
		}};
		#pragma endregion
	}
}
