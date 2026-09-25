#include "JacEval_P2Tet.hpp"
namespace miso {
	Real JacEval_P2Tet(const RealVector<10> &rx, const RealVector<10> &ry, const RealVector<10> &rz, const RealVector<10> &ux, const RealVector<10> &uy, const RealVector<10> &uz, const Real x0, const Real x1, const Real x2) {
		#pragma region Temporaries
		const Real _t0 = 4*rx[1];
		const Real _t1 = 4*ux[1];
		const Real _t2 = 8*x2;
		const Real _t3 = 4*x2;
		const Real _t4 = 4*x1;
		const Real _t5 = 4*x0;
		const Real _t6 = -3*rx[0];
		const Real _t7 = -3*ux[0];
		const Real _t8 = 4*rx[6];
		const Real _t9 = 4*ux[6];
		const Real _t10 = 4*rx[0];
		const Real _t11 = _t10*x0;
		const Real _t12 = _t10*x1;
		const Real _t13 = _t10*x2;
		const Real _t14 = _t5*ux[0];
		const Real _t15 = _t4*ux[0];
		const Real _t16 = _t3*ux[0];
		const Real _t17 = _t11 + _t12 + _t13 + _t14 + _t15 + _t16 + _t6 + _t7 - _t8*x0 - _t9*x0;
		const Real _t18 = 4*rx[3];
		const Real _t19 = 4*ux[3];
		const Real _t20 = -_t18*x1 - _t19*x1;
		const Real _t21 = -_t0*x0 - _t0*x1 + _t0 - _t1*x0 - _t1*x1 + _t1 + _t17 - _t2*rx[1] - _t2*ux[1] + _t20 + _t3*rx[2] + _t3*ux[2] + _t4*rx[4] + _t4*ux[4] + _t5*rx[7] + _t5*ux[7] - rx[2] - ux[2];
		const Real _t22 = -3*ry[0];
		const Real _t23 = -3*uy[0];
		const Real _t24 = _t5*ry[0];
		const Real _t25 = _t4*ry[0];
		const Real _t26 = _t3*ry[0];
		const Real _t27 = _t5*uy[0];
		const Real _t28 = _t4*uy[0];
		const Real _t29 = _t3*uy[0];
		const Real _t30 = 4*ry[3];
		const Real _t31 = 4*uy[3];
		const Real _t32 = 8*x1;
		const Real _t33 = 4*ry[1];
		const Real _t34 = 4*uy[1];
		const Real _t35 = -_t33*x2 - _t34*x2;
		const Real _t36 = 4*ry[6];
		const Real _t37 = 4*uy[6];
		const Real _t38 = -_t36*x0 - _t37*x0;
		const Real _t39 = _t22 + _t23 + _t24 + _t25 + _t26 + _t27 + _t28 + _t29 + _t3*ry[4] + _t3*uy[4] - _t30*x0 - _t30*x2 + _t30 - _t31*x0 - _t31*x2 + _t31 - _t32*ry[3] - _t32*uy[3] + _t35 + _t38 + _t4*ry[5] + _t4*uy[5] + _t5*ry[8] + _t5*uy[8] - ry[5] - uy[5];
		const Real _t40 = 4*rz[6];
		const Real _t41 = 4*uz[6];
		const Real _t42 = 8*x0;
		const Real _t43 = -3*rz[0];
		const Real _t44 = -3*uz[0];
		const Real _t45 = 4*rz[1];
		const Real _t46 = 4*uz[1];
		const Real _t47 = _t5*rz[0];
		const Real _t48 = _t4*rz[0];
		const Real _t49 = _t3*rz[0];
		const Real _t50 = _t5*uz[0];
		const Real _t51 = _t4*uz[0];
		const Real _t52 = _t3*uz[0];
		const Real _t53 = _t43 + _t44 - _t45*x2 - _t46*x2 + _t47 + _t48 + _t49 + _t50 + _t51 + _t52;
		const Real _t54 = 4*rz[3];
		const Real _t55 = 4*uz[3];
		const Real _t56 = -_t54*x1 - _t55*x1;
		const Real _t57 = _t3*rz[7] + _t3*uz[7] + _t4*rz[8] + _t4*uz[8] - _t40*x1 - _t40*x2 + _t40 - _t41*x1 - _t41*x2 + _t41 - _t42*rz[6] - _t42*uz[6] + _t5*rz[9] + _t5*uz[9] + _t53 + _t56 - rz[9] - uz[9];
		const Real _t58 = -_t0*x2 - _t1*x2;
		const Real _t59 = _t17 - _t18*x0 - _t18*x2 + _t18 - _t19*x0 - _t19*x2 + _t19 + _t3*rx[4] + _t3*ux[4] - _t32*rx[3] - _t32*ux[3] + _t4*rx[5] + _t4*ux[5] + _t5*rx[8] + _t5*ux[8] + _t58 - rx[5] - ux[5];
		const Real _t60 = _t22 + _t23 + _t24 + _t25 + _t26 + _t27 + _t28 + _t29 - _t30*x1 - _t31*x1;
		const Real _t61 = _t3*ry[7] + _t3*uy[7] + _t35 - _t36*x1 - _t36*x2 + _t36 - _t37*x1 - _t37*x2 + _t37 + _t4*ry[8] + _t4*uy[8] - _t42*ry[6] - _t42*uy[6] + _t5*ry[9] + _t5*uy[9] + _t60 - ry[9] - uy[9];
		const Real _t62 = -_t40*x0 - _t41*x0;
		const Real _t63 = -_t2*rz[1] - _t2*uz[1] + _t3*rz[2] + _t3*uz[2] + _t4*rz[4] + _t4*uz[4] + _t43 + _t44 - _t45*x0 - _t45*x1 + _t45 - _t46*x0 - _t46*x1 + _t46 + _t47 + _t48 + _t49 + _t5*rz[7] + _t5*uz[7] + _t50 + _t51 + _t52 + _t56 + _t62 - rz[2] - uz[2];
		const Real _t64 = _t11 + _t12 + _t13 + _t14 + _t15 + _t16 + _t20 + _t3*rx[7] + _t3*ux[7] + _t4*rx[8] + _t4*ux[8] - _t42*rx[6] - _t42*ux[6] + _t5*rx[9] + _t5*ux[9] + _t58 + _t6 + _t7 - _t8*x1 - _t8*x2 + _t8 - _t9*x1 - _t9*x2 + _t9 - rx[9] - ux[9];
		const Real _t65 = -_t2*ry[1] - _t2*uy[1] + _t3*ry[2] + _t3*uy[2] - _t33*x0 - _t33*x1 + _t33 - _t34*x0 - _t34*x1 + _t34 + _t38 + _t4*ry[4] + _t4*uy[4] + _t5*ry[7] + _t5*uy[7] + _t60 - ry[2] - uy[2];
		const Real _t66 = _t3*rz[4] + _t3*uz[4] - _t32*rz[3] - _t32*uz[3] + _t4*rz[5] + _t4*uz[5] + _t5*rz[8] + _t5*uz[8] + _t53 - _t54*x0 - _t54*x2 + _t54 - _t55*x0 - _t55*x2 + _t55 + _t62 - rz[5] - uz[5];
		#pragma endregion
		return -_t21*_t39*_t57 + _t21*_t61*_t66 + _t39*_t63*_t64 + _t57*_t59*_t65 - _t59*_t61*_t63 - _t64*_t65*_t66;
	}
}
