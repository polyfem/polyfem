#include <Eigen/Dense>
#include <cassert>

namespace polyfem {
namespace autogen {
namespace {
void q_2_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = (4.0*x - 3.0)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[1] = (x - 1)*(2.0*x - 1.0)*(4.0*y - 3.0)*(z - 1)*(2.0*z - 1.0);}
{val[2] = (x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(4.0*z - 3.0);}
}

void q_2_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = (4.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[1] = x*(2.0*x - 1.0)*(4.0*y - 3.0)*(z - 1)*(2.0*z - 1.0);}
{val[2] = x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(4.0*z - 3.0);}
}

void q_2_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = y*(4.0*x - 1.0)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[1] = x*(2.0*x - 1.0)*(4.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[2] = x*y*(2.0*x - 1.0)*(2.0*y - 1.0)*(4.0*z - 3.0);}
}

void q_2_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = y*(4.0*x - 3.0)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[1] = (x - 1)*(2.0*x - 1.0)*(4.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[2] = y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(4.0*z - 3.0);}
}

void q_2_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{val[0] = z*(4.0*x - 3.0)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);}
{val[1] = z*(x - 1)*(2.0*x - 1.0)*(4.0*y - 3.0)*(2.0*z - 1.0);}
{val[2] = (x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(4.0*z - 1.0);}
}

void q_2_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{val[0] = z*(4.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);}
{val[1] = x*z*(2.0*x - 1.0)*(4.0*y - 3.0)*(2.0*z - 1.0);}
{val[2] = x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(4.0*z - 1.0);}
}

void q_2_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{val[0] = y*z*(4.0*x - 1.0)*(2.0*y - 1.0)*(2.0*z - 1.0);}
{val[1] = x*z*(2.0*x - 1.0)*(4.0*y - 1.0)*(2.0*z - 1.0);}
{val[2] = x*y*(2.0*x - 1.0)*(2.0*y - 1.0)*(4.0*z - 1.0);}
}

void q_2_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = y*z*(4.0*x - 3.0)*(2.0*y - 1.0)*(2.0*z - 1.0);}
{val[1] = z*(x - 1)*(2.0*x - 1.0)*(4.0*y - 1.0)*(2.0*z - 1.0);}
{val[2] = y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(4.0*z - 1.0);}
}

void q_2_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{val[0] = -4.0*(2*x - 1)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[1] = -x*(x - 1)*(16.0*y - 12.0)*(z - 1)*(2.0*z - 1.0);}
{val[2] = -x*(x - 1)*(y - 1)*(2.0*y - 1.0)*(16.0*z - 12.0);}
}

void q_2_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{val[0] = -y*(16.0*x - 4.0)*(y - 1)*(z - 1)*(2.0*z - 1.0);}
{val[1] = -4.0*x*(2.0*x - 1.0)*(2*y - 1)*(z - 1)*(2.0*z - 1.0);}
{val[2] = -x*y*(2.0*x - 1.0)*(y - 1)*(16.0*z - 12.0);}
}

void q_2_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
{val[0] = -4.0*y*(2*x - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[1] = -x*(x - 1)*(16.0*y - 4.0)*(z - 1)*(2.0*z - 1.0);}
{val[2] = -x*y*(x - 1)*(2.0*y - 1.0)*(16.0*z - 12.0);}
}

void q_2_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
{val[0] = -y*(16.0*x - 12.0)*(y - 1)*(z - 1)*(2.0*z - 1.0);}
{val[1] = -4.0*(x - 1)*(2.0*x - 1.0)*(2*y - 1)*(z - 1)*(2.0*z - 1.0);}
{val[2] = -y*(x - 1)*(2.0*x - 1.0)*(y - 1)*(16.0*z - 12.0);}
}

void q_2_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
{val[0] = -z*(16.0*x - 12.0)*(y - 1)*(2.0*y - 1.0)*(z - 1);}
{val[1] = -z*(x - 1)*(2.0*x - 1.0)*(16.0*y - 12.0)*(z - 1);}
{val[2] = -4.0*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(2*z - 1);}
}

void q_2_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
{val[0] = -z*(16.0*x - 4.0)*(y - 1)*(2.0*y - 1.0)*(z - 1);}
{val[1] = -x*z*(2.0*x - 1.0)*(16.0*y - 12.0)*(z - 1);}
{val[2] = -4.0*x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(2*z - 1);}
}

void q_2_basis_grad_value_3d_single_14(double x, double y, double z, double *val) {
{val[0] = -y*z*(16.0*x - 4.0)*(2.0*y - 1.0)*(z - 1);}
{val[1] = -x*z*(2.0*x - 1.0)*(16.0*y - 4.0)*(z - 1);}
{val[2] = -4.0*x*y*(2.0*x - 1.0)*(2.0*y - 1.0)*(2*z - 1);}
}

void q_2_basis_grad_value_3d_single_15(double x, double y, double z, double *val) {
{val[0] = -y*z*(16.0*x - 12.0)*(2.0*y - 1.0)*(z - 1);}
{val[1] = -z*(x - 1)*(2.0*x - 1.0)*(16.0*y - 4.0)*(z - 1);}
{val[2] = -4.0*y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(2*z - 1);}
}

void q_2_basis_grad_value_3d_single_16(double x, double y, double z, double *val) {
{val[0] = -4.0*z*(2*x - 1)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);}
{val[1] = -x*z*(x - 1)*(16.0*y - 12.0)*(2.0*z - 1.0);}
{val[2] = -x*(x - 1)*(y - 1)*(2.0*y - 1.0)*(16.0*z - 4.0);}
}

void q_2_basis_grad_value_3d_single_17(double x, double y, double z, double *val) {
{val[0] = -y*z*(16.0*x - 4.0)*(y - 1)*(2.0*z - 1.0);}
{val[1] = -4.0*x*z*(2.0*x - 1.0)*(2*y - 1)*(2.0*z - 1.0);}
{val[2] = -x*y*(2.0*x - 1.0)*(y - 1)*(16.0*z - 4.0);}
}

void q_2_basis_grad_value_3d_single_18(double x, double y, double z, double *val) {
{val[0] = -4.0*y*z*(2*x - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);}
{val[1] = -x*z*(x - 1)*(16.0*y - 4.0)*(2.0*z - 1.0);}
{val[2] = -x*y*(x - 1)*(2.0*y - 1.0)*(16.0*z - 4.0);}
}

void q_2_basis_grad_value_3d_single_19(double x, double y, double z, double *val) {
{val[0] = -y*z*(16.0*x - 12.0)*(y - 1)*(2.0*z - 1.0);}
{val[1] = -4.0*z*(x - 1)*(2.0*x - 1.0)*(2*y - 1)*(2.0*z - 1.0);}
{val[2] = -y*(x - 1)*(2.0*x - 1.0)*(y - 1)*(16.0*z - 4.0);}
}

void q_2_basis_grad_value_3d_single_20(double x, double y, double z, double *val) {
{val[0] = y*z*(64.0*x - 48.0)*(y - 1)*(z - 1);}
{val[1] = 16.0*z*(x - 1)*(2.0*x - 1.0)*(2*y - 1)*(z - 1);}
{val[2] = 16.0*y*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2*z - 1);}
}

void q_2_basis_grad_value_3d_single_21(double x, double y, double z, double *val) {
{val[0] = y*z*(64.0*x - 16.0)*(y - 1)*(z - 1);}
{val[1] = 16.0*x*z*(2.0*x - 1.0)*(2*y - 1)*(z - 1);}
{val[2] = 16.0*x*y*(2.0*x - 1.0)*(y - 1)*(2*z - 1);}
}

void q_2_basis_grad_value_3d_single_22(double x, double y, double z, double *val) {
{val[0] = 16.0*z*(2*x - 1)*(y - 1)*(2.0*y - 1.0)*(z - 1);}
{val[1] = x*z*(x - 1)*(64.0*y - 48.0)*(z - 1);}
{val[2] = 16.0*x*(x - 1)*(y - 1)*(2.0*y - 1.0)*(2*z - 1);}
}

void q_2_basis_grad_value_3d_single_23(double x, double y, double z, double *val) {
{val[0] = 16.0*y*z*(2*x - 1)*(2.0*y - 1.0)*(z - 1);}
{val[1] = x*z*(x - 1)*(64.0*y - 16.0)*(z - 1);}
{val[2] = 16.0*x*y*(x - 1)*(2.0*y - 1.0)*(2*z - 1);}
}

void q_2_basis_grad_value_3d_single_24(double x, double y, double z, double *val) {
{val[0] = 16.0*y*(2*x - 1)*(y - 1)*(z - 1)*(2.0*z - 1.0);}
{val[1] = 16.0*x*(x - 1)*(2*y - 1)*(z - 1)*(2.0*z - 1.0);}
{val[2] = x*y*(x - 1)*(y - 1)*(64.0*z - 48.0);}
}

void q_2_basis_grad_value_3d_single_25(double x, double y, double z, double *val) {
{val[0] = 16.0*y*z*(2*x - 1)*(y - 1)*(2.0*z - 1.0);}
{val[1] = 16.0*x*z*(x - 1)*(2*y - 1)*(2.0*z - 1.0);}
{val[2] = x*y*(x - 1)*(y - 1)*(64.0*z - 16.0);}
}

void q_2_basis_grad_value_3d_single_26(double x, double y, double z, double *val) {
{val[0] = -64.0*y*z*(2*x - 1)*(y - 1)*(z - 1);}
{val[1] = -64.0*x*z*(x - 1)*(2*y - 1)*(z - 1);}
{val[2] = -64.0*x*y*(x - 1)*(y - 1)*(2*z - 1);}
}



}

void q_2_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 3);
double gradient[3];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_4(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_5(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_6(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_7(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_8(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_9(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 10:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_10(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 11:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_11(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 12:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_12(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 13:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_13(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 14:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_14(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 15:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_15(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 16:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_16(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 17:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_17(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 18:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_18(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 19:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_19(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 20:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_20(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 21:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_21(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 22:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_22(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 23:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_23(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 24:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_24(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 25:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_25(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 26:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_3d_single_26(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	default: assert(false);
}
}

}}