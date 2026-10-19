#include <Eigen/Dense>
#include <cassert>

namespace polyfem {
namespace autogen {
namespace {
void q_1_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = -1.0*(y - 1)*(z - 1);}
{val[1] = -1.0*(x - 1)*(z - 1);}
{val[2] = -1.0*(x - 1)*(y - 1);}
}

void q_1_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = 1.0*(y - 1)*(z - 1);}
{val[1] = 1.0*x*(z - 1);}
{val[2] = 1.0*x*(y - 1);}
}

void q_1_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = -1.0*y*(z - 1);}
{val[1] = -1.0*x*(z - 1);}
{val[2] = -1.0*x*y;}
}

void q_1_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 1.0*y*(z - 1);}
{val[1] = 1.0*(x - 1)*(z - 1);}
{val[2] = 1.0*y*(x - 1);}
}

void q_1_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{val[0] = 1.0*z*(y - 1);}
{val[1] = 1.0*z*(x - 1);}
{val[2] = 1.0*(x - 1)*(y - 1);}
}

void q_1_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{val[0] = -1.0*z*(y - 1);}
{val[1] = -1.0*x*z;}
{val[2] = -1.0*x*(y - 1);}
}

void q_1_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{val[0] = 1.0*y*z;}
{val[1] = 1.0*x*z;}
{val[2] = 1.0*x*y;}
}

void q_1_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = -1.0*y*z;}
{val[1] = -1.0*z*(x - 1);}
{val[2] = -1.0*y*(x - 1);}
}



}

void q_1_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 3);
double gradient[3];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_1_basis_grad_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_1_basis_grad_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_1_basis_grad_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_1_basis_grad_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_1_basis_grad_value_3d_single_4(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_1_basis_grad_value_3d_single_5(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_1_basis_grad_value_3d_single_6(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_1_basis_grad_value_3d_single_7(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	default: assert(false);
}
}

}}