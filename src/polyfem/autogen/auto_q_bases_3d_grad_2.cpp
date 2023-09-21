#include <Eigen/Dense>
 namespace polyfem {
namespace autogen {void q_2_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = (4.0*x - 3.0)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = (x - 1)*(2.0*x - 1.0)*(4.0*y - 3.0)*(z - 1)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = (x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(4.0*z - 3.0);val.col(2) = result_0; }} break;
	case 1: {{result_0 = (4.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = x*(2.0*x - 1.0)*(4.0*y - 3.0)*(z - 1)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(4.0*z - 3.0);val.col(2) = result_0; }} break;
	case 2: {{result_0 = y*(4.0*x - 1.0)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = x*(2.0*x - 1.0)*(4.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = x*y*(2.0*x - 1.0)*(2.0*y - 1.0)*(4.0*z - 3.0);val.col(2) = result_0; }} break;
	case 3: {{result_0 = y*(4.0*x - 3.0)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = (x - 1)*(2.0*x - 1.0)*(4.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(4.0*z - 3.0);val.col(2) = result_0; }} break;
	case 4: {{result_0 = z*(4.0*x - 3.0)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = z*(x - 1)*(2.0*x - 1.0)*(4.0*y - 3.0)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = (x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(4.0*z - 1.0);val.col(2) = result_0; }} break;
	case 5: {{result_0 = z*(4.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = x*z*(2.0*x - 1.0)*(4.0*y - 3.0)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(4.0*z - 1.0);val.col(2) = result_0; }} break;
	case 6: {{result_0 = y*z*(4.0*x - 1.0)*(2.0*y - 1.0)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = x*z*(2.0*x - 1.0)*(4.0*y - 1.0)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = x*y*(2.0*x - 1.0)*(2.0*y - 1.0)*(4.0*z - 1.0);val.col(2) = result_0; }} break;
	case 7: {{result_0 = y*z*(4.0*x - 3.0)*(2.0*y - 1.0)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = z*(x - 1)*(2.0*x - 1.0)*(4.0*y - 1.0)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(4.0*z - 1.0);val.col(2) = result_0; }} break;
	case 8: {{result_0 = -4.0*(2*x - 1)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = -x*(x - 1)*(16.0*y - 12.0)*(z - 1)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = -x*(x - 1)*(y - 1)*(2.0*y - 1.0)*(16.0*z - 12.0);val.col(2) = result_0; }} break;
	case 9: {{result_0 = -y*(16.0*x - 4.0)*(y - 1)*(z - 1)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = -4.0*x*(2.0*x - 1.0)*(2*y - 1)*(z - 1)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = -x*y*(2.0*x - 1.0)*(y - 1)*(16.0*z - 12.0);val.col(2) = result_0; }} break;
	case 10: {{result_0 = -4.0*y*(2*x - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = -x*(x - 1)*(16.0*y - 4.0)*(z - 1)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = -x*y*(x - 1)*(2.0*y - 1.0)*(16.0*z - 12.0);val.col(2) = result_0; }} break;
	case 11: {{result_0 = -y*(16.0*x - 12.0)*(y - 1)*(z - 1)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = -4.0*(x - 1)*(2.0*x - 1.0)*(2*y - 1)*(z - 1)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = -y*(x - 1)*(2.0*x - 1.0)*(y - 1)*(16.0*z - 12.0);val.col(2) = result_0; }} break;
	case 12: {{result_0 = -z*(16.0*x - 12.0)*(y - 1)*(2.0*y - 1.0)*(z - 1);val.col(0) = result_0; }{result_0 = -z*(x - 1)*(2.0*x - 1.0)*(16.0*y - 12.0)*(z - 1);val.col(1) = result_0; }{result_0 = -4.0*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(2*z - 1);val.col(2) = result_0; }} break;
	case 13: {{result_0 = -z*(16.0*x - 4.0)*(y - 1)*(2.0*y - 1.0)*(z - 1);val.col(0) = result_0; }{result_0 = -x*z*(2.0*x - 1.0)*(16.0*y - 12.0)*(z - 1);val.col(1) = result_0; }{result_0 = -4.0*x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(2*z - 1);val.col(2) = result_0; }} break;
	case 14: {{result_0 = -y*z*(16.0*x - 4.0)*(2.0*y - 1.0)*(z - 1);val.col(0) = result_0; }{result_0 = -x*z*(2.0*x - 1.0)*(16.0*y - 4.0)*(z - 1);val.col(1) = result_0; }{result_0 = -4.0*x*y*(2.0*x - 1.0)*(2.0*y - 1.0)*(2*z - 1);val.col(2) = result_0; }} break;
	case 15: {{result_0 = -y*z*(16.0*x - 12.0)*(2.0*y - 1.0)*(z - 1);val.col(0) = result_0; }{result_0 = -z*(x - 1)*(2.0*x - 1.0)*(16.0*y - 4.0)*(z - 1);val.col(1) = result_0; }{result_0 = -4.0*y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(2*z - 1);val.col(2) = result_0; }} break;
	case 16: {{result_0 = -4.0*z*(2*x - 1)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = -x*z*(x - 1)*(16.0*y - 12.0)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = -x*(x - 1)*(y - 1)*(2.0*y - 1.0)*(16.0*z - 4.0);val.col(2) = result_0; }} break;
	case 17: {{result_0 = -y*z*(16.0*x - 4.0)*(y - 1)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = -4.0*x*z*(2.0*x - 1.0)*(2*y - 1)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = -x*y*(2.0*x - 1.0)*(y - 1)*(16.0*z - 4.0);val.col(2) = result_0; }} break;
	case 18: {{result_0 = -4.0*y*z*(2*x - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = -x*z*(x - 1)*(16.0*y - 4.0)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = -x*y*(x - 1)*(2.0*y - 1.0)*(16.0*z - 4.0);val.col(2) = result_0; }} break;
	case 19: {{result_0 = -y*z*(16.0*x - 12.0)*(y - 1)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = -4.0*z*(x - 1)*(2.0*x - 1.0)*(2*y - 1)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = -y*(x - 1)*(2.0*x - 1.0)*(y - 1)*(16.0*z - 4.0);val.col(2) = result_0; }} break;
	case 20: {{result_0 = y*z*(64.0*x - 48.0)*(y - 1)*(z - 1);val.col(0) = result_0; }{result_0 = 16.0*z*(x - 1)*(2.0*x - 1.0)*(2*y - 1)*(z - 1);val.col(1) = result_0; }{result_0 = 16.0*y*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2*z - 1);val.col(2) = result_0; }} break;
	case 21: {{result_0 = y*z*(64.0*x - 16.0)*(y - 1)*(z - 1);val.col(0) = result_0; }{result_0 = 16.0*x*z*(2.0*x - 1.0)*(2*y - 1)*(z - 1);val.col(1) = result_0; }{result_0 = 16.0*x*y*(2.0*x - 1.0)*(y - 1)*(2*z - 1);val.col(2) = result_0; }} break;
	case 22: {{result_0 = 16.0*z*(2*x - 1)*(y - 1)*(2.0*y - 1.0)*(z - 1);val.col(0) = result_0; }{result_0 = x*z*(x - 1)*(64.0*y - 48.0)*(z - 1);val.col(1) = result_0; }{result_0 = 16.0*x*(x - 1)*(y - 1)*(2.0*y - 1.0)*(2*z - 1);val.col(2) = result_0; }} break;
	case 23: {{result_0 = 16.0*y*z*(2*x - 1)*(2.0*y - 1.0)*(z - 1);val.col(0) = result_0; }{result_0 = x*z*(x - 1)*(64.0*y - 16.0)*(z - 1);val.col(1) = result_0; }{result_0 = 16.0*x*y*(x - 1)*(2.0*y - 1.0)*(2*z - 1);val.col(2) = result_0; }} break;
	case 24: {{result_0 = 16.0*y*(2*x - 1)*(y - 1)*(z - 1)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = 16.0*x*(x - 1)*(2*y - 1)*(z - 1)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = x*y*(x - 1)*(y - 1)*(64.0*z - 48.0);val.col(2) = result_0; }} break;
	case 25: {{result_0 = 16.0*y*z*(2*x - 1)*(y - 1)*(2.0*z - 1.0);val.col(0) = result_0; }{result_0 = 16.0*x*z*(x - 1)*(2*y - 1)*(2.0*z - 1.0);val.col(1) = result_0; }{result_0 = x*y*(x - 1)*(y - 1)*(64.0*z - 16.0);val.col(2) = result_0; }} break;
	case 26: {{result_0 = -64.0*y*z*(2*x - 1)*(y - 1)*(z - 1);val.col(0) = result_0; }{result_0 = -64.0*x*z*(x - 1)*(2*y - 1)*(z - 1);val.col(1) = result_0; }{result_0 = -64.0*x*y*(x - 1)*(y - 1)*(2*z - 1);val.col(2) = result_0; }} break;
	default: assert(false);
}}

}}