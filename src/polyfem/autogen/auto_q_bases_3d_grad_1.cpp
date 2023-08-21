#include <Eigen/Dense>
 namespace polyfem {
namespace autogen {void q_1_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = -1.0*(y - 1)*(z - 1);val.col(0) = result_0; }{result_0 = -1.0*(x - 1)*(z - 1);val.col(1) = result_0; }{result_0 = -1.0*(x - 1)*(y - 1);val.col(2) = result_0; }} break;
	case 1: {{result_0 = 1.0*(y - 1)*(z - 1);val.col(0) = result_0; }{result_0 = 1.0*x*(z - 1);val.col(1) = result_0; }{result_0 = 1.0*x*(y - 1);val.col(2) = result_0; }} break;
	case 2: {{result_0 = -1.0*y*(z - 1);val.col(0) = result_0; }{result_0 = -1.0*x*(z - 1);val.col(1) = result_0; }{result_0 = -1.0*x*y;val.col(2) = result_0; }} break;
	case 3: {{result_0 = 1.0*y*(z - 1);val.col(0) = result_0; }{result_0 = 1.0*(x - 1)*(z - 1);val.col(1) = result_0; }{result_0 = 1.0*y*(x - 1);val.col(2) = result_0; }} break;
	case 4: {{result_0 = 1.0*z*(y - 1);val.col(0) = result_0; }{result_0 = 1.0*z*(x - 1);val.col(1) = result_0; }{result_0 = 1.0*(x - 1)*(y - 1);val.col(2) = result_0; }} break;
	case 5: {{result_0 = -1.0*z*(y - 1);val.col(0) = result_0; }{result_0 = -1.0*x*z;val.col(1) = result_0; }{result_0 = -1.0*x*(y - 1);val.col(2) = result_0; }} break;
	case 6: {{result_0 = 1.0*y*z;val.col(0) = result_0; }{result_0 = 1.0*x*z;val.col(1) = result_0; }{result_0 = 1.0*x*y;val.col(2) = result_0; }} break;
	case 7: {{result_0 = -1.0*y*z;val.col(0) = result_0; }{result_0 = -1.0*z*(x - 1);val.col(1) = result_0; }{result_0 = -1.0*y*(x - 1);val.col(2) = result_0; }} break;
	default: assert(false);
}}

}}