#include <Eigen/Dense>
 namespace polyfem {
namespace autogen {void q_0_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	default: assert(false);
}}

}}