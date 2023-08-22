#include "auto_q_bases_3d_grad.hpp"


namespace polyfem {
namespace autogen {
extern "C++" void q_0_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
extern "C++" void q_1_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
extern "C++" void q_2_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
extern "C++" void q_3_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
extern "C++" void q_m2_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

void q_grad_basis_value_3d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_basis_grad_value_3d(local_index, uv, val); break;
	case 1: q_1_basis_grad_value_3d(local_index, uv, val); break;
	case 2: q_2_basis_grad_value_3d(local_index, uv, val); break;
	case 3: q_3_basis_grad_value_3d(local_index, uv, val); break;
	case -2: q_m2_basis_grad_value_3d(local_index, uv, val); break;
	default: assert(false);
}}
}}
