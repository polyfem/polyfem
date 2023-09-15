#include "auto_q_bases_2d_grad.hpp"


namespace polyfem {
namespace autogen {
namespace {
void q_0_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }} break;
	default: assert(false);
}}

void q_1_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = 1.0*(y - 1);val.col(0) = result_0; }{result_0 = 1.0*(x - 1);val.col(1) = result_0; }} break;
	case 1: {{result_0 = 1.0*(1 - y);val.col(0) = result_0; }{result_0 = -1.0*x;val.col(1) = result_0; }} break;
	case 2: {{result_0 = 1.0*y;val.col(0) = result_0; }{result_0 = 1.0*x;val.col(1) = result_0; }} break;
	case 3: {{result_0 = -1.0*y;val.col(0) = result_0; }{result_0 = 1.0*(1 - x);val.col(1) = result_0; }} break;
	default: assert(false);
}}

void q_2_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = (4.0*x - 3.0)*(y - 1)*(2.0*y - 1.0);val.col(0) = result_0; }{result_0 = (x - 1)*(2.0*x - 1.0)*(4.0*y - 3.0);val.col(1) = result_0; }} break;
	case 1: {{result_0 = (4.0*x - 1.0)*(y - 1)*(2.0*y - 1.0);val.col(0) = result_0; }{result_0 = x*(2.0*x - 1.0)*(4.0*y - 3.0);val.col(1) = result_0; }} break;
	case 2: {{result_0 = y*(4.0*x - 1.0)*(2.0*y - 1.0);val.col(0) = result_0; }{result_0 = x*(2.0*x - 1.0)*(4.0*y - 1.0);val.col(1) = result_0; }} break;
	case 3: {{result_0 = y*(4.0*x - 3.0)*(2.0*y - 1.0);val.col(0) = result_0; }{result_0 = (x - 1)*(2.0*x - 1.0)*(4.0*y - 1.0);val.col(1) = result_0; }} break;
	case 4: {{result_0 = -4.0*(2*x - 1)*(y - 1)*(2.0*y - 1.0);val.col(0) = result_0; }{result_0 = -x*(x - 1)*(16.0*y - 12.0);val.col(1) = result_0; }} break;
	case 5: {{result_0 = -y*(16.0*x - 4.0)*(y - 1);val.col(0) = result_0; }{result_0 = -4.0*x*(2.0*x - 1.0)*(2*y - 1);val.col(1) = result_0; }} break;
	case 6: {{result_0 = -4.0*y*(2*x - 1)*(2.0*y - 1.0);val.col(0) = result_0; }{result_0 = -x*(x - 1)*(16.0*y - 4.0);val.col(1) = result_0; }} break;
	case 7: {{result_0 = -y*(16.0*x - 12.0)*(y - 1);val.col(0) = result_0; }{result_0 = -4.0*(x - 1)*(2.0*x - 1.0)*(2*y - 1);val.col(1) = result_0; }} break;
	case 8: {{result_0 = 16.0*y*(2*x - 1)*(y - 1);val.col(0) = result_0; }{result_0 = 16.0*x*(x - 1)*(2*y - 1);val.col(1) = result_0; }} break;
	default: assert(false);
}}

void q_3_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{const auto helper_0 = x - 1;
const auto helper_1 = 1.5*x - 1.0;
const auto helper_2 = 3.0*x - 1.0;
result_0 = (y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);val.col(0) = result_0; }{const auto helper_0 = y - 1;
const auto helper_1 = 1.5*y - 1.0;
const auto helper_2 = 3.0*y - 1.0;
result_0 = (x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);val.col(1) = result_0; }} break;
	case 1: {{const auto helper_0 = 1.4999999999999998*x;
const auto helper_1 = helper_0 - 0.49999999999999989;
const auto helper_2 = 2.9999999999999996*x;
const auto helper_3 = helper_2 - 1.9999999999999996;
result_0 = -(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);val.col(0) = result_0; }{const auto helper_0 = y - 1;
const auto helper_1 = 1.5*y - 1.0;
const auto helper_2 = 3.0*y - 1.0;
result_0 = -x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);val.col(1) = result_0; }} break;
	case 2: {{const auto helper_0 = 1.4999999999999998*x;
const auto helper_1 = helper_0 - 0.49999999999999989;
const auto helper_2 = 2.9999999999999996*x;
const auto helper_3 = helper_2 - 1.9999999999999996;
result_0 = y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);val.col(0) = result_0; }{const auto helper_0 = 1.4999999999999998*y;
const auto helper_1 = helper_0 - 0.49999999999999989;
const auto helper_2 = 2.9999999999999996*y;
const auto helper_3 = helper_2 - 1.9999999999999996;
result_0 = x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);val.col(1) = result_0; }} break;
	case 3: {{const auto helper_0 = x - 1;
const auto helper_1 = 1.5*x - 1.0;
const auto helper_2 = 3.0*x - 1.0;
result_0 = -y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);val.col(0) = result_0; }{const auto helper_0 = 1.4999999999999998*y;
const auto helper_1 = helper_0 - 0.49999999999999989;
const auto helper_2 = 2.9999999999999996*y;
const auto helper_3 = helper_2 - 1.9999999999999996;
result_0 = -(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);val.col(1) = result_0; }} break;
	case 4: {{const auto helper_0 = x - 1;
const auto helper_1 = 13.499999999999996*x - 8.9999999999999982;
result_0 = -(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 13.499999999999998*helper_0*x + helper_1*x);val.col(0) = result_0; }{const auto helper_0 = y - 1;
const auto helper_1 = 1.5*y - 1.0;
const auto helper_2 = 3.0*y - 1.0;
result_0 = -x*(x - 1)*(3.0*x - 2.0)*(13.499999999999998*helper_0*helper_1 + 6.7499999999999991*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);val.col(1) = result_0; }} break;
	case 5: {{const auto helper_0 = x - 1;
const auto helper_1 = 13.499999999999996*x - 4.4999999999999991;
result_0 = (y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 13.499999999999998*helper_0*x + helper_1*x);val.col(0) = result_0; }{const auto helper_0 = y - 1;
const auto helper_1 = 1.5*y - 1.0;
const auto helper_2 = 3.0*y - 1.0;
result_0 = x*(x - 1)*(3.0*x - 1.0)*(13.499999999999998*helper_0*helper_1 + 6.7499999999999991*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);val.col(1) = result_0; }} break;
	case 6: {{const auto helper_0 = 1.4999999999999998*x - 0.49999999999999989;
const auto helper_1 = 2.9999999999999996*x - 1.9999999999999996;
result_0 = y*(y - 1)*(3.0*y - 2.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);val.col(0) = result_0; }{const auto helper_0 = y - 1;
const auto helper_1 = 13.499999999999996*y - 8.9999999999999982;
result_0 = x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999998*helper_0*y + helper_1*y);val.col(1) = result_0; }} break;
	case 7: {{const auto helper_0 = 1.4999999999999998*x - 0.49999999999999989;
const auto helper_1 = 2.9999999999999996*x - 1.9999999999999996;
result_0 = -y*(y - 1)*(3.0*y - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);val.col(0) = result_0; }{const auto helper_0 = y - 1;
const auto helper_1 = 13.499999999999996*y - 4.4999999999999991;
result_0 = -x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999998*helper_0*y + helper_1*y);val.col(1) = result_0; }} break;
	case 8: {{const auto helper_0 = x - 1;
const auto helper_1 = 13.499999999999996*x - 4.4999999999999991;
result_0 = -y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999998*helper_0*x + helper_1*x);val.col(0) = result_0; }{const auto helper_0 = 1.4999999999999998*y - 0.49999999999999989;
const auto helper_1 = 2.9999999999999996*y - 1.9999999999999996;
result_0 = -x*(x - 1)*(3.0*x - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);val.col(1) = result_0; }} break;
	case 9: {{const auto helper_0 = x - 1;
const auto helper_1 = 13.499999999999996*x - 8.9999999999999982;
result_0 = y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999998*helper_0*x + helper_1*x);val.col(0) = result_0; }{const auto helper_0 = 1.4999999999999998*y - 0.49999999999999989;
const auto helper_1 = 2.9999999999999996*y - 1.9999999999999996;
result_0 = x*(x - 1)*(3.0*x - 2.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);val.col(1) = result_0; }} break;
	case 10: {{const auto helper_0 = x - 1;
const auto helper_1 = 1.5*x - 1.0;
const auto helper_2 = 3.0*x - 1.0;
result_0 = y*(y - 1)*(3.0*y - 1.0)*(13.499999999999998*helper_0*helper_1 + 6.7499999999999991*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);val.col(0) = result_0; }{const auto helper_0 = y - 1;
const auto helper_1 = 13.499999999999996*y - 4.4999999999999991;
result_0 = (x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(helper_0*helper_1 + 13.499999999999998*helper_0*y + helper_1*y);val.col(1) = result_0; }} break;
	case 11: {{const auto helper_0 = x - 1;
const auto helper_1 = 1.5*x - 1.0;
const auto helper_2 = 3.0*x - 1.0;
result_0 = -y*(y - 1)*(3.0*y - 2.0)*(13.499999999999998*helper_0*helper_1 + 6.7499999999999991*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);val.col(0) = result_0; }{const auto helper_0 = y - 1;
const auto helper_1 = 13.499999999999996*y - 8.9999999999999982;
result_0 = -(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(helper_0*helper_1 + 13.499999999999998*helper_0*y + helper_1*y);val.col(1) = result_0; }} break;
	case 12: {{const auto helper_0 = x - 1;
const auto helper_1 = 60.749999999999979*x - 40.499999999999986;
result_0 = y*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);val.col(0) = result_0; }{const auto helper_0 = y - 1;
const auto helper_1 = 60.749999999999979*y - 40.499999999999986;
result_0 = x*(x - 1)*(3.0*x - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);val.col(1) = result_0; }} break;
	case 13: {{const auto helper_0 = x - 1;
const auto helper_1 = 60.749999999999979*x - 40.499999999999986;
result_0 = -y*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);val.col(0) = result_0; }{const auto helper_0 = y - 1;
const auto helper_1 = 60.749999999999979*y - 20.249999999999993;
result_0 = -x*(x - 1)*(3.0*x - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);val.col(1) = result_0; }} break;
	case 14: {{const auto helper_0 = x - 1;
const auto helper_1 = 60.749999999999979*x - 20.249999999999993;
result_0 = -y*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);val.col(0) = result_0; }{const auto helper_0 = y - 1;
const auto helper_1 = 60.749999999999979*y - 40.499999999999986;
result_0 = -x*(x - 1)*(3.0*x - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);val.col(1) = result_0; }} break;
	case 15: {{const auto helper_0 = x - 1;
const auto helper_1 = 60.749999999999979*x - 20.249999999999993;
result_0 = y*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);val.col(0) = result_0; }{const auto helper_0 = y - 1;
const auto helper_1 = 60.749999999999979*y - 20.249999999999993;
result_0 = x*(x - 1)*(3.0*x - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);val.col(1) = result_0; }} break;
	default: assert(false);
}}

void q_m2_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = -(y - 1)*(4.0*x + 2.0*y - 3.0);val.col(0) = result_0; }{result_0 = -(x - 1)*(2.0*x + 4.0*y - 3.0);val.col(1) = result_0; }} break;
	case 1: {{result_0 = (y - 1)*(-4.0*x + 2*y + 1);val.col(0) = result_0; }{result_0 = -x*(2.0*x - 4.0*y + 1.0);val.col(1) = result_0; }} break;
	case 2: {{result_0 = y*(4.0*x + 2.0*y - 3.0);val.col(0) = result_0; }{result_0 = x*(2.0*x + 4.0*y - 3.0);val.col(1) = result_0; }} break;
	case 3: {{result_0 = -y*(-4.0*x + 2.0*y + 1.0);val.col(0) = result_0; }{result_0 = (x - 1)*(2.0*x - 4.0*y + 1.0);val.col(1) = result_0; }} break;
	case 4: {{result_0 = 4*(2*x - 1)*(y - 1);val.col(0) = result_0; }{result_0 = 4*x*(x - 1);val.col(1) = result_0; }} break;
	case 5: {{result_0 = -4*y*(y - 1);val.col(0) = result_0; }{result_0 = -4*x*(2*y - 1);val.col(1) = result_0; }} break;
	case 6: {{result_0 = -4*y*(2*x - 1);val.col(0) = result_0; }{result_0 = -4*x*(x - 1);val.col(1) = result_0; }} break;
	case 7: {{result_0 = 4*y*(y - 1);val.col(0) = result_0; }{result_0 = 4*(x - 1)*(2*y - 1);val.col(1) = result_0; }} break;
	default: assert(false);
}}

}

void q_grad_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_basis_grad_value_2d(local_index, uv, val); break;
	case 1: q_1_basis_grad_value_2d(local_index, uv, val); break;
	case 2: q_2_basis_grad_value_2d(local_index, uv, val); break;
	case 3: q_3_basis_grad_value_2d(local_index, uv, val); break;
	case -2: q_m2_basis_grad_value_2d(local_index, uv, val); break;
	default: assert(false);
}}
}}
