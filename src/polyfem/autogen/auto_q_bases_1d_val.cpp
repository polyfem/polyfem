#include "auto_q_bases_1d_val.hpp"


namespace polyfem {
namespace autogen {
namespace {
void q_0_basis_value_1d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();

result_0.resize(x.size(),1);
switch(local_index){
	case 0: {result_0.setOnes();} break;
	default: assert(false);
}}
void q_1_basis_value_1d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();

switch(local_index){
	case 0: {result_0 = 1.0*(1 - x);} break;
	case 1: {result_0 = 1.0*x;} break;
	default: assert(false);
}}
void q_2_basis_value_1d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();

switch(local_index){
	case 0: {result_0 = 1.0*(x - 1)*(2.0*x - 1.0);} break;
	case 1: {result_0 = -4.0*x*(x - 1);} break;
	case 2: {result_0 = x*(2.0*x - 1.0);} break;
	default: assert(false);
}}
void q_3_basis_value_1d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();

switch(local_index){
	case 0: {result_0 = -4.5*pow(x, 3) + 9.0*pow(x, 2) - 5.5*x + 1.0;} break;
	case 1: {result_0 = 4.4999999999999991*x*(x - 1)*(3.0*x - 2.0);} break;
	case 2: {result_0 = -x*(13.499999999999996*pow(x, 2) - 17.999999999999996*x + 4.4999999999999991);} break;
	case 3: {result_0 = x*(4.4999999999999991*pow(x, 2) - 4.4999999999999982*x + 0.99999999999999956);} break;
	default: assert(false);
}}
void q_m2_basis_value_1d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();

switch(local_index){
	case 0: {result_0 = 1.0*(x - 1)*(2.0*x - 1.0);} break;
	case 1: {result_0 = -4.0*x*(x - 1);} break;
	case 2: {result_0 = x*(2.0*x - 1.0);} break;
	default: assert(false);
}}
void q_4_basis_value_1d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();

switch(local_index){
	case 0: {result_0 = 1.0*(x - 1)*(1.3333333333333333*x - 1.0)*(2.0*x - 1.0)*(4.0*x - 1.0);} break;
	case 1: {result_0 = -x*(42.666666666666664*pow(x, 3) - 96.0*pow(x, 2) + 69.333333333333329*x - 16.0);} break;
	case 2: {const auto helper_0 = 4.0*x;
result_0 = helper_0*(helper_0 - 3.0)*(helper_0 - 1.0)*(x - 1);} break;
	case 3: {result_0 = -x*(42.666666666666664*pow(x, 3) - 74.666666666666657*pow(x, 2) + 37.333333333333329*x - 5.333333333333333);} break;
	case 4: {result_0 = x*(10.666666666666666*pow(x, 3) - 16.0*pow(x, 2) + 7.333333333333333*x - 1.0);} break;
	default: assert(false);
}}
void q_5_basis_value_1d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();

switch(local_index){
	case 0: {result_0 = -26.041666666666671*pow(x, 5) + 78.125*pow(x, 4) - 88.541666666666671*pow(x, 3) + 46.875*pow(x, 2) - 11.416666666666668*x + 1.0;} break;
	case 1: {result_0 = 6.25*x*(x - 1)*(1.6666666666666665*x - 1.3333333333333333)*(2.5*x - 1.5)*(5.0*x - 2.0);} break;
	case 2: {result_0 = -x*(260.41666666666674*pow(x, 4) - 677.08333333333348*pow(x, 3) + 614.58333333333348*pow(x, 2) - 222.91666666666674*x + 25.000000000000007);} break;
	case 3: {result_0 = 4.166666666666667*x*(x - 1)*(2.5*x - 0.5)*(4.9999999999999982*x - 3.9999999999999987)*(5.0000000000000009*x - 2.0000000000000004);} break;
	case 4: {result_0 = -x*(130.20833333333329*pow(x, 4) - 286.45833333333326*pow(x, 3) + 213.5416666666666*pow(x, 2) - 63.541666666666636*x + 6.2499999999999982);} break;
	case 5: {result_0 = x*(26.041666666666675*pow(x, 4) - 52.08333333333335*pow(x, 3) + 36.458333333333343*pow(x, 2) - 10.41666666666667*x + 1.0000000000000002);} break;
	default: assert(false);
}}
void q_6_basis_value_1d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();

switch(local_index){
	case 0: {result_0 = 1.0*(x - 1)*(1.2*x - 1.0)*(1.5*x - 1.0)*(2.0*x - 1.0)*(3.0*x - 1.0)*(6.0*x - 1.0);} break;
	case 1: {result_0 = -x*(388.79999999999984*pow(x, 5) - 1295.9999999999995*pow(x, 4) + 1673.9999999999993*pow(x, 3) - 1043.9999999999995*pow(x, 2) + 313.19999999999987*x - 35.999999999999986);} break;
	case 2: {result_0 = 4.4999999999999991*x*(x - 1)*(2.0*x - 1.6666666666666667)*(3.0*x - 2.0)*(5.9999999999999991*x - 2.9999999999999996)*(6.0*x - 1.0);} break;
	case 3: {result_0 = -x*(1295.9999999999998*pow(x, 5) - 3887.9999999999991*pow(x, 4) + 4355.9999999999982*pow(x, 3) - 2231.9999999999991*pow(x, 2) + 507.99999999999983*x - 39.999999999999986);} break;
	case 4: {result_0 = 4.4999999999999991*x*(x - 1)*(2.0*x - 0.33333333333333331)*(3.0*x - 1.0)*(5.9999999999999973*x - 4.9999999999999982)*(6.0000000000000018*x - 3.0000000000000009);} break;
	case 5: {result_0 = -x*(388.79999999999984*pow(x, 5) - 1036.7999999999997*pow(x, 4) + 1025.9999999999995*pow(x, 3) - 467.99999999999983*pow(x, 2) + 97.199999999999946*x - 7.1999999999999957);} break;
	case 6: {result_0 = x*(64.799999999999997*pow(x, 5) - 162.0*pow(x, 4) + 153.0*pow(x, 3) - 67.5*pow(x, 2) + 13.699999999999998*x - 0.99999999999999967);} break;
	default: assert(false);
}}
}

void q_basis_value_1d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_basis_value_1d(local_index, uv, val); break;
	case 1: q_1_basis_value_1d(local_index, uv, val); break;
	case 2: q_2_basis_value_1d(local_index, uv, val); break;
	case 3: q_3_basis_value_1d(local_index, uv, val); break;
	case -2: q_m2_basis_value_1d(local_index, uv, val); break;
	case 4: q_4_basis_value_1d(local_index, uv, val); break;
	case 5: q_5_basis_value_1d(local_index, uv, val); break;
	case 6: q_6_basis_value_1d(local_index, uv, val); break;
	default: assert(false);
}}
}}
