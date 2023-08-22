#include "auto_q_bases_2d_val.hpp"


namespace polyfem {
namespace autogen {
namespace {
void q_0_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

result_0.resize(x.size(),1);
switch(local_index){
	case 0: {result_0.setOnes();} break;
	default: assert(false);
}}
void q_1_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

switch(local_index){
	case 0: {result_0 = 1.0*(x - 1)*(y - 1);} break;
	case 1: {result_0 = -1.0*x*(y - 1);} break;
	case 2: {result_0 = 1.0*x*y;} break;
	case 3: {result_0 = -1.0*y*(x - 1);} break;
	default: assert(false);
}}
void q_2_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

switch(local_index){
	case 0: {result_0 = 1.0*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0);} break;
	case 1: {result_0 = 1.0*x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0);} break;
	case 2: {result_0 = 1.0*x*y*(2.0*x - 1.0)*(2.0*y - 1.0);} break;
	case 3: {result_0 = 1.0*y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0);} break;
	case 4: {result_0 = -4.0*x*(x - 1)*(y - 1)*(2.0*y - 1.0);} break;
	case 5: {result_0 = -4.0*x*y*(2.0*x - 1.0)*(y - 1);} break;
	case 6: {result_0 = -4.0*x*y*(x - 1)*(2.0*y - 1.0);} break;
	case 7: {result_0 = -4.0*y*(x - 1)*(2.0*x - 1.0)*(y - 1);} break;
	case 8: {result_0 = 16.0*x*y*(x - 1)*(y - 1);} break;
	default: assert(false);
}}
void q_3_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

switch(local_index){
	case 0: {result_0 = 1.0*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0);} break;
	case 1: {result_0 = -1.0*x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0);} break;
	case 2: {result_0 = 1.0*x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996);} break;
	case 3: {result_0 = -1.0*y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996);} break;
	case 4: {result_0 = -4.4999999999999991*x*(x - 1)*(3.0*x - 2.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0);} break;
	case 5: {result_0 = 4.4999999999999991*x*(x - 1)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0);} break;
	case 6: {result_0 = 4.4999999999999991*x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 2.0);} break;
	case 7: {result_0 = -4.4999999999999991*x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 1.0);} break;
	case 8: {result_0 = -4.4999999999999991*x*y*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996);} break;
	case 9: {result_0 = 4.4999999999999991*x*y*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996);} break;
	case 10: {result_0 = 4.4999999999999991*y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0);} break;
	case 11: {result_0 = -4.4999999999999991*y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0);} break;
	case 12: {result_0 = 20.249999999999993*x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 2.0);} break;
	case 13: {result_0 = -20.249999999999993*x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 1.0);} break;
	case 14: {result_0 = -20.249999999999993*x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0);} break;
	case 15: {result_0 = 20.249999999999993*x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0);} break;
	default: assert(false);
}}
void q_m2_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

switch(local_index){
	case 0: {result_0 = -1.0*(x - 1)*(y - 1)*(2*x + 2*y - 1);} break;
	case 1: {result_0 = 1.0*x*(y - 1)*(-2*x + 2*y + 1);} break;
	case 2: {result_0 = x*y*(2.0*x + 2.0*y - 3.0);} break;
	case 3: {result_0 = 1.0*y*(x - 1)*(2*x - 2*y + 1);} break;
	case 4: {result_0 = 4*x*(x - 1)*(y - 1);} break;
	case 5: {result_0 = -4*x*y*(y - 1);} break;
	case 6: {result_0 = -4*x*y*(x - 1);} break;
	case 7: {result_0 = 4*y*(x - 1)*(y - 1);} break;
	default: assert(false);
}}
}

void q_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_basis_value_2d(local_index, uv, val); break;
	case 1: q_1_basis_value_2d(local_index, uv, val); break;
	case 2: q_2_basis_value_2d(local_index, uv, val); break;
	case 3: q_3_basis_value_2d(local_index, uv, val); break;
	case -2: q_m2_basis_value_2d(local_index, uv, val); break;
	default: assert(false);
}}
}}
