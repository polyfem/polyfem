#include "auto_p_bases.hpp"


namespace polyfem {
namespace autogen {
namespace {
void p_0_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

result_0.resize(x.size(),1);
switch(local_index){
	case 0: {result_0.setOnes();} break;
	default: assert(false);
}}
void p_0_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }} break;
	default: assert(false);
}}


void p_0_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(1, 2); res << 
0.33333333333333331, 0.33333333333333331;
}


void p_1_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

switch(local_index){
	case 0: {result_0 = -x - y + 1;} break;
	case 1: {result_0 = x;} break;
	case 2: {result_0 = y;} break;
	default: assert(false);
}}
void p_1_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0.setConstant(-1);val.col(0) = result_0; }{result_0.setConstant(-1);val.col(1) = result_0; }} break;
	case 1: {{result_0.setOnes();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }} break;
	case 2: {{result_0.setZero();val.col(0) = result_0; }{result_0.setOnes();val.col(1) = result_0; }} break;
	default: assert(false);
}}


void p_1_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(3, 2); res << 
0, 0,
1, 0,
0, 1;
}


void p_2_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

switch(local_index){
	case 0: {result_0 = (x + y - 1)*(2*x + 2*y - 1);} break;
	case 1: {result_0 = x*(2*x - 1);} break;
	case 2: {result_0 = y*(2*y - 1);} break;
	case 3: {result_0 = -4*x*(x + y - 1);} break;
	case 4: {result_0 = 4*x*y;} break;
	case 5: {result_0 = -4*y*(x + y - 1);} break;
	default: assert(false);
}}
void p_2_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = 4*x + 4*y - 3;val.col(0) = result_0; }{result_0 = 4*x + 4*y - 3;val.col(1) = result_0; }} break;
	case 1: {{result_0 = 4*x - 1;val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }} break;
	case 2: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 4*y - 1;val.col(1) = result_0; }} break;
	case 3: {{result_0 = 4*(-2*x - y + 1);val.col(0) = result_0; }{result_0 = -4*x;val.col(1) = result_0; }} break;
	case 4: {{result_0 = 4*y;val.col(0) = result_0; }{result_0 = 4*x;val.col(1) = result_0; }} break;
	case 5: {{result_0 = -4*y;val.col(0) = result_0; }{result_0 = 4*(-x - 2*y + 1);val.col(1) = result_0; }} break;
	default: assert(false);
}}


void p_2_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(6, 2); res << 
0, 0,
1, 0,
0, 1,
1.0/2.0, 0,
1.0/2.0, 1.0/2.0,
0, 1.0/2.0;
}


void p_3_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

switch(local_index){
	case 0: {const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
result_0 = -27.0/2.0*helper_0*y + 9*helper_0 - 27.0/2.0*helper_1*x + 9*helper_1 - 9.0/2.0*pow(x, 3) + 18*x*y - 11.0/2.0*x - 9.0/2.0*pow(y, 3) - 11.0/2.0*y + 1;} break;
	case 1: {result_0 = (1.0/2.0)*x*(9*pow(x, 2) - 9*x + 2);} break;
	case 2: {result_0 = (1.0/2.0)*y*(9*pow(y, 2) - 9*y + 2);} break;
	case 3: {result_0 = (9.0/2.0)*x*(x + y - 1)*(3*x + 3*y - 2);} break;
	case 4: {result_0 = -9.0/2.0*x*(3*pow(x, 2) + 3*x*y - 4*x - y + 1);} break;
	case 5: {result_0 = (9.0/2.0)*x*y*(3*x - 1);} break;
	case 6: {result_0 = (9.0/2.0)*x*y*(3*y - 1);} break;
	case 7: {result_0 = -9.0/2.0*y*(3*x*y - x + 3*pow(y, 2) - 4*y + 1);} break;
	case 8: {result_0 = (9.0/2.0)*y*(x + y - 1)*(3*x + 3*y - 2);} break;
	case 9: {result_0 = -27*x*y*(x + y - 1);} break;
	default: assert(false);
}}
void p_3_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = -27.0/2.0*pow(x, 2) - 27*x*y + 18*x - 27.0/2.0*pow(y, 2) + 18*y - 11.0/2.0;val.col(0) = result_0; }{result_0 = -27.0/2.0*pow(x, 2) - 27*x*y + 18*x - 27.0/2.0*pow(y, 2) + 18*y - 11.0/2.0;val.col(1) = result_0; }} break;
	case 1: {{result_0 = (27.0/2.0)*pow(x, 2) - 9*x + 1;val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }} break;
	case 2: {{result_0.setZero();val.col(0) = result_0; }{result_0 = (27.0/2.0)*pow(y, 2) - 9*y + 1;val.col(1) = result_0; }} break;
	case 3: {{result_0 = 9*((9.0/2.0)*pow(x, 2) + 6*x*y - 5*x + (3.0/2.0)*pow(y, 2) - 5.0/2.0*y + 1);val.col(0) = result_0; }{result_0 = (9.0/2.0)*x*(6*x + 6*y - 5);val.col(1) = result_0; }} break;
	case 4: {{result_0 = 9*(-9.0/2.0*pow(x, 2) - 3*x*y + 4*x + (1.0/2.0)*y - 1.0/2.0);val.col(0) = result_0; }{result_0 = -9.0/2.0*x*(3*x - 1);val.col(1) = result_0; }} break;
	case 5: {{result_0 = (9.0/2.0)*y*(6*x - 1);val.col(0) = result_0; }{result_0 = (9.0/2.0)*x*(3*x - 1);val.col(1) = result_0; }} break;
	case 6: {{result_0 = (9.0/2.0)*y*(3*y - 1);val.col(0) = result_0; }{result_0 = (9.0/2.0)*x*(6*y - 1);val.col(1) = result_0; }} break;
	case 7: {{result_0 = -9.0/2.0*y*(3*y - 1);val.col(0) = result_0; }{result_0 = 9*(-3*x*y + (1.0/2.0)*x - 9.0/2.0*pow(y, 2) + 4*y - 1.0/2.0);val.col(1) = result_0; }} break;
	case 8: {{result_0 = (9.0/2.0)*y*(6*x + 6*y - 5);val.col(0) = result_0; }{result_0 = 9*((3.0/2.0)*pow(x, 2) + 6*x*y - 5.0/2.0*x + (9.0/2.0)*pow(y, 2) - 5*y + 1);val.col(1) = result_0; }} break;
	case 9: {{result_0 = -27*y*(2*x + y - 1);val.col(0) = result_0; }{result_0 = -27*x*(x + 2*y - 1);val.col(1) = result_0; }} break;
	default: assert(false);
}}


void p_3_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(10, 2); res << 
0, 0,
1, 0,
0, 1,
1.0/3.0, 0,
2.0/3.0, 0,
2.0/3.0, 1.0/3.0,
1.0/3.0, 2.0/3.0,
0, 2.0/3.0,
0, 1.0/3.0,
1.0/3.0, 1.0/3.0;
}


void p_4_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

switch(local_index){
	case 0: {const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(x, 3);
const auto helper_2 = pow(y, 2);
const auto helper_3 = pow(y, 3);
result_0 = 64*helper_0*helper_2 - 80*helper_0*y + (70.0/3.0)*helper_0 + (128.0/3.0)*helper_1*y - 80.0/3.0*helper_1 - 80*helper_2*x + (70.0/3.0)*helper_2 + (128.0/3.0)*helper_3*x - 80.0/3.0*helper_3 + (32.0/3.0)*pow(x, 4) + (140.0/3.0)*x*y - 25.0/3.0*x + (32.0/3.0)*pow(y, 4) - 25.0/3.0*y + 1;} break;
	case 1: {result_0 = (1.0/3.0)*x*(32*pow(x, 3) - 48*pow(x, 2) + 22*x - 3);} break;
	case 2: {result_0 = (1.0/3.0)*y*(32*pow(y, 3) - 48*pow(y, 2) + 22*y - 3);} break;
	case 3: {const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
result_0 = -16.0/3.0*x*(24*helper_0*y - 18*helper_0 + 24*helper_1*x - 18*helper_1 + 8*pow(x, 3) - 36*x*y + 13*x + 8*pow(y, 3) + 13*y - 3);} break;
	case 4: {const auto helper_0 = 32*pow(x, 2);
const auto helper_1 = pow(y, 2);
result_0 = 4*x*(helper_0*y - helper_0 + 16*helper_1*x - 4*helper_1 + 16*pow(x, 3) - 36*x*y + 19*x + 7*y - 3);} break;
	case 5: {const auto helper_0 = pow(x, 2);
result_0 = -16.0/3.0*x*(8*helper_0*y - 14*helper_0 + 8*pow(x, 3) - 6*x*y + 7*x + y - 1);} break;
	case 6: {result_0 = (16.0/3.0)*x*y*(8*pow(x, 2) - 6*x + 1);} break;
	case 7: {const auto helper_0 = 4*x;
result_0 = helper_0*y*(-helper_0 + 16*x*y - 4*y + 1);} break;
	case 8: {result_0 = (16.0/3.0)*x*y*(8*pow(y, 2) - 6*y + 1);} break;
	case 9: {const auto helper_0 = pow(y, 2);
result_0 = -16.0/3.0*y*(8*helper_0*x - 14*helper_0 - 6*x*y + x + 8*pow(y, 3) + 7*y - 1);} break;
	case 10: {const auto helper_0 = pow(x, 2);
const auto helper_1 = 32*pow(y, 2);
result_0 = 4*y*(16*helper_0*y - 4*helper_0 + helper_1*x - helper_1 - 36*x*y + 7*x + 16*pow(y, 3) + 19*y - 3);} break;
	case 11: {const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
result_0 = -16.0/3.0*y*(24*helper_0*y - 18*helper_0 + 24*helper_1*x - 18*helper_1 + 8*pow(x, 3) - 36*x*y + 13*x + 8*pow(y, 3) + 13*y - 3);} break;
	case 12: {result_0 = 32*x*y*(x + y - 1)*(4*x + 4*y - 3);} break;
	case 13: {result_0 = -32*x*y*(4*y - 1)*(x + y - 1);} break;
	case 14: {result_0 = -32*x*y*(4*x - 1)*(x + y - 1);} break;
	default: assert(false);
}}
void p_4_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
result_0 = 128*helper_0*y - 80*helper_0 + 128*helper_1*x - 80*helper_1 + (128.0/3.0)*pow(x, 3) - 160*x*y + (140.0/3.0)*x + (128.0/3.0)*pow(y, 3) + (140.0/3.0)*y - 25.0/3.0;val.col(0) = result_0; }{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
result_0 = 128*helper_0*y - 80*helper_0 + 128*helper_1*x - 80*helper_1 + (128.0/3.0)*pow(x, 3) - 160*x*y + (140.0/3.0)*x + (128.0/3.0)*pow(y, 3) + (140.0/3.0)*y - 25.0/3.0;val.col(1) = result_0; }} break;
	case 1: {{result_0 = (128.0/3.0)*pow(x, 3) - 48*pow(x, 2) + (44.0/3.0)*x - 1;val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }} break;
	case 2: {{result_0.setZero();val.col(0) = result_0; }{result_0 = (128.0/3.0)*pow(y, 3) - 48*pow(y, 2) + (44.0/3.0)*y - 1;val.col(1) = result_0; }} break;
	case 3: {{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
result_0 = -384*helper_0*y + 288*helper_0 - 256*helper_1*x + 96*helper_1 - 512.0/3.0*pow(x, 3) + 384*x*y - 416.0/3.0*x - 128.0/3.0*pow(y, 3) - 208.0/3.0*y + 16;val.col(0) = result_0; }{result_0 = -16.0/3.0*x*(24*pow(x, 2) + 48*x*y - 36*x + 24*pow(y, 2) - 36*y + 13);val.col(1) = result_0; }} break;
	case 4: {{const auto helper_0 = 96*pow(x, 2);
const auto helper_1 = pow(y, 2);
result_0 = 4*helper_0*y - 4*helper_0 + 128*helper_1*x - 16*helper_1 + 256*pow(x, 3) - 288*x*y + 152*x + 28*y - 12;val.col(0) = result_0; }{result_0 = 4*x*(32*pow(x, 2) + 32*x*y - 36*x - 8*y + 7);val.col(1) = result_0; }} break;
	case 5: {{const auto helper_0 = pow(x, 2);
result_0 = -128*helper_0*y + 224*helper_0 - 512.0/3.0*pow(x, 3) + 64*x*y - 224.0/3.0*x - 16.0/3.0*y + 16.0/3.0;val.col(0) = result_0; }{result_0 = -16.0/3.0*x*(8*pow(x, 2) - 6*x + 1);val.col(1) = result_0; }} break;
	case 6: {{result_0 = (16.0/3.0)*y*(24*pow(x, 2) - 12*x + 1);val.col(0) = result_0; }{result_0 = (16.0/3.0)*x*(8*pow(x, 2) - 6*x + 1);val.col(1) = result_0; }} break;
	case 7: {{const auto helper_0 = 4*y;
result_0 = helper_0*(-helper_0 + 32*x*y - 8*x + 1);val.col(0) = result_0; }{const auto helper_0 = 4*x;
result_0 = helper_0*(-helper_0 + 32*x*y - 8*y + 1);val.col(1) = result_0; }} break;
	case 8: {{result_0 = (16.0/3.0)*y*(8*pow(y, 2) - 6*y + 1);val.col(0) = result_0; }{result_0 = (16.0/3.0)*x*(24*pow(y, 2) - 12*y + 1);val.col(1) = result_0; }} break;
	case 9: {{result_0 = -16.0/3.0*y*(8*pow(y, 2) - 6*y + 1);val.col(0) = result_0; }{const auto helper_0 = pow(y, 2);
result_0 = -128*helper_0*x + 224*helper_0 + 64*x*y - 16.0/3.0*x - 512.0/3.0*pow(y, 3) - 224.0/3.0*y + 16.0/3.0;val.col(1) = result_0; }} break;
	case 10: {{result_0 = 4*y*(32*x*y - 8*x + 32*pow(y, 2) - 36*y + 7);val.col(0) = result_0; }{const auto helper_0 = pow(x, 2);
const auto helper_1 = 96*pow(y, 2);
result_0 = 128*helper_0*y - 16*helper_0 + 4*helper_1*x - 4*helper_1 - 288*x*y + 28*x + 256*pow(y, 3) + 152*y - 12;val.col(1) = result_0; }} break;
	case 11: {{result_0 = -16.0/3.0*y*(24*pow(x, 2) + 48*x*y - 36*x + 24*pow(y, 2) - 36*y + 13);val.col(0) = result_0; }{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
result_0 = -256*helper_0*y + 96*helper_0 - 384*helper_1*x + 288*helper_1 - 128.0/3.0*pow(x, 3) + 384*x*y - 208.0/3.0*x - 512.0/3.0*pow(y, 3) - 416.0/3.0*y + 16;val.col(1) = result_0; }} break;
	case 12: {{result_0 = 32*y*(12*pow(x, 2) + 16*x*y - 14*x + 4*pow(y, 2) - 7*y + 3);val.col(0) = result_0; }{result_0 = 32*x*(4*pow(x, 2) + 16*x*y - 7*x + 12*pow(y, 2) - 14*y + 3);val.col(1) = result_0; }} break;
	case 13: {{result_0 = -32*y*(8*x*y - 2*x + 4*pow(y, 2) - 5*y + 1);val.col(0) = result_0; }{result_0 = -32*x*(8*x*y - x + 12*pow(y, 2) - 10*y + 1);val.col(1) = result_0; }} break;
	case 14: {{result_0 = -32*y*(12*pow(x, 2) + 8*x*y - 10*x - y + 1);val.col(0) = result_0; }{result_0 = -32*x*(4*pow(x, 2) + 8*x*y - 5*x - 2*y + 1);val.col(1) = result_0; }} break;
	default: assert(false);
}}


void p_4_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(15, 2); res << 
0, 0,
1, 0,
0, 1,
1.0/4.0, 0,
1.0/2.0, 0,
3.0/4.0, 0,
3.0/4.0, 1.0/4.0,
1.0/2.0, 1.0/2.0,
1.0/4.0, 3.0/4.0,
0, 3.0/4.0,
0, 1.0/2.0,
0, 1.0/4.0,
1.0/4.0, 1.0/4.0,
1.0/4.0, 1.0/2.0,
1.0/2.0, 1.0/4.0;
}


}

void p_nodes_2d(const int p, Eigen::MatrixXd &val){
switch(p){
	case 0: p_0_nodes_2d(val); break;
	case 1: p_1_nodes_2d(val); break;
	case 2: p_2_nodes_2d(val); break;
	case 3: p_3_nodes_2d(val); break;
	case 4: p_4_nodes_2d(val); break;
	default: p_n_nodes_2d(p, val);
}}
void p_basis_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(p){
	case 0: p_0_basis_value_2d(local_index, uv, val); break;
	case 1: p_1_basis_value_2d(local_index, uv, val); break;
	case 2: p_2_basis_value_2d(local_index, uv, val); break;
	case 3: p_3_basis_value_2d(local_index, uv, val); break;
	case 4: p_4_basis_value_2d(local_index, uv, val); break;
	default: p_n_basis_value_2d(p, local_index, uv, val);
}}

void p_grad_basis_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(p){
	case 0: p_0_basis_grad_value_2d(local_index, uv, val); break;
	case 1: p_1_basis_grad_value_2d(local_index, uv, val); break;
	case 2: p_2_basis_grad_value_2d(local_index, uv, val); break;
	case 3: p_3_basis_grad_value_2d(local_index, uv, val); break;
	case 4: p_4_basis_grad_value_2d(local_index, uv, val); break;
	default: p_n_basis_grad_value_2d(p, local_index, uv, val);
}}

namespace {
void p_0_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

result_0.resize(x.size(),1);
switch(local_index){
	case 0: {result_0.setOnes();} break;
	default: assert(false);
}}
void p_0_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	default: assert(false);
}}


void p_0_nodes_3d(Eigen::MatrixXd &res) {
 res.resize(1, 3); res << 
0.33333333333333331, 0.33333333333333331, 0.33333333333333331;
}


void p_1_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

switch(local_index){
	case 0: {result_0 = -x - y - z + 1;} break;
	case 1: {result_0 = x;} break;
	case 2: {result_0 = y;} break;
	case 3: {result_0 = z;} break;
	default: assert(false);
}}
void p_1_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0.setConstant(-1);val.col(0) = result_0; }{result_0.setConstant(-1);val.col(1) = result_0; }{result_0.setConstant(-1);val.col(2) = result_0; }} break;
	case 1: {{result_0.setOnes();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 2: {{result_0.setZero();val.col(0) = result_0; }{result_0.setOnes();val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 3: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0.setOnes();val.col(2) = result_0; }} break;
	default: assert(false);
}}


void p_1_nodes_3d(Eigen::MatrixXd &res) {
 res.resize(4, 3); res << 
0, 0, 0,
1, 0, 0,
0, 1, 0,
0, 0, 1;
}


void p_2_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

switch(local_index){
	case 0: {result_0 = (x + y + z - 1)*(2*x + 2*y + 2*z - 1);} break;
	case 1: {result_0 = x*(2*x - 1);} break;
	case 2: {result_0 = y*(2*y - 1);} break;
	case 3: {result_0 = z*(2*z - 1);} break;
	case 4: {result_0 = -4*x*(x + y + z - 1);} break;
	case 5: {result_0 = 4*x*y;} break;
	case 6: {result_0 = -4*y*(x + y + z - 1);} break;
	case 7: {result_0 = -4*z*(x + y + z - 1);} break;
	case 8: {result_0 = 4*x*z;} break;
	case 9: {result_0 = 4*y*z;} break;
	default: assert(false);
}}
void p_2_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = 4*x + 4*y + 4*z - 3;val.col(0) = result_0; }{result_0 = 4*x + 4*y + 4*z - 3;val.col(1) = result_0; }{result_0 = 4*x + 4*y + 4*z - 3;val.col(2) = result_0; }} break;
	case 1: {{result_0 = 4*x - 1;val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 2: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 4*y - 1;val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 3: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = 4*z - 1;val.col(2) = result_0; }} break;
	case 4: {{result_0 = 4*(-2*x - y - z + 1);val.col(0) = result_0; }{result_0 = -4*x;val.col(1) = result_0; }{result_0 = -4*x;val.col(2) = result_0; }} break;
	case 5: {{result_0 = 4*y;val.col(0) = result_0; }{result_0 = 4*x;val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 6: {{result_0 = -4*y;val.col(0) = result_0; }{result_0 = 4*(-x - 2*y - z + 1);val.col(1) = result_0; }{result_0 = -4*y;val.col(2) = result_0; }} break;
	case 7: {{result_0 = -4*z;val.col(0) = result_0; }{result_0 = -4*z;val.col(1) = result_0; }{result_0 = 4*(-x - y - 2*z + 1);val.col(2) = result_0; }} break;
	case 8: {{result_0 = 4*z;val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = 4*x;val.col(2) = result_0; }} break;
	case 9: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 4*z;val.col(1) = result_0; }{result_0 = 4*y;val.col(2) = result_0; }} break;
	default: assert(false);
}}


void p_2_nodes_3d(Eigen::MatrixXd &res) {
 res.resize(10, 3); res << 
0, 0, 0,
1, 0, 0,
0, 1, 0,
0, 0, 1,
1.0/2.0, 0, 0,
1.0/2.0, 1.0/2.0, 0,
0, 1.0/2.0, 0,
0, 0, 1.0/2.0,
1.0/2.0, 0, 1.0/2.0,
0, 1.0/2.0, 1.0/2.0;
}


void p_3_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

switch(local_index){
	case 0: {const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
const auto helper_2 = pow(z, 2);
const auto helper_3 = (27.0/2.0)*x;
const auto helper_4 = (27.0/2.0)*y;
const auto helper_5 = (27.0/2.0)*z;
result_0 = -helper_0*helper_4 - helper_0*helper_5 + 9*helper_0 - helper_1*helper_3 - helper_1*helper_5 + 9*helper_1 - helper_2*helper_3 - helper_2*helper_4 + 9*helper_2 - 9.0/2.0*pow(x, 3) - 27*x*y*z + 18*x*y + 18*x*z - 11.0/2.0*x - 9.0/2.0*pow(y, 3) + 18*y*z - 11.0/2.0*y - 9.0/2.0*pow(z, 3) - 11.0/2.0*z + 1;} break;
	case 1: {result_0 = (1.0/2.0)*x*(9*pow(x, 2) - 9*x + 2);} break;
	case 2: {result_0 = (1.0/2.0)*y*(9*pow(y, 2) - 9*y + 2);} break;
	case 3: {result_0 = (1.0/2.0)*z*(9*pow(z, 2) - 9*z + 2);} break;
	case 4: {result_0 = (9.0/2.0)*x*(x + y + z - 1)*(3*x + 3*y + 3*z - 2);} break;
	case 5: {const auto helper_0 = 3*x;
result_0 = -9.0/2.0*x*(helper_0*y + helper_0*z + 3*pow(x, 2) - 4*x - y - z + 1);} break;
	case 6: {result_0 = (9.0/2.0)*x*y*(3*x - 1);} break;
	case 7: {result_0 = (9.0/2.0)*x*y*(3*y - 1);} break;
	case 8: {const auto helper_0 = 3*y;
result_0 = -9.0/2.0*y*(helper_0*x + helper_0*z - x + 3*pow(y, 2) - 4*y - z + 1);} break;
	case 9: {result_0 = (9.0/2.0)*y*(x + y + z - 1)*(3*x + 3*y + 3*z - 2);} break;
	case 10: {result_0 = (9.0/2.0)*z*(x + y + z - 1)*(3*x + 3*y + 3*z - 2);} break;
	case 11: {const auto helper_0 = 3*z;
result_0 = -9.0/2.0*z*(helper_0*x + helper_0*y - x - y + 3*pow(z, 2) - 4*z + 1);} break;
	case 12: {result_0 = (9.0/2.0)*x*z*(3*x - 1);} break;
	case 13: {result_0 = (9.0/2.0)*x*z*(3*z - 1);} break;
	case 14: {result_0 = (9.0/2.0)*y*z*(3*y - 1);} break;
	case 15: {result_0 = (9.0/2.0)*y*z*(3*z - 1);} break;
	case 16: {result_0 = -27*x*y*(x + y + z - 1);} break;
	case 17: {result_0 = -27*x*z*(x + y + z - 1);} break;
	case 18: {result_0 = 27*x*y*z;} break;
	case 19: {result_0 = -27*y*z*(x + y + z - 1);} break;
	default: assert(false);
}}
void p_3_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{const auto helper_0 = 27*x;
result_0 = -helper_0*y - helper_0*z - 27.0/2.0*pow(x, 2) + 18*x - 27.0/2.0*pow(y, 2) - 27*y*z + 18*y - 27.0/2.0*pow(z, 2) + 18*z - 11.0/2.0;val.col(0) = result_0; }{const auto helper_0 = 27*x;
result_0 = -helper_0*y - helper_0*z - 27.0/2.0*pow(x, 2) + 18*x - 27.0/2.0*pow(y, 2) - 27*y*z + 18*y - 27.0/2.0*pow(z, 2) + 18*z - 11.0/2.0;val.col(1) = result_0; }{const auto helper_0 = 27*x;
result_0 = -helper_0*y - helper_0*z - 27.0/2.0*pow(x, 2) + 18*x - 27.0/2.0*pow(y, 2) - 27*y*z + 18*y - 27.0/2.0*pow(z, 2) + 18*z - 11.0/2.0;val.col(2) = result_0; }} break;
	case 1: {{result_0 = (27.0/2.0)*pow(x, 2) - 9*x + 1;val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 2: {{result_0.setZero();val.col(0) = result_0; }{result_0 = (27.0/2.0)*pow(y, 2) - 9*y + 1;val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 3: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = (27.0/2.0)*pow(z, 2) - 9*z + 1;val.col(2) = result_0; }} break;
	case 4: {{const auto helper_0 = 6*x;
result_0 = 9*helper_0*y + 9*helper_0*z + (81.0/2.0)*pow(x, 2) - 45*x + (27.0/2.0)*pow(y, 2) + 27*y*z - 45.0/2.0*y + (27.0/2.0)*pow(z, 2) - 45.0/2.0*z + 9;val.col(0) = result_0; }{result_0 = (9.0/2.0)*x*(6*x + 6*y + 6*z - 5);val.col(1) = result_0; }{result_0 = (9.0/2.0)*x*(6*x + 6*y + 6*z - 5);val.col(2) = result_0; }} break;
	case 5: {{const auto helper_0 = 3*x;
result_0 = -9*helper_0*y - 9*helper_0*z - 81.0/2.0*pow(x, 2) + 36*x + (9.0/2.0)*y + (9.0/2.0)*z - 9.0/2.0;val.col(0) = result_0; }{result_0 = -9.0/2.0*x*(3*x - 1);val.col(1) = result_0; }{result_0 = -9.0/2.0*x*(3*x - 1);val.col(2) = result_0; }} break;
	case 6: {{result_0 = (9.0/2.0)*y*(6*x - 1);val.col(0) = result_0; }{result_0 = (9.0/2.0)*x*(3*x - 1);val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 7: {{result_0 = (9.0/2.0)*y*(3*y - 1);val.col(0) = result_0; }{result_0 = (9.0/2.0)*x*(6*y - 1);val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 8: {{result_0 = -9.0/2.0*y*(3*y - 1);val.col(0) = result_0; }{const auto helper_0 = 3*y;
result_0 = -9*helper_0*x - 9*helper_0*z + (9.0/2.0)*x - 81.0/2.0*pow(y, 2) + 36*y + (9.0/2.0)*z - 9.0/2.0;val.col(1) = result_0; }{result_0 = -9.0/2.0*y*(3*y - 1);val.col(2) = result_0; }} break;
	case 9: {{result_0 = (9.0/2.0)*y*(6*x + 6*y + 6*z - 5);val.col(0) = result_0; }{const auto helper_0 = 6*y;
result_0 = 9*helper_0*x + 9*helper_0*z + (27.0/2.0)*pow(x, 2) + 27*x*z - 45.0/2.0*x + (81.0/2.0)*pow(y, 2) - 45*y + (27.0/2.0)*pow(z, 2) - 45.0/2.0*z + 9;val.col(1) = result_0; }{result_0 = (9.0/2.0)*y*(6*x + 6*y + 6*z - 5);val.col(2) = result_0; }} break;
	case 10: {{result_0 = (9.0/2.0)*z*(6*x + 6*y + 6*z - 5);val.col(0) = result_0; }{result_0 = (9.0/2.0)*z*(6*x + 6*y + 6*z - 5);val.col(1) = result_0; }{const auto helper_0 = 6*z;
result_0 = 9*helper_0*x + 9*helper_0*y + (27.0/2.0)*pow(x, 2) + 27*x*y - 45.0/2.0*x + (27.0/2.0)*pow(y, 2) - 45.0/2.0*y + (81.0/2.0)*pow(z, 2) - 45*z + 9;val.col(2) = result_0; }} break;
	case 11: {{result_0 = -9.0/2.0*z*(3*z - 1);val.col(0) = result_0; }{result_0 = -9.0/2.0*z*(3*z - 1);val.col(1) = result_0; }{const auto helper_0 = 3*z;
result_0 = -9*helper_0*x - 9*helper_0*y + (9.0/2.0)*x + (9.0/2.0)*y - 81.0/2.0*pow(z, 2) + 36*z - 9.0/2.0;val.col(2) = result_0; }} break;
	case 12: {{result_0 = (9.0/2.0)*z*(6*x - 1);val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = (9.0/2.0)*x*(3*x - 1);val.col(2) = result_0; }} break;
	case 13: {{result_0 = (9.0/2.0)*z*(3*z - 1);val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = (9.0/2.0)*x*(6*z - 1);val.col(2) = result_0; }} break;
	case 14: {{result_0.setZero();val.col(0) = result_0; }{result_0 = (9.0/2.0)*z*(6*y - 1);val.col(1) = result_0; }{result_0 = (9.0/2.0)*y*(3*y - 1);val.col(2) = result_0; }} break;
	case 15: {{result_0.setZero();val.col(0) = result_0; }{result_0 = (9.0/2.0)*z*(3*z - 1);val.col(1) = result_0; }{result_0 = (9.0/2.0)*y*(6*z - 1);val.col(2) = result_0; }} break;
	case 16: {{result_0 = -27*y*(2*x + y + z - 1);val.col(0) = result_0; }{result_0 = -27*x*(x + 2*y + z - 1);val.col(1) = result_0; }{result_0 = -27*x*y;val.col(2) = result_0; }} break;
	case 17: {{result_0 = -27*z*(2*x + y + z - 1);val.col(0) = result_0; }{result_0 = -27*x*z;val.col(1) = result_0; }{result_0 = -27*x*(x + y + 2*z - 1);val.col(2) = result_0; }} break;
	case 18: {{result_0 = 27*y*z;val.col(0) = result_0; }{result_0 = 27*x*z;val.col(1) = result_0; }{result_0 = 27*x*y;val.col(2) = result_0; }} break;
	case 19: {{result_0 = -27*y*z;val.col(0) = result_0; }{result_0 = -27*z*(x + 2*y + z - 1);val.col(1) = result_0; }{result_0 = -27*y*(x + y + 2*z - 1);val.col(2) = result_0; }} break;
	default: assert(false);
}}


void p_3_nodes_3d(Eigen::MatrixXd &res) {
 res.resize(20, 3); res << 
0, 0, 0,
1, 0, 0,
0, 1, 0,
0, 0, 1,
1.0/3.0, 0, 0,
2.0/3.0, 0, 0,
2.0/3.0, 1.0/3.0, 0,
1.0/3.0, 2.0/3.0, 0,
0, 2.0/3.0, 0,
0, 1.0/3.0, 0,
0, 0, 1.0/3.0,
0, 0, 2.0/3.0,
2.0/3.0, 0, 1.0/3.0,
1.0/3.0, 0, 2.0/3.0,
0, 2.0/3.0, 1.0/3.0,
0, 1.0/3.0, 2.0/3.0,
1.0/3.0, 1.0/3.0, 0,
1.0/3.0, 0, 1.0/3.0,
1.0/3.0, 1.0/3.0, 1.0/3.0,
0, 1.0/3.0, 1.0/3.0;
}


void p_4_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

switch(local_index){
	case 0: {const auto helper_0 = x + y + z - 1;
const auto helper_1 = pow(y, 2);
const auto helper_2 = 3.0*x;
const auto helper_3 = pow(z, 2);
const auto helper_4 = pow(x, 2);
const auto helper_5 = 2.9999999999999996*helper_4;
const auto helper_6 = helper_0*y;
const auto helper_7 = 8.6666666666666643*x;
const auto helper_8 = 4.333333333333333*pow(helper_0, 2);
const auto helper_9 = 4.333333333333333*helper_0;
result_0 = helper_0*(1.0*pow(helper_0, 3) + helper_0*helper_7*z + helper_1*helper_2 + helper_1*helper_9 + 3.0000000000000018*helper_1*z + helper_2*helper_3 + helper_3*helper_9 + 3.0000000000000018*helper_3*y + helper_4*helper_9 + helper_5*y + helper_5*z + helper_6*helper_7 + 8.6666666666666679*helper_6*z + helper_8*x + helper_8*y + helper_8*z + 1.0*pow(x, 3) + 6.0000000000000027*x*y*z + 1.0*pow(y, 3) + 1.0*pow(z, 3));} break;
	case 1: {const auto helper_0 = 2.6645352591003757e-15*x;
const auto helper_1 = y*z;
const auto helper_2 = pow(x, 2);
const auto helper_3 = pow(x, 3);
const auto helper_4 = pow(y, 2);
const auto helper_5 = 3.5527136788005009e-15*x;
const auto helper_6 = pow(y, 3);
const auto helper_7 = 8.8817841970012523e-16*x;
const auto helper_8 = pow(z, 2);
const auto helper_9 = pow(z, 3);
const auto helper_10 = 1.7763568394002505e-15*y;
const auto helper_11 = 1.7763568394002505e-15*z;
const auto helper_12 = helper_8*y;
const auto helper_13 = helper_4*z;
result_0 = helper_0*y + helper_0*z + 2.1316282072803006e-14*helper_1*helper_2 - 2.8421709430404007e-14*helper_1*x + 6.3540920113395232e-15*helper_1 - helper_10*helper_2 - helper_10*helper_3 - helper_11*helper_2 - helper_11*helper_3 + 3.3750779948604759e-14*helper_12*x - 2.0023736341933787e-14*helper_12 + 2.5757174171303632e-14*helper_13*x - 1.1922975078814218e-14*helper_13 + 7.333333333333333*helper_2 - 16.0*helper_3 - helper_4*helper_5 + 1.9238527398068959e-14*helper_4*helper_8 - helper_5*helper_8 + helper_6*helper_7 + 5.5688830674746946e-15*helper_6*z + helper_7*helper_9 + 1.3669644330594264e-14*helper_9*y + 10.666666666666666*pow(x, 4) - 1.0*x;} break;
	case 2: {const auto helper_0 = 4.4408920985006262e-15*x;
const auto helper_1 = 7.1054273576010019e-15*z;
const auto helper_2 = pow(x, 2);
const auto helper_3 = pow(y, 2);
const auto helper_4 = pow(z, 2);
result_0 = -y*(helper_0*helper_4 - helper_0*z + helper_1*helper_3 - helper_1*y + 1.7763568394002505e-15*helper_2*y + 3.5527136788005009e-15*helper_2*z - 8.8817841970012523e-16*helper_2 + 16.0*helper_3 + 1.0658141036401503e-14*helper_4*y - 2.6645352591003757e-15*helper_4 + 8.8817841970012523e-16*pow(x, 3) + 8.8817841970012523e-15*x*y*z - 10.666666666666666*pow(y, 3) - 7.333333333333333*y + 1.3322676295501878e-15*pow(z, 3) + 1.3322676295501878e-15*z + 1.0);} break;
	case 3: {const auto helper_0 = 1.7763568394002505e-15*y;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(y, 2);
const auto helper_3 = pow(z, 2);
const auto helper_4 = 1.7763568394002505e-15*z;
result_0 = z*(-helper_0*helper_3 + helper_0 - helper_1*helper_4 + 3.5527136788005009e-15*helper_1*y + 8.8817841970012523e-16*helper_1 + helper_2*helper_4 + 6.2172489379008766e-15*helper_2*x - 4.4408920985006262e-15*helper_2 - 16.0*helper_3 - 8.8817841970012523e-16*pow(x, 3) - 5.3290705182007514e-15*x*y + 2.6645352591003757e-15*pow(y, 3) + 10.666666666666666*pow(z, 3) + 7.333333333333333*z - 1.0);} break;
	case 4: {const auto helper_0 = x + y + z - 1;
const auto helper_1 = pow(y, 2);
const auto helper_2 = 5.3333333333333313*x;
const auto helper_3 = pow(z, 2);
const auto helper_4 = pow(x, 2);
const auto helper_5 = 10.666666666666666*helper_4;
const auto helper_6 = helper_0*y;
const auto helper_7 = 21.333333333333314*x;
result_0 = -helper_0*(16.0*pow(helper_0, 2)*x + 21.333333333333332*helper_0*helper_4 + helper_0*helper_7*z + helper_1*helper_2 + 8.8185004464819241e-15*helper_1*z + helper_2*helper_3 + 1.3586025735758804e-14*helper_3*y + helper_5*y + helper_5*z + helper_6*helper_7 + 5.8653157027749433e-15*helper_6*z + 5.333333333333333*pow(x, 3) + 10.666666666666684*x*y*z);} break;
	case 5: {const auto helper_0 = x + y + z - 1;
const auto helper_1 = x*y;
const auto helper_2 = pow(y, 2);
const auto helper_3 = 4.0000000000000036*x;
const auto helper_4 = pow(z, 2);
const auto helper_5 = pow(x, 2);
const auto helper_6 = 7.9999999999999956*helper_5;
const auto helper_7 = 8.0000000000000124*helper_0;
result_0 = helper_0*(12.0*pow(helper_0, 2)*x + 40.0*helper_0*helper_5 - helper_1*helper_7 - 7.9999999999999787*helper_1*z - helper_2*helper_3 + 1.4496165117504531e-14*helper_2*z - helper_3*helper_4 + 2.3827798982715442e-14*helper_4*y + helper_6*y + helper_6*z - helper_7*x*z + 12.0*pow(x, 3));} break;
	case 6: {const auto helper_0 = x + y + z - 1;
const auto helper_1 = pow(y, 2);
const auto helper_2 = 5.3333333333333259*x;
const auto helper_3 = pow(z, 2);
const auto helper_4 = pow(x, 2);
result_0 = helper_0*(-5.333333333333333*pow(helper_0, 2)*x - 21.333333333333332*helper_0*helper_4 + 10.666666666666686*helper_0*x*y + 10.666666666666686*helper_0*x*z - helper_1*helper_2 - 7.7312880626690831e-15*helper_1*z - helper_2*helper_3 - 2.173764117721409e-14*helper_3*y + 21.333333333333343*helper_4*y + 21.333333333333343*helper_4*z - 16.0*pow(x, 3) - 10.666666666666686*x*y*z);} break;
	case 7: {const auto helper_0 = x*y;
const auto helper_1 = 8.8817841970012523e-15*x;
const auto helper_2 = 1.7395398141005437e-14*y;
const auto helper_3 = pow(x, 2);
const auto helper_4 = pow(y, 2);
const auto helper_5 = pow(z, 2);
result_0 = y*(2.4868995751603507e-14*helper_0*z + 1.2434497875801753e-14*helper_0 + helper_1*helper_5 - helper_1*z + helper_2*helper_5 - helper_2*z - 1.7763568394002505e-15*helper_3*y + 8.8817841970012523e-15*helper_3*z - 31.999999999999986*helper_3 - 1.7763568394002505e-15*helper_4*x + 1.7395398141005437e-14*helper_4*z + 42.666666666666664*pow(x, 3) + 5.3333333333333224*x);} break;
	case 8: {const auto helper_0 = x*y;
const auto helper_1 = 2.1316282072803006e-14*x;
const auto helper_2 = 1.3046548605754078e-14*y;
const auto helper_3 = pow(x, 2);
const auto helper_4 = pow(y, 2);
const auto helper_5 = pow(z, 2);
result_0 = -y*(3.4362830678557085e-14*helper_0*z + 15.999999999999988*helper_0 + helper_1*helper_5 - helper_1*z + helper_2*helper_5 - helper_2*z - 63.999999999999979*helper_3*y + 2.1316282072803006e-14*helper_3*z + 15.999999999999986*helper_3 + 8.8817841970012523e-15*helper_4*x + 1.3046548605754078e-14*helper_4*z + 1.0658141036401503e-14*pow(x, 3) - 3.9999999999999964*x);} break;
	case 9: {const auto helper_0 = x*y;
const auto helper_1 = 1.0658141036401503e-14*x;
const auto helper_2 = 5.7984660470018123e-15*y;
const auto helper_3 = pow(x, 2);
const auto helper_4 = pow(y, 2);
const auto helper_5 = pow(z, 2);
result_0 = y*(1.5987211554602254e-14*helper_0*z - 32.000000000000014*helper_0 + helper_1*helper_5 - helper_1*z + helper_2*helper_5 - helper_2*z + 1.0658141036401503e-14*helper_3*y + 8.8817841970012523e-15*helper_3*z - 5.3290705182007514e-15*helper_3 + 42.666666666666671*helper_4*x + 5.7984660470018123e-15*helper_4*z + 1.7763568394002505e-15*pow(x, 3) + 5.3333333333333357*x);} break;
	case 10: {const auto helper_0 = y*z;
const auto helper_1 = pow(x, 2);
const auto helper_2 = 1.7763568394002505e-15*helper_1;
const auto helper_3 = pow(y, 2);
const auto helper_4 = pow(z, 2);
const auto helper_5 = 5.5067062021407764e-14*x;
result_0 = y*(helper_0*helper_5 + 31.99999999999995*helper_0 + 2.4868995751603507e-14*helper_1*z + helper_2*y + helper_2 - 42.666666666666657*helper_3*x - 42.666666666666643*helper_3*z + 74.666666666666657*helper_3 + helper_4*helper_5 + 5.3290705182007514e-14*helper_4*y - 5.1514348342607263e-14*helper_4 - 1.7763568394002505e-15*pow(x, 3) + 31.999999999999996*x*y - 4.9737991503207013e-14*x*z - 5.333333333333333*x - 42.666666666666664*pow(y, 3) - 37.333333333333336*y + 2.8421709430404007e-14*pow(z, 3) - 5.3333333333333099*z + 5.333333333333333);} break;
	case 11: {const auto helper_0 = x + y + z - 1;
const auto helper_1 = helper_0*y;
result_0 = -helper_1*(-12.0*pow(helper_0, 2) + 7.9999999999999982*helper_0*x + 8.0000000000000195*helper_0*z - 40.0*helper_1 + 4.0*pow(x, 2) - 8.0*x*y + 8.0000000000000036*x*z - 12.0*pow(y, 2) - 7.9999999999999964*y*z + 4.0000000000000089*pow(z, 2));} break;
	case 12: {const auto helper_0 = x*y;
const auto helper_1 = 191.99999999999994*z;
const auto helper_2 = pow(x, 2);
const auto helper_3 = pow(y, 2);
const auto helper_4 = pow(z, 2);
const auto helper_5 = 127.99999999999996*helper_4;
const auto helper_6 = 127.99999999999997*z;
result_0 = -y*(255.99999999999994*helper_0*z - 192.0*helper_0 - helper_1*x - helper_1*y + helper_2*helper_6 + 128.0*helper_2*y - 96.0*helper_2 + helper_3*helper_6 + 128.0*helper_3*x - 96.0*helper_3 - 95.999999999999957*helper_4 + helper_5*x + helper_5*y + 42.666666666666664*pow(x, 3) + 69.333333333333329*x + 42.666666666666664*pow(y, 3) + 69.333333333333329*y + 42.666666666666643*pow(z, 3) + 69.333333333333314*z - 16.0);} break;
	case 13: {const auto helper_0 = 192.0*x;
const auto helper_1 = y*z;
const auto helper_2 = pow(x, 2);
const auto helper_3 = pow(y, 2);
const auto helper_4 = pow(z, 2);
const auto helper_5 = 128.0*x;
const auto helper_6 = 128.0*y;
const auto helper_7 = 128.0*z;
result_0 = -z*(-helper_0*y - helper_0*z + 256.0*helper_1*x - 192.0*helper_1 + helper_2*helper_6 + helper_2*helper_7 - 96.0*helper_2 + helper_3*helper_5 + helper_3*helper_7 - 96.0*helper_3 + helper_4*helper_5 + helper_4*helper_6 - 96.0*helper_4 + 42.666666666666664*pow(x, 3) + 69.333333333333329*x + 42.666666666666671*pow(y, 3) + 69.333333333333343*y + 42.666666666666664*pow(z, 3) + 69.333333333333329*z - 16.0);} break;
	case 14: {const auto helper_0 = x + y + z - 1;
const auto helper_1 = helper_0*z;
result_0 = -helper_1*(-12.0*pow(helper_0, 2) + 7.9999999999999982*helper_0*x + 8.0*helper_0*y - 40.0*helper_1 + 4.0*pow(x, 2) + 8.0000000000000036*x*y - 8.0*x*z + 4.0000000000000027*pow(y, 2) - 7.9999999999999947*y*z - 12.0*pow(z, 2));} break;
	case 15: {const auto helper_0 = 7.1054273576010019e-15*y;
const auto helper_1 = y*z;
const auto helper_2 = pow(x, 2);
const auto helper_3 = 1.7763568394002505e-15*helper_2;
const auto helper_4 = pow(y, 2);
const auto helper_5 = pow(z, 2);
const auto helper_6 = 42.666666666666657*helper_5;
result_0 = -z*(helper_0*helper_2 - helper_0*x - 1.7763568394002505e-15*helper_1*x - 32.0*helper_1 - helper_3*z - helper_3 + 8.8817841970012523e-15*helper_4*x - 5.3290705182007514e-15*helper_4 - 74.666666666666657*helper_5 + helper_6*x + helper_6*y + 1.7763568394002505e-15*pow(x, 3) - 31.999999999999996*x*z + 5.333333333333333*x + 3.5527136788005009e-15*pow(y, 3) + 5.3333333333333348*y + 42.666666666666664*pow(z, 3) + 37.333333333333336*z - 5.333333333333333);} break;
	case 16: {const auto helper_0 = 2.3461262811099773e-14*y;
const auto helper_1 = x*z;
const auto helper_2 = pow(x, 2);
const auto helper_3 = pow(y, 2);
const auto helper_4 = 5.5620224692702261e-14*helper_3;
const auto helper_5 = pow(z, 2);
result_0 = -z*(helper_0*helper_5 + helper_0 + 4.9737991503207013e-14*helper_1*y - 1.2434497875801753e-14*helper_1 + 5.3290705182007514e-14*helper_2*y + 1.7763568394002505e-15*helper_2*z + 31.999999999999986*helper_2 + 8.1712414612411521e-14*helper_3*x + helper_4*z - helper_4 + 1.7763568394002505e-15*helper_5*x - 42.666666666666664*pow(x, 3) - 7.638334409421077e-14*x*y - 5.3333333333333224*x + 3.2158961881602488e-14*pow(y, 3) - 4.6922525622199546e-14*y*z);} break;
	case 17: {const auto helper_0 = 1.7595947108324833e-14*y;
const auto helper_1 = x*y;
const auto helper_2 = pow(x, 2);
const auto helper_3 = pow(y, 2);
const auto helper_4 = 4.1715168519526706e-14*helper_3;
const auto helper_5 = pow(z, 2);
result_0 = z*(helper_0*helper_5 + helper_0 + 3.1086244689504383e-14*helper_1*z - 4.8849813083506888e-14*helper_1 + 2.7533531010703882e-14*helper_2*y + 63.999999999999979*helper_2*z - 15.999999999999986*helper_2 + 6.3060667798708891e-14*helper_3*x + helper_4*z - helper_4 - 8.8817841970012523e-15*helper_5*x - 1.0658141036401503e-14*pow(x, 3) - 15.999999999999988*x*z + 3.9999999999999964*x + 2.4119221411201873e-14*pow(y, 3) - 3.5191894216649666e-14*y*z);} break;
	case 18: {const auto helper_0 = x*z;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(y, 2);
const auto helper_3 = pow(z, 2);
result_0 = -z*(3.8311764513104643e-15*helper_0*y + 32.000000000000014*helper_0 + 1.9539925233402755e-14*helper_1*y - 1.0658141036401503e-14*helper_1*z + 5.3290705182007514e-15*helper_1 + 3.730349362740526e-14*helper_2*x + 2.982536114213201e-14*helper_2*z - 2.9546898369622047e-14*helper_2 - 42.666666666666671*helper_3*x + 1.2816093768995203e-14*helper_3*y - 1.7763568394002505e-15*pow(x, 3) - 2.8421709430404007e-14*x*y - 5.3333333333333357*x + 1.7009267373136807e-14*pow(y, 3) - 2.5353724765480443e-14*y*z + 1.253763099648524e-14*y);} break;
	case 19: {const auto helper_0 = 7.1054273576010019e-15*x;
const auto helper_1 = 1.7763568394002505e-15*z;
const auto helper_2 = y*z;
result_0 = helper_2*(helper_0*y - helper_0 + helper_1*x - helper_1 + 5.3290705182007514e-15*helper_2 + 5.3290705182007514e-15*pow(x, 2) + 42.666666666666671*pow(y, 2) - 32.000000000000007*y - 1.7763568394002505e-15*pow(z, 2) + 5.3333333333333357);} break;
	case 20: {const auto helper_0 = y*z;
result_0 = -helper_0*(-63.999999999999979*helper_0 + 1.2434497875801753e-14*pow(x, 2) + 2.9309887850104133e-14*x*y + 1.865174681370263e-14*x*z - 2.5757174171303632e-14*x + 1.5987211554602254e-14*pow(y, 2) + 15.999999999999972*y + 7.1054273576010019e-15*pow(z, 2) + 15.999999999999979*z - 3.9999999999999871);} break;
	case 21: {const auto helper_0 = y*z;
result_0 = helper_0*(1.7763568394002505e-14*helper_0 + 1.4210854715202004e-14*pow(x, 2) + 2.8421709430404007e-14*x*y + 2.1316282072803006e-14*x*z - 2.4868995751603507e-14*x + 1.4210854715202004e-14*pow(y, 2) - 2.4868995751603507e-14*y + 42.666666666666671*pow(z, 2) - 32.000000000000014*z + 5.3333333333333446);} break;
	case 22: {result_0 = x*y*(x + y + z - 1)*(128.0*x + 128.0*y + 128.0*z - 96.0);} break;
	case 23: {result_0 = x*y*(2.8421709430404007e-14*x - 127.99999999999994*y + 2.8421709430404007e-14*z + 31.999999999999972)*(x + y + z - 1);} break;
	case 24: {const auto helper_0 = x + y + z - 1;
result_0 = -helper_0*y*(32.000000000000021*helper_0*x + 96.0*pow(x, 2) - 31.999999999999993*x*y - 32.0*x*z + 1.7395398141005437e-14*y*z + 1.5049136199609754e-14*pow(z, 2));} break;
	case 25: {result_0 = x*z*(x + y + z - 1)*(128.0*x + 128.0*y + 128.0*z - 96.0);} break;
	case 26: {result_0 = x*z*(2.8421709430404007e-14*x + 2.8421709430404007e-14*y - 127.99999999999994*z + 31.999999999999972)*(x + y + z - 1);} break;
	case 27: {const auto helper_0 = x + y + z - 1;
result_0 = helper_0*z*(-32.000000000000021*helper_0*x + 4.6922525622199546e-14*helper_0*y - 96.0*pow(x, 2) + 32.000000000000014*x*y + 31.999999999999993*x*z + 1.7395398141005437e-14*pow(y, 2));} break;
	case 28: {result_0 = x*y*z*(128.0*x + 7.1054273576010019e-15*y - 2.1316282072803006e-14*z - 32.0);} break;
	case 29: {result_0 = x*y*z*(128.0*z - 32.0);} break;
	case 30: {result_0 = x*y*z*(127.99999999999997*y + 2.8421709430404007e-14*z - 32.0);} break;
	case 31: {result_0 = y*z*(2.1316282072803006e-14*x - 127.99999999999997*y + 1.4210854715202004e-14*z + 31.999999999999979)*(x + y + z - 1);} break;
	case 32: {result_0 = y*z*(7.1054273576010019e-15*x + 2.1316282072803006e-14*y - 127.99999999999997*z + 31.999999999999993)*(x + y + z - 1);} break;
	case 33: {result_0 = y*z*(x + y + z - 1)*(128.00000000000003*x + 128.00000000000006*y + 128.00000000000006*z - 96.000000000000028);} break;
	case 34: {result_0 = y*z*(-256.0*x + 3.4790796282010874e-14*y + 3.0098272399219508e-14*z)*(x + y + z - 1);} break;
	default: assert(false);
}}
void p_4_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{const auto helper_0 = 159.99999999999994*x;
const auto helper_1 = y*z;
const auto helper_2 = pow(x, 2);
const auto helper_3 = pow(y, 2);
const auto helper_4 = pow(z, 2);
const auto helper_5 = 127.99999999999999*x;
const auto helper_6 = 128.0*y;
const auto helper_7 = 128.0*z;
result_0 = -helper_0*y - helper_0*z + 255.99999999999991*helper_1*x - 159.99999999999997*helper_1 + helper_2*helper_6 + helper_2*helper_7 - 80.0*helper_2 + helper_3*helper_5 + helper_3*helper_7 - 80.0*helper_3 + helper_4*helper_5 + helper_4*helper_6 - 80.0*helper_4 + 42.666666666666664*pow(x, 3) + 46.666666666666657*x + 42.666666666666664*pow(y, 3) + 46.666666666666657*y + 42.666666666666664*pow(z, 3) + 46.666666666666657*z - 8.3333333333333321;val.col(0) = result_0; }{const auto helper_0 = x*y;
const auto helper_1 = 160.0*z;
const auto helper_2 = pow(x, 2);
const auto helper_3 = pow(y, 2);
const auto helper_4 = pow(z, 2);
const auto helper_5 = 127.99999999999999*x;
const auto helper_6 = 128.0*y;
const auto helper_7 = 128.0*z;
result_0 = 256.0*helper_0*z - 159.99999999999994*helper_0 - helper_1*x - helper_1*y + helper_2*helper_6 + helper_2*helper_7 - 80.0*helper_2 + helper_3*helper_5 + helper_3*helper_7 - 80.0*helper_3 + helper_4*helper_5 + helper_4*helper_6 - 80.0*helper_4 + 42.666666666666664*pow(x, 3) + 46.666666666666657*x + 42.666666666666664*pow(y, 3) + 46.666666666666657*y + 42.666666666666671*pow(z, 3) + 46.666666666666664*z - 8.3333333333333321;val.col(1) = result_0; }{const auto helper_0 = 160.0*y;
const auto helper_1 = x*z;
const auto helper_2 = pow(x, 2);
const auto helper_3 = pow(y, 2);
const auto helper_4 = pow(z, 2);
const auto helper_5 = 127.99999999999999*x;
const auto helper_6 = 128.0*y;
const auto helper_7 = 128.0*z;
result_0 = -helper_0*x - helper_0*z + 256.0*helper_1*y - 159.99999999999994*helper_1 + helper_2*helper_6 + helper_2*helper_7 - 80.0*helper_2 + helper_3*helper_5 + helper_3*helper_7 - 80.0*helper_3 + helper_4*helper_5 + helper_4*helper_6 - 80.0*helper_4 + 42.666666666666664*pow(x, 3) + 46.666666666666657*x + 42.666666666666671*pow(y, 3) + 46.666666666666664*y + 42.666666666666664*pow(z, 3) + 46.666666666666657*z - 8.3333333333333321;val.col(2) = result_0; }} break;
	case 1: {{const auto helper_0 = 2.8421709430404007e-14*y*z;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(y, 2);
const auto helper_3 = pow(z, 2);
const auto helper_4 = 7.1054273576010019e-15*helper_1;
result_0 = helper_0*x - helper_0 - 48.0*helper_1 + 2.4868995751603507e-14*helper_2*z - 3.5527136788005009e-15*helper_2 + 3.3750779948604759e-14*helper_3*y - 3.5527136788005009e-15*helper_3 - helper_4*y - helper_4*z + 42.666666666666664*pow(x, 3) + 14.666666666666666*x + 8.8817841970012523e-16*pow(y, 3) + 2.6645352591003757e-15*y + 8.8817841970012523e-16*pow(z, 3) + 2.6645352591003757e-15*z - 1.0;val.col(0) = result_0; }{const auto helper_0 = 2.6645352591003757e-15*x;
const auto helper_1 = x*y;
const auto helper_2 = pow(x, 2);
const auto helper_3 = pow(z, 2);
const auto helper_4 = pow(y, 2);
result_0 = helper_0*helper_4 + helper_0 + 5.0491302748632193e-14*helper_1*z - 7.1054273576010019e-15*helper_1 + 1.8788589887141276e-14*helper_2*z - 1.7763568394002505e-15*helper_2 + 3.4234591057135791e-14*helper_3*x + 3.8477054796137918e-14*helper_3*y - 2.0023736341933787e-14*helper_3 + 1.6706649202424084e-14*helper_4*z - 8.8817841970012523e-16*pow(x, 3) - 2.8695395577281301e-14*x*z - 2.3845950157628436e-14*y*z + 1.3669644330594264e-14*pow(z, 3) + 6.3540920113395232e-15*z;val.col(1) = result_0; }{const auto helper_0 = 2.6645352591003757e-15*x;
const auto helper_1 = x*z;
const auto helper_2 = pow(x, 2);
const auto helper_3 = pow(y, 2);
const auto helper_4 = pow(z, 2);
result_0 = helper_0*helper_4 + helper_0 + 6.8469182114271582e-14*helper_1*y - 7.1054273576010019e-15*helper_1 + 1.8788589887141276e-14*helper_2*y - 1.7763568394002505e-15*helper_2 + 2.5245651374316096e-14*helper_3*x + 3.8477054796137918e-14*helper_3*z - 1.1922975078814218e-14*helper_3 + 4.1008932991782792e-14*helper_4*y - 8.8817841970012523e-16*pow(x, 3) - 2.8695395577281301e-14*x*y + 5.5688830674746946e-15*pow(y, 3) - 4.0047472683867575e-14*y*z + 6.3540920113395232e-15*y;val.col(2) = result_0; }} break;
	case 2: {{const auto helper_0 = 1.7763568394002505e-15*x;
result_0 = -y*(helper_0*y - helper_0 + 2.6645352591003757e-15*pow(x, 2) + 6.2172489379008766e-15*x*z + 9.7699626167013776e-15*y*z + 4.4408920985006262e-15*pow(z, 2) - 4.4408920985006262e-15*z);val.col(0) = result_0; }{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
const auto helper_2 = pow(z, 2);
const auto helper_3 = 3.5527136788005009e-15*helper_0;
result_0 = 8.8817841970012523e-16*helper_0 - 2.1316282072803006e-14*helper_1*z - 48.0*helper_1 - 4.4408920985006262e-15*helper_2*x - 2.1316282072803006e-14*helper_2*y + 2.6645352591003757e-15*helper_2 - helper_3*y - helper_3*z - 8.8817841970012523e-16*pow(x, 3) - 2.1316282072803006e-14*x*y*z + 4.4408920985006262e-15*x*z + 42.666666666666664*pow(y, 3) + 1.7763568394002505e-14*y*z + 14.666666666666666*y - 1.3322676295501878e-15*pow(z, 3) - 1.3322676295501878e-15*z - 1.0;val.col(1) = result_0; }{result_0 = -y*(3.1086244689504383e-15*pow(x, 2) + 9.7699626167013776e-15*x*y + 8.8817841970012523e-15*x*z - 4.4408920985006262e-15*x + 6.6613381477509392e-15*pow(y, 2) + 1.5987211554602254e-14*y*z - 7.9936057773011271e-15*y + 3.9968028886505635e-15*pow(z, 2) - 5.3290705182007514e-15*z + 1.3322676295501878e-15);val.col(2) = result_0; }} break;
	case 3: {{const auto helper_0 = 1.7763568394002505e-15*x;
result_0 = -z*(helper_0*z - helper_0 + 2.6645352591003757e-15*pow(x, 2) - 6.2172489379008766e-15*x*y - 6.2172489379008766e-15*pow(y, 2) + 5.3290705182007514e-15*y);val.col(0) = result_0; }{result_0 = z*(3.1086244689504383e-15*pow(x, 2) + 1.2434497875801753e-14*x*y - 5.3290705182007514e-15*x + 7.9936057773011271e-15*pow(y, 2) + 1.7763568394002505e-15*y*z - 8.8817841970012523e-15*y - 1.7763568394002505e-15*pow(z, 2) + 1.7763568394002505e-15);val.col(1) = result_0; }{const auto helper_0 = x*y;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(y, 2);
const auto helper_3 = pow(z, 2);
result_0 = -1.0658141036401503e-14*helper_0*z - 5.3290705182007514e-15*helper_0 + 3.5527136788005009e-15*helper_1*y - 3.5527136788005009e-15*helper_1*z + 8.8817841970012523e-16*helper_1 + 6.2172489379008766e-15*helper_2*x + 3.5527136788005009e-15*helper_2*z - 4.4408920985006262e-15*helper_2 - 7.1054273576010019e-15*helper_3*y - 48.0*helper_3 - 8.8817841970012523e-16*pow(x, 3) + 2.6645352591003757e-15*pow(y, 3) + 1.7763568394002505e-15*y + 42.666666666666664*pow(z, 3) + 14.666666666666666*z - 1.0;val.col(2) = result_0; }} break;
	case 4: {{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
const auto helper_2 = pow(z, 2);
const auto helper_3 = 255.99999999999994*x;
const auto helper_4 = 383.99999999999989*helper_0;
result_0 = 288.0*helper_0 - helper_1*helper_3 - 127.99999999999997*helper_1*z + 95.999999999999957*helper_1 - helper_2*helper_3 - 127.99999999999999*helper_2*y + 95.999999999999957*helper_2 - helper_4*y - helper_4*z - 170.66666666666666*pow(x, 3) - 511.99999999999983*x*y*z + 383.99999999999989*x*y + 383.99999999999989*x*z - 138.66666666666666*x - 42.666666666666643*pow(y, 3) + 191.99999999999994*y*z - 69.333333333333314*y - 42.666666666666643*pow(z, 3) - 69.333333333333314*z + 16.0;val.col(0) = result_0; }{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(z, 2);
const auto helper_2 = pow(y, 2);
result_0 = -255.99999999999991*helper_0*y - 255.99999999999994*helper_0*z + 191.99999999999994*helper_0 - 127.99999999999999*helper_1*x - 6.8270315175581223e-14*helper_1*y + 2.5316657141308691e-14*helper_1 - 127.99999999999994*helper_2*x - 4.4051448447770605e-14*helper_2*z - 127.99999999999997*pow(x, 3) - 255.99999999999994*x*y*z + 191.99999999999991*x*y + 191.99999999999994*x*z - 69.333333333333314*x + 4.1098263704063621e-14*y*z - 1.9451341438533747e-14*pow(z, 3) - 5.8653157027749433e-15*z;val.col(1) = result_0; }{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
const auto helper_2 = pow(z, 2);
result_0 = -255.99999999999994*helper_0*y - 255.99999999999991*helper_0*z + 191.99999999999994*helper_0 - 127.99999999999999*helper_1*x - 6.8270315175581223e-14*helper_1*z + 2.0549131852031811e-14*helper_1 - 127.99999999999994*helper_2*x - 5.8354024315601242e-14*helper_2*y - 127.99999999999997*pow(x, 3) - 255.99999999999994*x*y*z + 191.99999999999994*x*y + 191.99999999999991*x*z - 69.333333333333314*x - 1.4683816149256867e-14*pow(y, 3) + 5.0633314282617381e-14*y*z - 5.8653157027749433e-15*y;val.col(2) = result_0; }} break;
	case 5: {{const auto helper_0 = 287.99999999999994*x;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(y, 2);
const auto helper_3 = pow(z, 2);
result_0 = -helper_0*y - helper_0*z + 383.99999999999994*helper_1*y + 383.99999999999994*helper_1*z - 384.0*helper_1 + 127.99999999999994*helper_2*x - 1.7763568394002505e-15*helper_2*z - 15.999999999999972*helper_2 + 127.99999999999994*helper_3*x - 3.5527136788005009e-15*helper_3*y - 15.999999999999972*helper_3 + 256.0*pow(x, 3) + 255.99999999999991*x*y*z + 152.0*x - 1.5987211554602254e-14*pow(y, 3) - 31.999999999999964*y*z + 27.999999999999986*y - 1.5987211554602254e-14*pow(z, 3) + 27.999999999999986*z - 12.0;val.col(0) = result_0; }{const auto helper_0 = y*z;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(z, 2);
const auto helper_3 = pow(y, 2);
result_0 = -1.4210854715202004e-14*helper_0*x - 2.8992330235009062e-14*helper_0 + 127.99999999999994*helper_1*y + 127.99999999999997*helper_1*z - 143.99999999999997*helper_1 + 3.5527136788005009e-15*helper_2*x + 7.6647928200439939e-14*helper_2*y - 2.3827798982715442e-14*helper_2 - 4.9737991503207013e-14*helper_3*x + 4.3488495352513589e-14*helper_3*z + 127.99999999999999*pow(x, 3) - 31.99999999999994*x*y - 31.999999999999968*x*z + 27.999999999999986*x + 2.3827798982715442e-14*pow(z, 3);val.col(1) = result_0; }{const auto helper_0 = y*z;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(y, 2);
const auto helper_3 = 7.1054273576010019e-15*x;
const auto helper_4 = pow(z, 2);
result_0 = helper_0*helper_3 - 4.7655597965430884e-14*helper_0 + 127.99999999999997*helper_1*y + 127.99999999999994*helper_1*z - 143.99999999999997*helper_1 - helper_2*helper_3 + 7.6647928200439939e-14*helper_2*z - 1.4496165117504531e-14*helper_2 - 4.9737991503207013e-14*helper_4*x + 7.148339694814632e-14*helper_4*y + 127.99999999999999*pow(x, 3) - 31.999999999999968*x*y - 31.99999999999994*x*z + 27.999999999999986*x + 1.4496165117504531e-14*pow(y, 3);val.col(2) = result_0; }} break;
	case 6: {{const auto helper_0 = 63.999999999999901*x;
const auto helper_1 = y*z;
const auto helper_2 = pow(x, 2);
const auto helper_3 = pow(y, 2);
const auto helper_4 = pow(z, 2);
const auto helper_5 = 1.1368683772161603e-13*x;
const auto helper_6 = 127.99999999999989*helper_2;
result_0 = helper_0*y + helper_0*z + 1.5631940186722204e-13*helper_1*x - 5.6843418860808015e-14*helper_1 + 224.0*helper_2 + helper_3*helper_5 + 3.907985046680551e-14*helper_3*z - 4.5297099404706387e-14*helper_3 + helper_4*helper_5 + 2.4868995751603507e-14*helper_4*y - 4.5297099404706387e-14*helper_4 - helper_6*y - helper_6*z - 170.66666666666666*pow(x, 3) - 74.666666666666657*x + 2.6645352591003757e-14*pow(y, 3) - 5.3333333333333144*y + 2.6645352591003757e-14*pow(z, 3) - 5.3333333333333144*z + 5.333333333333333;val.col(0) = result_0; }{const auto helper_0 = y*z;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(z, 2);
const auto helper_3 = pow(y, 2);
result_0 = 7.5131622684074608e-14*helper_0*x + 1.5462576125338166e-14*helper_0 + 1.1191048088221578e-13*helper_1*y + 7.815970093361102e-14*helper_1*z + 31.99999999999995*helper_1 + 2.4868995751603507e-14*helper_2*x - 5.8937858479766339e-14*helper_2*y + 2.173764117721409e-14*helper_2 + 7.9936057773011271e-14*helper_3*x - 2.3193864188007249e-14*helper_3*z - 42.666666666666636*pow(x, 3) - 9.0594198809412774e-14*x*y - 5.6843418860808015e-14*x*z - 5.3333333333333144*x - 2.173764117721409e-14*pow(z, 3);val.col(1) = result_0; }{const auto helper_0 = y*z;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(y, 2);
const auto helper_3 = pow(z, 2);
result_0 = 4.7118916454984594e-14*helper_0*x + 4.3475282354428179e-14*helper_0 + 7.815970093361102e-14*helper_1*y + 1.1191048088221578e-13*helper_1*z + 31.99999999999995*helper_1 + 3.907985046680551e-14*helper_2*x - 5.8937858479766339e-14*helper_2*z + 7.7312880626690831e-15*helper_2 + 7.9936057773011271e-14*helper_3*x - 6.5212923531642272e-14*helper_3*y - 42.666666666666636*pow(x, 3) - 5.6843418860808015e-14*x*y - 9.0594198809412774e-14*x*z - 5.3333333333333144*x - 7.7312880626690831e-15*pow(y, 3);val.col(2) = result_0; }} break;
	case 7: {{result_0 = y*(128.0*pow(x, 2) - 7.1054273576010019e-15*x*y + 1.4210854715202004e-14*x*z - 63.999999999999972*x - 1.7763568394002505e-15*pow(y, 2) + 2.8421709430404007e-14*y*z + 1.2434497875801753e-14*y + 8.8817841970012523e-15*pow(z, 2) - 8.8817841970012523e-15*z + 5.3333333333333224);val.col(0) = result_0; }{const auto helper_0 = x*y;
const auto helper_1 = 8.8817841970012523e-15*x;
const auto helper_2 = 3.4790796282010874e-14*y;
const auto helper_3 = pow(x, 2);
const auto helper_4 = pow(y, 2);
const auto helper_5 = pow(z, 2);
result_0 = 5.1514348342607263e-14*helper_0*z + 2.4868995751603507e-14*helper_0 + helper_1*helper_5 - helper_1*z + helper_2*helper_5 - helper_2*z - 3.5527136788005009e-15*helper_3*y + 8.8817841970012523e-15*helper_3*z - 31.999999999999986*helper_3 - 5.3290705182007514e-15*helper_4*x + 5.2186194423016311e-14*helper_4*z + 42.666666666666664*pow(x, 3) + 5.3333333333333224*x;val.col(1) = result_0; }{const auto helper_0 = 8.8817841970012523e-15*x;
const auto helper_1 = 1.7395398141005437e-14*y;
const auto helper_2 = x + y + z - 1;
result_0 = y*(helper_0*helper_2 + helper_0*z + helper_1*helper_2 + helper_1*z);val.col(2) = result_0; }} break;
	case 8: {{result_0 = -y*(3.1974423109204508e-14*pow(x, 2) - 127.99999999999997*x*y + 4.2632564145606011e-14*x*z + 31.999999999999972*x + 8.8817841970012523e-15*pow(y, 2) + 3.3750779948604759e-14*y*z + 15.999999999999988*y + 2.1316282072803006e-14*pow(z, 2) - 2.1316282072803006e-14*z - 3.9999999999999964);val.col(0) = result_0; }{const auto helper_0 = x*y;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(y, 2);
const auto helper_3 = pow(z, 2);
result_0 = -6.872566135711417e-14*helper_0*z - 31.999999999999975*helper_0 + 127.99999999999999*helper_1*y - 2.1316282072803006e-14*helper_1*z - 15.999999999999986*helper_1 - 2.8421709430404007e-14*helper_2*x - 3.9139645817262231e-14*helper_2*z - 2.1316282072803006e-14*helper_3*x - 2.6093097211508155e-14*helper_3*y - 1.0658141036401503e-14*pow(x, 3) + 2.1316282072803006e-14*x*z + 3.9999999999999964*x + 2.6093097211508155e-14*y*z;val.col(1) = result_0; }{const auto helper_0 = 2.1316282072803006e-14*x;
const auto helper_1 = 1.3046548605754078e-14*y;
const auto helper_2 = x + y + z - 1;
result_0 = -y*(helper_0*helper_2 + helper_0*z + helper_1*helper_2 + helper_1*z);val.col(2) = result_0; }} break;
	case 9: {{result_0 = y*(8.8817841970012523e-15*pow(x, 2) + 2.4868995751603507e-14*x*y + 2.1316282072803006e-14*x*z - 1.0658141036401503e-14*x + 42.666666666666671*pow(y, 2) + 1.7763568394002505e-14*y*z - 32.000000000000014*y + 1.0658141036401503e-14*pow(z, 2) - 1.0658141036401503e-14*z + 5.3333333333333357);val.col(0) = result_0; }{const auto helper_0 = x*y;
const auto helper_1 = 1.0658141036401503e-14*x;
const auto helper_2 = 1.1596932094003625e-14*y;
const auto helper_3 = pow(x, 2);
const auto helper_4 = pow(y, 2);
const auto helper_5 = pow(z, 2);
result_0 = 3.1974423109204508e-14*helper_0*z - 64.000000000000028*helper_0 + helper_1*helper_5 - helper_1*z + helper_2*helper_5 - helper_2*z + 2.8421709430404007e-14*helper_3*y + 8.8817841970012523e-15*helper_3*z - 5.3290705182007514e-15*helper_3 + 128.00000000000003*helper_4*x + 1.7395398141005437e-14*helper_4*z + 1.7763568394002505e-15*pow(x, 3) + 5.3333333333333357*x;val.col(1) = result_0; }{const auto helper_0 = 1.0658141036401503e-14*x;
const auto helper_1 = 5.7984660470018123e-15*y;
const auto helper_2 = x + y + z - 1;
result_0 = y*(helper_0*helper_2 + helper_0*z + helper_1*helper_2 + helper_1*z);val.col(2) = result_0; }} break;
	case 10: {{const auto helper_0 = 3.5527136788005009e-15*x;
result_0 = y*(helper_0*y + helper_0 - 3.5527136788005009e-15*pow(x, 2) + 5.3290705182007514e-14*x*z - 42.666666666666664*pow(y, 2) + 5.6843418860808015e-14*y*z + 31.999999999999996*y + 5.6843418860808015e-14*pow(z, 2) - 4.9737991503207013e-14*z - 5.3333333333333339);val.col(0) = result_0; }{const auto helper_0 = x*y;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(y, 2);
const auto helper_3 = pow(z, 2);
result_0 = 1.2434497875801753e-13*helper_0*z + 64.0*helper_0 + 2.6645352591003757e-14*helper_1*z + 1.7763568394002505e-15*helper_1 - 128.0*helper_2*x - 127.99999999999989*helper_2*z + 224.0*helper_2 + 5.5067062021407764e-14*helper_3*x + 1.1368683772161603e-13*helper_3*y - 5.1514348342607263e-14*helper_3 - 8.8817841970012523e-16*pow(x, 3) - 4.9737991503207013e-14*x*z - 5.3333333333333339*x - 170.66666666666666*pow(y, 3) + 63.999999999999886*y*z - 74.666666666666657*y + 2.9309887850104133e-14*pow(z, 3) - 5.3333333333333108*z + 5.333333333333333;val.col(1) = result_0; }{result_0 = y*(2.6645352591003757e-14*pow(x, 2) + 5.6843418860808015e-14*x*y + 1.1368683772161603e-13*x*z - 4.9737991503207013e-14*x - 42.666666666666643*pow(y, 2) + 1.1013412404281553e-13*y*z + 31.99999999999995*y + 8.7041485130612273e-14*pow(z, 2) - 1.0302869668521453e-13*z - 5.3333333333333108);val.col(2) = result_0; }} break;
	case 11: {{result_0 = y*(3.5527136788005009e-15*pow(x, 2) + 128.0*x*y - 3.907985046680551e-14*x*z - 32.0*x + 128.0*pow(y, 2) + 127.99999999999996*y*z - 144.0*y - 4.9737991503207013e-14*pow(z, 2) - 31.999999999999957*z + 28.0);val.col(0) = result_0; }{const auto helper_0 = x*y;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(y, 2);
const auto helper_3 = 384.0*helper_2;
const auto helper_4 = pow(z, 2);
result_0 = 255.99999999999989*helper_0*z - 288.0*helper_0 + 128.0*helper_1*y - 1.9539925233402755e-14*helper_1*z - 16.000000000000004*helper_1 + 383.99999999999994*helper_2*z + helper_3*x - helper_3 - 4.9737991503207013e-14*helper_4*x + 127.99999999999991*helper_4*y - 15.999999999999954*helper_4 + 1.7763568394002505e-15*pow(x, 3) - 31.999999999999954*x*z + 28.0*x + 256.0*pow(y, 3) - 287.99999999999989*y*z + 152.0*y - 2.8421709430404007e-14*pow(z, 3) + 27.999999999999979*z - 12.0;val.col(1) = result_0; }{result_0 = -y*(2.1316282072803006e-14*pow(x, 2) - 127.99999999999994*x*y + 9.9475983006414026e-14*x*z + 31.999999999999957*x - 127.99999999999997*pow(y, 2) - 127.99999999999989*y*z + 143.99999999999994*y + 8.5265128291212022e-14*pow(z, 2) + 31.999999999999901*z - 27.999999999999979);val.col(2) = result_0; }} break;
	case 12: {{const auto helper_0 = 255.99999999999991*z;
result_0 = -y*(helper_0*x + helper_0*y + 127.99999999999999*pow(x, 2) + 255.99999999999994*x*y - 192.0*x + 127.99999999999999*pow(y, 2) - 192.0*y + 127.99999999999994*pow(z, 2) - 191.99999999999994*z + 69.333333333333329);val.col(0) = result_0; }{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
const auto helper_2 = pow(z, 2);
result_0 = -256.0*helper_0*y - 127.99999999999997*helper_0*z + 95.999999999999986*helper_0 - 384.0*helper_1*x - 383.99999999999994*helper_1*z + 288.0*helper_1 - 127.99999999999994*helper_2*x - 255.99999999999994*helper_2*y + 95.999999999999957*helper_2 - 42.666666666666664*pow(x, 3) - 511.99999999999989*x*y*z + 384.0*x*y + 191.99999999999994*x*z - 69.333333333333329*x - 170.66666666666666*pow(y, 3) + 383.99999999999989*y*z - 138.66666666666666*y - 42.666666666666643*pow(z, 3) - 69.333333333333314*z + 16.0;val.col(1) = result_0; }{result_0 = -y*(127.99999999999997*pow(x, 2) + 255.99999999999994*x*y + 255.99999999999989*x*z - 191.99999999999994*x + 127.99999999999997*pow(y, 2) + 255.99999999999991*y*z - 191.99999999999994*y + 127.99999999999994*pow(z, 2) - 191.99999999999991*z + 69.333333333333314);val.col(2) = result_0; }} break;
	case 13: {{const auto helper_0 = 256.0*y;
result_0 = -z*(helper_0*x + helper_0*z + 127.99999999999999*pow(x, 2) + 255.99999999999994*x*z - 192.0*x + 128.0*pow(y, 2) - 192.0*y + 127.99999999999999*pow(z, 2) - 192.0*z + 69.333333333333329);val.col(0) = result_0; }{const auto helper_0 = 256.0*x;
result_0 = -z*(helper_0*y + helper_0*z + 128.0*pow(x, 2) - 192.0*x + 128.0*pow(y, 2) + 256.0*y*z - 192.00000000000003*y + 128.0*pow(z, 2) - 192.0*z + 69.333333333333343);val.col(1) = result_0; }{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
const auto helper_2 = pow(z, 2);
const auto helper_3 = 384.0*helper_2;
const auto helper_4 = 256.0*z;
result_0 = -helper_0*helper_4 - 127.99999999999999*helper_0*y + 95.999999999999986*helper_0 - helper_1*helper_4 - 128.0*helper_1*x + 96.0*helper_1 + 288.0*helper_2 - helper_3*x - helper_3*y - 42.666666666666664*pow(x, 3) - 512.0*x*y*z + 191.99999999999997*x*y + 384.0*x*z - 69.333333333333329*x - 42.666666666666671*pow(y, 3) + 384.0*y*z - 69.333333333333343*y - 170.66666666666666*pow(z, 3) - 138.66666666666666*z + 16.0;val.col(2) = result_0; }} break;
	case 14: {{const auto helper_0 = 128.0*z;
result_0 = z*(helper_0*x + helper_0*y + 3.5527136788005009e-15*pow(x, 2) - 32.0*x - 7.1054273576010019e-15*pow(y, 2) - 31.999999999999996*y + 128.0*pow(z, 2) - 144.0*z + 28.0);val.col(0) = result_0; }{const auto helper_0 = 128.0*z;
result_0 = -z*(-helper_0*x - helper_0*y + 7.1054273576010019e-15*x*y + 32.0*x + 1.0658141036401503e-14*pow(y, 2) + 31.999999999999993*y - 128.0*pow(z, 2) + 144.0*z - 28.0);val.col(1) = result_0; }{const auto helper_0 = x*y;
const auto helper_1 = 288.0*z;
const auto helper_2 = pow(x, 2);
const auto helper_3 = pow(y, 2);
const auto helper_4 = 384.0*pow(z, 2);
result_0 = 256.0*helper_0*z - 31.999999999999996*helper_0 - helper_1*x - helper_1*y + 128.0*helper_2*z - 16.000000000000004*helper_2 - 4.4408920985006262e-15*helper_3*x + 127.99999999999999*helper_3*z - 15.999999999999996*helper_3 + helper_4*x + helper_4*y - helper_4 + 1.7763568394002505e-15*pow(x, 3) + 28.0*x - 2.6645352591003757e-15*pow(y, 3) + 28.0*y + 256.0*pow(z, 3) + 152.0*z - 12.0;val.col(2) = result_0; }} break;
	case 15: {{const auto helper_0 = 3.5527136788005009e-15*x;
result_0 = -z*(-helper_0*z - helper_0 + 3.5527136788005009e-15*pow(x, 2) + 1.0658141036401503e-14*x*y + 7.1054273576010019e-15*pow(y, 2) - 7.1054273576010019e-15*y + 42.666666666666664*pow(z, 2) - 31.999999999999996*z + 5.3333333333333339);val.col(0) = result_0; }{result_0 = -z*(5.3290705182007514e-15*pow(x, 2) + 1.5987211554602254e-14*x*y - 7.1054273576010019e-15*x + 1.2434497875801753e-14*pow(y, 2) + 7.1054273576010019e-15*y*z - 1.2434497875801753e-14*y + 42.666666666666664*pow(z, 2) - 32.0*z + 5.3333333333333357);val.col(1) = result_0; }{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
const auto helper_2 = pow(z, 2);
const auto helper_3 = 128.0*helper_2;
result_0 = -5.3290705182007514e-15*helper_0*y + 1.7763568394002505e-15*helper_0 - 7.1054273576010019e-15*helper_1*x + 6.2172489379008766e-15*helper_1 + 224.0*helper_2 - helper_3*x - helper_3*y - 8.8817841970012523e-16*pow(x, 3) - 1.0658141036401503e-14*x*y*z + 7.1054273576010019e-15*x*y + 64.0*x*z - 5.3333333333333339*x - 3.5527136788005009e-15*pow(y, 3) + 64.0*y*z - 5.3333333333333357*y - 170.66666666666666*pow(z, 3) - 74.666666666666657*z + 5.333333333333333;val.col(2) = result_0; }} break;
	case 16: {{result_0 = -z*(-128.0*pow(x, 2) + 1.0658141036401503e-13*x*y + 7.1054273576010019e-15*x*z + 63.999999999999972*x + 8.1712414612411521e-14*pow(y, 2) + 4.9737991503207013e-14*y*z - 7.638334409421077e-14*y + 1.7763568394002505e-15*pow(z, 2) - 1.2434497875801753e-14*z - 5.3333333333333224);val.col(0) = result_0; }{const auto helper_0 = 1.1124044938540452e-13*y;
result_0 = -z*(helper_0*z - helper_0 + 5.3659329080904031e-14*pow(x, 2) + 1.6453115456741204e-13*x*y + 4.8698882461599797e-14*x*z - 7.7120591892003804e-14*x + 9.6476885644807465e-14*pow(y, 2) + 2.3461262811099773e-14*pow(z, 2) - 4.6922525622199546e-14*z + 2.3461262811099773e-14);val.col(1) = result_0; }{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
const auto helper_2 = pow(z, 2);
result_0 = -5.3290705182007514e-14*helper_0*y - 3.5527136788005009e-15*helper_0*z - 31.999999999999986*helper_0 - 8.1712414612411521e-14*helper_1*x - 1.1124044938540452e-13*helper_1*z + 5.5620224692702261e-14*helper_1 - 5.3290705182007514e-15*helper_2*x - 7.0383788433299319e-14*helper_2*y + 42.666666666666664*pow(x, 3) - 9.4146912488213275e-14*x*y*z + 7.638334409421077e-14*x*y + 2.4868995751603507e-14*x*z + 5.3333333333333224*x - 3.2158961881602488e-14*pow(y, 3) + 9.3845051244399092e-14*y*z - 2.3461262811099773e-14*y;val.col(2) = result_0; }} break;
	case 17: {{result_0 = -z*(3.1974423109204508e-14*pow(x, 2) - 5.595524044110789e-14*x*y - 127.99999999999997*x*z + 31.999999999999972*x - 6.3060667798708891e-14*pow(y, 2) - 3.1086244689504383e-14*y*z + 4.8849813083506888e-14*y + 8.8817841970012523e-15*pow(z, 2) + 15.999999999999988*z - 3.9999999999999964);val.col(0) = result_0; }{const auto helper_0 = 8.3430337039053411e-14*y;
result_0 = z*(helper_0*z - helper_0 + 2.8254088144726336e-14*pow(x, 2) + 1.2428654434525918e-13*x*y + 3.075100211814904e-14*x*z - 4.8514570512151544e-14*x + 7.2357664233605618e-14*pow(y, 2) + 1.7595947108324833e-14*pow(z, 2) - 3.5191894216649666e-14*z + 1.7595947108324833e-14);val.col(1) = result_0; }{const auto helper_0 = y*z;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(y, 2);
const auto helper_3 = 6.3060667798708891e-14*x;
const auto helper_4 = pow(z, 2);
result_0 = helper_0*helper_3 - 7.0383788433299332e-14*helper_0 + 2.7533531010703882e-14*helper_1*y + 127.99999999999999*helper_1*z - 15.999999999999986*helper_1 + helper_2*helper_3 + 8.3430337039053411e-14*helper_2*z - 4.1715168519526706e-14*helper_2 - 2.8421709430404007e-14*helper_4*x + 5.2787841324974499e-14*helper_4*y - 1.0658141036401503e-14*pow(x, 3) - 4.8849813083506888e-14*x*y - 31.999999999999975*x*z + 3.9999999999999964*x + 2.4119221411201873e-14*pow(y, 3) + 1.7595947108324833e-14*y;val.col(2) = result_0; }} break;
	case 18: {{result_0 = z*(8.8817841970012523e-15*pow(x, 2) - 3.1974423109204508e-14*x*y + 2.4868995751603507e-14*x*z - 1.0658141036401503e-14*x - 3.730349362740526e-14*pow(y, 2) - 3.5527136788005009e-15*y*z + 2.8421709430404007e-14*y + 42.666666666666671*pow(z, 2) - 32.000000000000014*z + 5.3333333333333357);val.col(0) = result_0; }{result_0 = -z*(1.6090344675285741e-14*pow(x, 2) + 7.5081008293846347e-14*x*y + 4.0374426926774372e-15*x*z - 2.862797567177098e-14*x + 5.1027802119410421e-14*pow(y, 2) + 5.965072228426402e-14*y*z - 5.9093796739244093e-14*y + 1.2816093768995203e-14*pow(z, 2) - 2.5353724765480443e-14*z + 1.253763099648524e-14);val.col(1) = result_0; }{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(y, 2);
const auto helper_2 = pow(z, 2);
result_0 = -1.9539925233402755e-14*helper_0*y + 2.8421709430404007e-14*helper_0*z - 5.3290705182007514e-15*helper_0 - 3.730349362740526e-14*helper_1*x - 5.965072228426402e-14*helper_1*z + 2.9546898369622047e-14*helper_1 + 128.00000000000003*helper_2*x - 3.8448281306985609e-14*helper_2*y + 1.7763568394002505e-15*pow(x, 3) - 7.1054273576010019e-15*x*y*z + 2.8421709430404007e-14*x*y - 64.000000000000028*x*z + 5.3333333333333357*x - 1.7009267373136807e-14*pow(y, 3) + 5.0707449530960886e-14*y*z - 1.253763099648524e-14*y;val.col(2) = result_0; }} break;
	case 19: {{result_0 = y*z*(1.0658141036401503e-14*x + 7.1054273576010019e-15*y + 3.5527136788005009e-15*z - 7.1054273576010019e-15);val.col(0) = result_0; }{const auto helper_0 = 1.7763568394002505e-15*z;
result_0 = z*(helper_0*x - helper_0 + 5.3290705182007514e-15*pow(x, 2) + 1.4210854715202004e-14*x*y - 7.1054273576010019e-15*x + 128.00000000000003*pow(y, 2) + 8.8817841970012523e-15*y*z - 64.000000000000014*y - 1.7763568394002505e-15*pow(z, 2) + 5.3333333333333357);val.col(1) = result_0; }{const auto helper_0 = 7.1054273576010019e-15*x;
result_0 = y*(helper_0*y + helper_0*z - helper_0 + 5.3290705182007514e-15*pow(x, 2) + 42.666666666666671*pow(y, 2) + 1.0658141036401503e-14*y*z - 32.000000000000007*y - 1.7763568394002505e-15*pow(z, 2) - 3.5527136788005009e-15*z + 5.3333333333333357);val.col(2) = result_0; }} break;
	case 20: {{result_0 = -y*z*(2.5757174171303632e-14*x + 2.9309887850104133e-14*y + 1.865174681370263e-14*z - 2.5757174171303632e-14);val.col(0) = result_0; }{result_0 = -z*(1.2434497875801753e-14*pow(x, 2) + 5.6843418860808015e-14*x*y + 1.865174681370263e-14*x*z - 2.5757174171303632e-14*x + 4.9737991503207013e-14*pow(y, 2) - 127.99999999999994*y*z + 31.999999999999943*y + 7.1054273576010019e-15*pow(z, 2) + 15.999999999999979*z - 3.9999999999999871);val.col(1) = result_0; }{result_0 = -y*(1.2434497875801753e-14*pow(x, 2) + 2.9309887850104133e-14*x*y + 3.5527136788005009e-14*x*z - 2.5757174171303632e-14*x + 1.5987211554602254e-14*pow(y, 2) - 127.99999999999996*y*z + 15.999999999999972*y + 2.1316282072803006e-14*pow(z, 2) + 31.999999999999961*z - 3.9999999999999871);val.col(2) = result_0; }} break;
	case 21: {{result_0 = y*z*(2.8421709430404007e-14*x + 2.8421709430404007e-14*y + 2.1316282072803006e-14*z - 2.4868995751603507e-14);val.col(0) = result_0; }{result_0 = z*(1.4210854715202004e-14*pow(x, 2) + 5.6843418860808015e-14*x*y + 2.1316282072803006e-14*x*z - 2.4868995751603507e-14*x + 4.0856207306205761e-14*pow(y, 2) + 3.5527136788005009e-14*y*z - 4.9737991503207013e-14*y + 42.666666666666671*pow(z, 2) - 32.000000000000014*z + 5.3333333333333446);val.col(1) = result_0; }{result_0 = y*(1.4210854715202004e-14*pow(x, 2) + 2.8421709430404007e-14*x*y + 4.9737991503207013e-14*x*z - 2.4868995751603507e-14*x + 1.4210854715202004e-14*pow(y, 2) + 3.5527136788005009e-14*y*z - 2.4868995751603507e-14*y + 128.0*pow(z, 2) - 64.000000000000028*z + 5.3333333333333446);val.col(2) = result_0; }} break;
	case 22: {{const auto helper_0 = 512.0*x;
result_0 = y*(helper_0*y + helper_0*z + 384.0*pow(x, 2) - 448.0*x + 128.0*pow(y, 2) + 256.0*y*z - 224.0*y + 128.0*pow(z, 2) - 224.0*z + 96.0);val.col(0) = result_0; }{const auto helper_0 = 512.0*y;
result_0 = x*(helper_0*x + helper_0*z + 128.0*pow(x, 2) + 256.0*x*z - 224.0*x + 384.0*pow(y, 2) - 448.0*y + 128.0*pow(z, 2) - 224.0*z + 96.0);val.col(1) = result_0; }{result_0 = x*y*(256.0*x + 256.0*y + 256.0*z - 224.0);val.col(2) = result_0; }} break;
	case 23: {{result_0 = y*(8.5265128291212022e-14*pow(x, 2) - 255.99999999999983*x*y + 1.1368683772161603e-13*x*z + 63.999999999999886*x - 127.99999999999994*pow(y, 2) - 127.99999999999991*y*z + 159.99999999999991*y + 2.8421709430404007e-14*pow(z, 2) + 31.999999999999943*z - 31.999999999999972);val.col(0) = result_0; }{const auto helper_0 = 255.99999999999983*y;
result_0 = x*(-helper_0*x - helper_0*z + 2.8421709430404007e-14*pow(x, 2) + 5.6843418860808015e-14*x*z + 31.999999999999943*x - 383.99999999999983*pow(y, 2) + 319.99999999999983*y + 2.8421709430404007e-14*pow(z, 2) + 31.999999999999943*z - 31.999999999999972);val.col(1) = result_0; }{result_0 = x*y*(5.6843418860808015e-14*x - 127.99999999999991*y + 5.6843418860808015e-14*z + 31.999999999999943);val.col(2) = result_0; }} break;
	case 24: {{const auto helper_0 = 256.00000000000011*x;
result_0 = -y*(helper_0*y + helper_0*z + 384.00000000000011*pow(x, 2) - 320.00000000000011*x + 2.8421709430404007e-14*pow(y, 2) + 6.3948846218409017e-14*y*z - 32.00000000000005*y + 3.5527136788005009e-14*pow(z, 2) - 32.000000000000043*z + 32.000000000000021);val.col(0) = result_0; }{const auto helper_0 = pow(x, 2);
const auto helper_1 = pow(z, 2);
const auto helper_2 = pow(y, 2);
result_0 = -256.00000000000011*helper_0*y - 128.00000000000006*helper_0*z + 160.00000000000006*helper_0 - 3.5527136788005009e-14*helper_1*x - 6.4889068681230382e-14*helper_1*y + 1.5049136199609754e-14*helper_1 - 8.5265128291212022e-14*helper_2*x - 5.2186194423016311e-14*helper_2*z - 128.00000000000003*pow(x, 3) - 1.3426677928842489e-13*x*y*z + 64.000000000000099*x*y + 32.000000000000043*x*z - 32.000000000000021*x + 3.4790796282010874e-14*y*z - 1.5049136199609754e-14*pow(z, 3);val.col(1) = result_0; }{result_0 = -y*(128.00000000000006*pow(x, 2) + 6.7133389644212444e-14*x*y + 7.1054273576010019e-14*x*z - 32.000000000000043*x + 1.7395398141005437e-14*pow(y, 2) + 6.4889068681230382e-14*y*z - 1.7395398141005437e-14*y + 4.5147408598829263e-14*pow(z, 2) - 3.0098272399219508e-14*z);val.col(2) = result_0; }} break;
	case 25: {{const auto helper_0 = 512.0*x;
result_0 = z*(helper_0*y + helper_0*z + 384.0*pow(x, 2) - 448.0*x + 128.0*pow(y, 2) + 256.0*y*z - 224.0*y + 128.0*pow(z, 2) - 224.0*z + 96.0);val.col(0) = result_0; }{result_0 = x*z*(256.0*x + 256.0*y + 256.0*z - 224.0);val.col(1) = result_0; }{const auto helper_0 = 512.0*z;
result_0 = x*(helper_0*x + helper_0*y + 128.0*pow(x, 2) + 256.0*x*y - 224.0*x + 128.0*pow(y, 2) - 224.0*y + 384.0*pow(z, 2) - 448.0*z + 96.0);val.col(2) = result_0; }} break;
	case 26: {{result_0 = z*(8.5265128291212022e-14*pow(x, 2) + 1.1368683772161603e-13*x*y - 255.99999999999983*x*z + 63.999999999999886*x + 2.8421709430404007e-14*pow(y, 2) - 127.99999999999991*y*z + 31.999999999999943*y - 127.99999999999994*pow(z, 2) + 159.99999999999991*z - 31.999999999999972);val.col(0) = result_0; }{result_0 = x*z*(5.6843418860808015e-14*x + 5.6843418860808015e-14*y - 127.99999999999991*z + 31.999999999999943);val.col(1) = result_0; }{const auto helper_0 = 255.99999999999983*z;
result_0 = x*(-helper_0*x - helper_0*y + 2.8421709430404007e-14*pow(x, 2) + 5.6843418860808015e-14*x*y + 31.999999999999943*x + 2.8421709430404007e-14*pow(y, 2) + 31.999999999999943*y - 383.99999999999983*pow(z, 2) + 319.99999999999983*z - 31.999999999999972);val.col(2) = result_0; }} break;
	case 27: {{result_0 = -z*(384.00000000000011*pow(x, 2) + 255.99999999999997*x*y + 256.00000000000011*x*z - 320.00000000000011*x - 9.9475983006414026e-14*pow(y, 2) - 5.6843418860808015e-14*y*z - 31.999999999999936*y + 2.8421709430404007e-14*pow(z, 2) - 32.00000000000005*z + 32.000000000000021);val.col(0) = result_0; }{const auto helper_0 = 2.2248089877080905e-13*y;
result_0 = z*(helper_0*z - helper_0 - 127.99999999999999*pow(x, 2) + 2.0605739337042905e-13*x*y + 5.6843418860808015e-14*x*z + 31.999999999999936*x + 1.9295377128961493e-13*pow(y, 2) + 4.6922525622199546e-14*pow(z, 2) - 9.3845051244399092e-14*z + 4.6922525622199546e-14);val.col(1) = result_0; }{const auto helper_0 = x*y;
const auto helper_1 = pow(x, 2);
const auto helper_2 = pow(y, 2);
const auto helper_3 = pow(z, 2);
result_0 = 1.1368683772161603e-13*helper_0*z + 31.999999999999936*helper_0 - 127.99999999999999*helper_1*y - 256.00000000000011*helper_1*z + 160.00000000000006*helper_1 + 1.0302869668521453e-13*helper_2*x + 2.2248089877080905e-13*helper_2*z - 1.1124044938540452e-13*helper_2 - 8.5265128291212022e-14*helper_3*x + 1.4076757686659864e-13*helper_3*y - 128.00000000000003*pow(x, 3) + 64.000000000000099*x*z - 32.000000000000021*x + 6.4317923763204977e-14*pow(y, 3) - 1.8769010248879818e-13*y*z + 4.6922525622199546e-14*y;val.col(2) = result_0; }} break;
	case 28: {{result_0 = y*z*(256.0*x + 7.1054273576010019e-15*y - 2.1316282072803006e-14*z - 32.0);val.col(0) = result_0; }{result_0 = x*z*(128.0*x + 1.4210854715202004e-14*y - 2.1316282072803006e-14*z - 32.0);val.col(1) = result_0; }{result_0 = x*y*(128.0*x + 7.1054273576010019e-15*y - 4.2632564145606011e-14*z - 32.0);val.col(2) = result_0; }} break;
	case 29: {{result_0 = y*z*(128.0*z - 32.0);val.col(0) = result_0; }{result_0 = x*z*(128.0*z - 32.0);val.col(1) = result_0; }{result_0 = x*y*(256.0*z - 32.0);val.col(2) = result_0; }} break;
	case 30: {{result_0 = y*z*(127.99999999999997*y + 2.8421709430404007e-14*z - 32.0);val.col(0) = result_0; }{result_0 = x*z*(255.99999999999994*y + 2.8421709430404007e-14*z - 32.0);val.col(1) = result_0; }{result_0 = x*y*(127.99999999999997*y + 5.6843418860808015e-14*z - 32.0);val.col(2) = result_0; }} break;
	case 31: {{result_0 = y*z*(4.2632564145606011e-14*x - 127.99999999999996*y + 3.5527136788005009e-14*z + 31.999999999999957);val.col(0) = result_0; }{result_0 = z*(2.1316282072803006e-14*pow(x, 2) - 255.99999999999989*x*y + 3.5527136788005009e-14*x*z + 31.999999999999957*x - 383.99999999999989*pow(y, 2) - 255.99999999999991*y*z + 319.99999999999989*y + 1.4210854715202004e-14*pow(z, 2) + 31.999999999999964*z - 31.999999999999979);val.col(1) = result_0; }{result_0 = y*(2.1316282072803006e-14*pow(x, 2) - 127.99999999999996*x*y + 7.1054273576010019e-14*x*z + 31.999999999999957*x - 127.99999999999997*pow(y, 2) - 255.99999999999991*y*z + 159.99999999999994*y + 4.2632564145606011e-14*pow(z, 2) + 63.999999999999929*z - 31.999999999999979);val.col(2) = result_0; }} break;
	case 32: {{result_0 = y*z*(1.4210854715202004e-14*x + 2.8421709430404007e-14*y - 127.99999999999997*z + 31.999999999999986);val.col(0) = result_0; }{result_0 = z*(7.1054273576010019e-15*pow(x, 2) + 5.6843418860808015e-14*x*y - 127.99999999999997*x*z + 31.999999999999986*x + 6.3948846218409017e-14*pow(y, 2) - 255.99999999999994*y*z + 63.999999999999943*y - 127.99999999999997*pow(z, 2) + 159.99999999999997*z - 31.999999999999993);val.col(1) = result_0; }{result_0 = y*(7.1054273576010019e-15*pow(x, 2) + 2.8421709430404007e-14*x*y - 255.99999999999994*x*z + 31.999999999999986*x + 2.1316282072803006e-14*pow(y, 2) - 255.99999999999989*y*z + 31.999999999999972*y - 383.99999999999989*pow(z, 2) + 319.99999999999994*z - 31.999999999999993);val.col(2) = result_0; }} break;
	case 33: {{result_0 = y*z*(256.00000000000006*x + 256.00000000000006*y + 256.00000000000006*z - 224.00000000000006);val.col(0) = result_0; }{const auto helper_0 = 512.00000000000023*y;
result_0 = z*(helper_0*x + helper_0*z + 128.00000000000003*pow(x, 2) + 256.00000000000006*x*z - 224.00000000000006*x + 384.00000000000017*pow(y, 2) - 448.00000000000017*y + 128.00000000000006*pow(z, 2) - 224.00000000000006*z + 96.000000000000028);val.col(1) = result_0; }{const auto helper_0 = 512.00000000000023*z;
result_0 = y*(helper_0*x + helper_0*y + 128.00000000000003*pow(x, 2) + 256.00000000000006*x*y - 224.00000000000006*x + 128.00000000000006*pow(y, 2) - 224.00000000000006*y + 384.00000000000017*pow(z, 2) - 448.00000000000017*z + 96.000000000000028);val.col(2) = result_0; }} break;
	case 34: {{result_0 = -y*z*(512.0*x + 255.99999999999997*y + 255.99999999999997*z - 256.0);val.col(0) = result_0; }{result_0 = -z*(256.0*pow(x, 2) + 511.99999999999994*x*y + 255.99999999999997*x*z - 256.0*x - 1.0437238884603262e-13*pow(y, 2) - 1.2977813736246076e-13*y*z + 6.9581592564021748e-14*y - 3.0098272399219508e-14*pow(z, 2) + 3.0098272399219508e-14*z);val.col(1) = result_0; }{result_0 = -y*(256.0*pow(x, 2) + 255.99999999999997*x*y + 511.99999999999994*x*z - 256.0*x - 3.4790796282010874e-14*pow(y, 2) - 1.2977813736246076e-13*y*z + 3.4790796282010874e-14*y - 9.0294817197658525e-14*pow(z, 2) + 6.0196544798439017e-14*z);val.col(2) = result_0; }} break;
	default: assert(false);
}}


void p_4_nodes_3d(Eigen::MatrixXd &res) {
 res.resize(35, 3); res << 
0, 0, 0,
1, 0, 0,
0, 1, 0,
0, 0, 1,
1.0/4.0, 0, 0,
1.0/2.0, 0, 0,
3.0/4.0, 0, 0,
3.0/4.0, 1.0/4.0, 0,
1.0/2.0, 1.0/2.0, 0,
1.0/4.0, 3.0/4.0, 0,
0, 3.0/4.0, 0,
0, 1.0/2.0, 0,
0, 1.0/4.0, 0,
0, 0, 1.0/4.0,
0, 0, 1.0/2.0,
0, 0, 3.0/4.0,
3.0/4.0, 0, 1.0/4.0,
1.0/2.0, 0, 1.0/2.0,
1.0/4.0, 0, 3.0/4.0,
0, 3.0/4.0, 1.0/4.0,
0, 1.0/2.0, 1.0/2.0,
0, 1.0/4.0, 3.0/4.0,
1.0/4.0, 1.0/4.0, 0,
1.0/4.0, 1.0/2.0, 0,
1.0/2.0, 1.0/4.0, 0,
1.0/4.0, 0, 1.0/4.0,
1.0/4.0, 0, 1.0/2.0,
1.0/2.0, 0, 1.0/4.0,
1.0/2.0, 1.0/4.0, 1.0/4.0,
1.0/4.0, 1.0/4.0, 1.0/2.0,
1.0/4.0, 1.0/2.0, 1.0/4.0,
0, 1.0/2.0, 1.0/4.0,
0, 1.0/4.0, 1.0/2.0,
0, 1.0/4.0, 1.0/4.0,
1.0/4.0, 1.0/4.0, 1.0/4.0;
}


}

void p_nodes_3d(const int p, Eigen::MatrixXd &val){
switch(p){
	case 0: p_0_nodes_3d(val); break;
	case 1: p_1_nodes_3d(val); break;
	case 2: p_2_nodes_3d(val); break;
	case 3: p_3_nodes_3d(val); break;
	case 4: p_4_nodes_3d(val); break;
	default: p_n_nodes_3d(p, val);
}}
void p_basis_value_3d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(p){
	case 0: p_0_basis_value_3d(local_index, uv, val); break;
	case 1: p_1_basis_value_3d(local_index, uv, val); break;
	case 2: p_2_basis_value_3d(local_index, uv, val); break;
	case 3: p_3_basis_value_3d(local_index, uv, val); break;
	case 4: p_4_basis_value_3d(local_index, uv, val); break;
	default: p_n_basis_value_3d(p, local_index, uv, val);
}}

void p_grad_basis_value_3d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(p){
	case 0: p_0_basis_grad_value_3d(local_index, uv, val); break;
	case 1: p_1_basis_grad_value_3d(local_index, uv, val); break;
	case 2: p_2_basis_grad_value_3d(local_index, uv, val); break;
	case 3: p_3_basis_grad_value_3d(local_index, uv, val); break;
	case 4: p_4_basis_grad_value_3d(local_index, uv, val); break;
	default: p_n_basis_grad_value_3d(p, local_index, uv, val);
}}

namespace {

}}}
