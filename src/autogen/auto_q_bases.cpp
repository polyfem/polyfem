#include <polyfem/auto_q_bases.hpp>


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
void q_0_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }} break;
	default: assert(false);
}}


void q_0_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(1, 2); res << 
0.5, 0.5;
}


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
void q_1_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = 1.0*(y - 1);val.col(0) = result_0; }{result_0 = 1.0*(x - 1);val.col(1) = result_0; }} break;
	case 1: {{result_0 = 1.0*(-y + 1);val.col(0) = result_0; }{result_0 = -1.0*x;val.col(1) = result_0; }} break;
	case 2: {{result_0 = 1.0*y;val.col(0) = result_0; }{result_0 = 1.0*x;val.col(1) = result_0; }} break;
	case 3: {{result_0 = -1.0*y;val.col(0) = result_0; }{result_0 = 1.0*(-x + 1);val.col(1) = result_0; }} break;
	default: assert(false);
}}


void q_1_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(4, 2); res << 
0, 0,
1, 0,
1, 1,
0, 1;
}


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


void q_2_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(9, 2); res << 
0, 0,
1, 0,
1, 1,
0, 1,
1.0/2.0, 0,
1, 1.0/2.0,
1.0/2.0, 1,
0, 1.0/2.0,
1.0/2.0, 1.0/2.0;
}


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
void q_m2_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = -(y - 1)*(4.0*x + 2.0*y - 3.0);val.col(0) = result_0; }{const auto helper_0 = 4.0*y;
result_0 = -helper_0*x + helper_0 - 2.0*pow(x, 2) + 5.0*x - 3.0;val.col(1) = result_0; }} break;
	case 1: {{result_0 = (y - 1)*(-4.0*x + 2.0*y + 1.0);val.col(0) = result_0; }{result_0 = -x*(2.0*x - 4.0*y + 1.0);val.col(1) = result_0; }} break;
	case 2: {{result_0 = y*(4.0*x + 2.0*y - 3.0);val.col(0) = result_0; }{result_0 = x*(2.0*x + 4.0*y - 3.0);val.col(1) = result_0; }} break;
	case 3: {{result_0 = -y*(-4.0*x + 2.0*y + 1.0);val.col(0) = result_0; }{result_0 = (x - 1)*(2.0*x - 4.0*y + 1.0);val.col(1) = result_0; }} break;
	case 4: {{result_0 = 4*(2*x - 1)*(y - 1);val.col(0) = result_0; }{result_0 = 4*x*(x - 1);val.col(1) = result_0; }} break;
	case 5: {{result_0 = -4*y*(y - 1);val.col(0) = result_0; }{result_0 = -4*x*(2*y - 1);val.col(1) = result_0; }} break;
	case 6: {{result_0 = -4*y*(2*x - 1);val.col(0) = result_0; }{result_0 = -4*x*(x - 1);val.col(1) = result_0; }} break;
	case 7: {{result_0 = 4*y*(y - 1);val.col(0) = result_0; }{result_0 = 4*(x - 1)*(2*y - 1);val.col(1) = result_0; }} break;
	default: assert(false);
}}


void q_m2_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(8, 2); res << 
0.0, 0.0,
1.0, 0.0,
1.0, 1.0,
0.0, 1.0,
0.5, 0.0,
1.0, 0.5,
0.5, 1.0,
0.0, 0.5;
}


}

void q_nodes_2d(const int q, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_nodes_2d(val); break;
	case 1: q_1_nodes_2d(val); break;
	case 2: q_2_nodes_2d(val); break;
	case -2: q_m2_nodes_2d(val); break;
	default: assert(false);
}}
void q_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_basis_value_2d(local_index, uv, val); break;
	case 1: q_1_basis_value_2d(local_index, uv, val); break;
	case 2: q_2_basis_value_2d(local_index, uv, val); break;
	case -2: q_m2_basis_value_2d(local_index, uv, val); break;
	default: assert(false);
}}

void q_grad_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_basis_grad_value_2d(local_index, uv, val); break;
	case 1: q_1_basis_grad_value_2d(local_index, uv, val); break;
	case 2: q_2_basis_grad_value_2d(local_index, uv, val); break;
	case -2: q_m2_basis_grad_value_2d(local_index, uv, val); break;
	default: assert(false);
}}

namespace {
void q_0_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

result_0.resize(x.size(),1);
switch(local_index){
	case 0: {result_0.setOnes();} break;
	default: assert(false);
}}
void q_0_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	default: assert(false);
}}


void q_0_nodes_3d(Eigen::MatrixXd &res) {
 res.resize(1, 3); res << 
0.5, 0.5, 0.5;
}


void q_1_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

switch(local_index){
	case 0: {result_0 = -1.0*(x - 1)*(y - 1)*(z - 1);} break;
	case 1: {result_0 = 1.0*x*(y - 1)*(z - 1);} break;
	case 2: {result_0 = -1.0*x*y*(z - 1);} break;
	case 3: {result_0 = 1.0*y*(x - 1)*(z - 1);} break;
	case 4: {result_0 = 1.0*z*(x - 1)*(y - 1);} break;
	case 5: {result_0 = -1.0*x*z*(y - 1);} break;
	case 6: {result_0 = 1.0*x*y*z;} break;
	case 7: {result_0 = -1.0*y*z*(x - 1);} break;
	default: assert(false);
}}
void q_1_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

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


void q_1_nodes_3d(Eigen::MatrixXd &res) {
 res.resize(8, 3); res << 
0, 0, 0,
1, 0, 0,
1, 1, 0,
0, 1, 0,
0, 0, 1,
1, 0, 1,
1, 1, 1,
0, 1, 1;
}


void q_2_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

switch(local_index){
	case 0: {result_0 = 1.0*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);} break;
	case 1: {result_0 = 1.0*x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);} break;
	case 2: {result_0 = 1.0*x*y*(2.0*x - 1.0)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);} break;
	case 3: {result_0 = 1.0*y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);} break;
	case 4: {result_0 = 1.0*z*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);} break;
	case 5: {result_0 = 1.0*x*z*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);} break;
	case 6: {result_0 = 1.0*x*y*z*(2.0*x - 1.0)*(2.0*y - 1.0)*(2.0*z - 1.0);} break;
	case 7: {result_0 = 1.0*y*z*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(2.0*z - 1.0);} break;
	case 8: {result_0 = -4.0*x*(x - 1)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);} break;
	case 9: {result_0 = -4.0*x*y*(2.0*x - 1.0)*(y - 1)*(z - 1)*(2.0*z - 1.0);} break;
	case 10: {result_0 = -4.0*x*y*(x - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);} break;
	case 11: {result_0 = -4.0*y*(x - 1)*(2.0*x - 1.0)*(y - 1)*(z - 1)*(2.0*z - 1.0);} break;
	case 12: {result_0 = -4.0*z*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(z - 1);} break;
	case 13: {result_0 = -4.0*x*z*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(z - 1);} break;
	case 14: {result_0 = -4.0*x*y*z*(2.0*x - 1.0)*(2.0*y - 1.0)*(z - 1);} break;
	case 15: {result_0 = -4.0*y*z*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(z - 1);} break;
	case 16: {result_0 = -4.0*x*z*(x - 1)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);} break;
	case 17: {result_0 = -4.0*x*y*z*(2.0*x - 1.0)*(y - 1)*(2.0*z - 1.0);} break;
	case 18: {result_0 = -4.0*x*y*z*(x - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);} break;
	case 19: {result_0 = -4.0*y*z*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*z - 1.0);} break;
	case 20: {result_0 = 16.0*y*z*(x - 1)*(2.0*x - 1.0)*(y - 1)*(z - 1);} break;
	case 21: {result_0 = 16.0*x*y*z*(2.0*x - 1.0)*(y - 1)*(z - 1);} break;
	case 22: {result_0 = 16.0*x*z*(x - 1)*(y - 1)*(2.0*y - 1.0)*(z - 1);} break;
	case 23: {result_0 = 16.0*x*y*z*(x - 1)*(2.0*y - 1.0)*(z - 1);} break;
	case 24: {result_0 = 16.0*x*y*(x - 1)*(y - 1)*(z - 1)*(2.0*z - 1.0);} break;
	case 25: {result_0 = 16.0*x*y*z*(x - 1)*(y - 1)*(2.0*z - 1.0);} break;
	case 26: {result_0 = -64.0*x*y*z*(x - 1)*(y - 1)*(z - 1);} break;
	default: assert(false);
}}
void q_2_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

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


void q_2_nodes_3d(Eigen::MatrixXd &res) {
 res.resize(27, 3); res << 
0, 0, 0,
1, 0, 0,
1, 1, 0,
0, 1, 0,
0, 0, 1,
1, 0, 1,
1, 1, 1,
0, 1, 1,
1.0/2.0, 0, 0,
1, 1.0/2.0, 0,
1.0/2.0, 1, 0,
0, 1.0/2.0, 0,
0, 0, 1.0/2.0,
1, 0, 1.0/2.0,
1, 1, 1.0/2.0,
0, 1, 1.0/2.0,
1.0/2.0, 0, 1,
1, 1.0/2.0, 1,
1.0/2.0, 1, 1,
0, 1.0/2.0, 1,
0, 1.0/2.0, 1.0/2.0,
1, 1.0/2.0, 1.0/2.0,
1.0/2.0, 0, 1.0/2.0,
1.0/2.0, 1, 1.0/2.0,
1.0/2.0, 1.0/2.0, 0,
1.0/2.0, 1.0/2.0, 1,
1.0/2.0, 1.0/2.0, 1.0/2.0;
}


void q_m2_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

switch(local_index){
	case 0: {result_0 = 1.0*(x - 1)*(y - 1)*(z - 1)*(2*x + 2*y + 2*z - 1);} break;
	case 1: {result_0 = -1.0*x*(y - 1)*(z - 1)*(-2*x + 2*y + 2*z + 1);} break;
	case 2: {result_0 = -1.0*x*y*(z - 1)*(2*x + 2*y - 2*z - 3);} break;
	case 3: {result_0 = -1.0*y*(x - 1)*(z - 1)*(2*x - 2*y + 2*z + 1);} break;
	case 4: {result_0 = -1.0*z*(x - 1)*(y - 1)*(2*x + 2*y - 2*z + 1);} break;
	case 5: {result_0 = -1.0*x*z*(y - 1)*(2*x - 2*y + 2*z - 3);} break;
	case 6: {result_0 = x*y*z*(2.0*x + 2.0*y + 2.0*z - 5.0);} break;
	case 7: {result_0 = 1.0*y*z*(x - 1)*(2*x - 2*y - 2*z + 3);} break;
	case 8: {result_0 = -4*x*(x - 1)*(y - 1)*(z - 1);} break;
	case 9: {result_0 = 4*x*y*(y - 1)*(z - 1);} break;
	case 10: {result_0 = 4*x*y*(x - 1)*(z - 1);} break;
	case 11: {result_0 = -4*y*(x - 1)*(y - 1)*(z - 1);} break;
	case 12: {result_0 = -4*z*(x - 1)*(y - 1)*(z - 1);} break;
	case 13: {result_0 = 4*x*z*(y - 1)*(z - 1);} break;
	case 14: {result_0 = -4*x*y*z*(z - 1);} break;
	case 15: {result_0 = 4*y*z*(x - 1)*(z - 1);} break;
	case 16: {result_0 = 4*x*z*(x - 1)*(y - 1);} break;
	case 17: {result_0 = -4*x*y*z*(y - 1);} break;
	case 18: {result_0 = -4*x*y*z*(x - 1);} break;
	case 19: {result_0 = 4*y*z*(x - 1)*(y - 1);} break;
	default: assert(false);
}}
void q_m2_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = (y - 1)*(z - 1)*(4.0*x + 2*y + 2*z - 3.0);val.col(0) = result_0; }{result_0 = (x - 1)*(z - 1)*(2.0*x + 4.0*y + 2.0*z - 3.0);val.col(1) = result_0; }{result_0 = (x - 1)*(y - 1)*(2.0*x + 2.0*y + 4.0*z - 3.0);val.col(2) = result_0; }} break;
	case 1: {{result_0 = -(y - 1)*(z - 1)*(-4.0*x + 2.0*y + 2.0*z + 1.0);val.col(0) = result_0; }{result_0 = x*(z - 1)*(2.0*x - 4.0*y - 2.0*z + 1.0);val.col(1) = result_0; }{result_0 = x*(y - 1)*(2.0*x - 2.0*y - 4.0*z + 1.0);val.col(2) = result_0; }} break;
	case 2: {{result_0 = -y*(z - 1)*(4.0*x + 2.0*y - 2.0*z - 3.0);val.col(0) = result_0; }{result_0 = -x*(z - 1)*(2.0*x + 4.0*y - 2.0*z - 3.0);val.col(1) = result_0; }{result_0 = -x*y*(2.0*x + 2.0*y - 4.0*z - 1.0);val.col(2) = result_0; }} break;
	case 3: {{result_0 = -y*(z - 1)*(4.0*x - 2.0*y + 2.0*z - 1.0);val.col(0) = result_0; }{result_0 = -(x - 1)*(z - 1)*(2.0*x - 4.0*y + 2.0*z + 1.0);val.col(1) = result_0; }{result_0 = -y*(x - 1)*(2.0*x - 2.0*y + 4.0*z - 1.0);val.col(2) = result_0; }} break;
	case 4: {{result_0 = -z*(y - 1)*(4.0*x + 2.0*y - 2.0*z - 1.0);val.col(0) = result_0; }{result_0 = -z*(x - 1)*(2.0*x + 4.0*y - 2.0*z - 1.0);val.col(1) = result_0; }{result_0 = -(x - 1)*(y - 1)*(2.0*x + 2.0*y - 4.0*z + 1.0);val.col(2) = result_0; }} break;
	case 5: {{result_0 = -z*(y - 1)*(4.0*x - 2.0*y + 2.0*z - 3.0);val.col(0) = result_0; }{result_0 = -x*z*(2.0*x - 4.0*y + 2.0*z - 1.0);val.col(1) = result_0; }{result_0 = -x*(y - 1)*(2.0*x - 2.0*y + 4.0*z - 3.0);val.col(2) = result_0; }} break;
	case 6: {{result_0 = y*z*(4.0*x + 2.0*y + 2.0*z - 5.0);val.col(0) = result_0; }{result_0 = x*z*(2.0*x + 4.0*y + 2.0*z - 5.0);val.col(1) = result_0; }{result_0 = x*y*(2.0*x + 2.0*y + 4.0*z - 5.0);val.col(2) = result_0; }} break;
	case 7: {{result_0 = y*z*(4.0*x - 2.0*y - 2.0*z + 1.0);val.col(0) = result_0; }{result_0 = z*(x - 1)*(2.0*x - 4.0*y - 2.0*z + 3.0);val.col(1) = result_0; }{result_0 = y*(x - 1)*(2.0*x - 2.0*y - 4.0*z + 3.0);val.col(2) = result_0; }} break;
	case 8: {{result_0 = -4*(2*x - 1)*(y - 1)*(z - 1);val.col(0) = result_0; }{result_0 = -4*x*(x - 1)*(z - 1);val.col(1) = result_0; }{result_0 = -4*x*(x - 1)*(y - 1);val.col(2) = result_0; }} break;
	case 9: {{result_0 = 4*y*(y - 1)*(z - 1);val.col(0) = result_0; }{result_0 = 4*x*(2*y - 1)*(z - 1);val.col(1) = result_0; }{result_0 = 4*x*y*(y - 1);val.col(2) = result_0; }} break;
	case 10: {{result_0 = 4*y*(2*x - 1)*(z - 1);val.col(0) = result_0; }{result_0 = 4*x*(x - 1)*(z - 1);val.col(1) = result_0; }{result_0 = 4*x*y*(x - 1);val.col(2) = result_0; }} break;
	case 11: {{result_0 = -4*y*(y - 1)*(z - 1);val.col(0) = result_0; }{result_0 = -4*(x - 1)*(2*y - 1)*(z - 1);val.col(1) = result_0; }{result_0 = -4*y*(x - 1)*(y - 1);val.col(2) = result_0; }} break;
	case 12: {{result_0 = -4*z*(y - 1)*(z - 1);val.col(0) = result_0; }{result_0 = -4*z*(x - 1)*(z - 1);val.col(1) = result_0; }{result_0 = -4*(x - 1)*(y - 1)*(2*z - 1);val.col(2) = result_0; }} break;
	case 13: {{result_0 = 4*z*(y - 1)*(z - 1);val.col(0) = result_0; }{result_0 = 4*x*z*(z - 1);val.col(1) = result_0; }{result_0 = 4*x*(y - 1)*(2*z - 1);val.col(2) = result_0; }} break;
	case 14: {{result_0 = -4*y*z*(z - 1);val.col(0) = result_0; }{result_0 = -4*x*z*(z - 1);val.col(1) = result_0; }{result_0 = -4*x*y*(2*z - 1);val.col(2) = result_0; }} break;
	case 15: {{result_0 = 4*y*z*(z - 1);val.col(0) = result_0; }{result_0 = 4*z*(x - 1)*(z - 1);val.col(1) = result_0; }{result_0 = 4*y*(x - 1)*(2*z - 1);val.col(2) = result_0; }} break;
	case 16: {{result_0 = 4*z*(2*x - 1)*(y - 1);val.col(0) = result_0; }{result_0 = 4*x*z*(x - 1);val.col(1) = result_0; }{result_0 = 4*x*(x - 1)*(y - 1);val.col(2) = result_0; }} break;
	case 17: {{result_0 = -4*y*z*(y - 1);val.col(0) = result_0; }{result_0 = -4*x*z*(2*y - 1);val.col(1) = result_0; }{result_0 = -4*x*y*(y - 1);val.col(2) = result_0; }} break;
	case 18: {{result_0 = -4*y*z*(2*x - 1);val.col(0) = result_0; }{result_0 = -4*x*z*(x - 1);val.col(1) = result_0; }{result_0 = -4*x*y*(x - 1);val.col(2) = result_0; }} break;
	case 19: {{result_0 = 4*y*z*(y - 1);val.col(0) = result_0; }{result_0 = 4*z*(x - 1)*(2*y - 1);val.col(1) = result_0; }{result_0 = 4*y*(x - 1)*(y - 1);val.col(2) = result_0; }} break;
	default: assert(false);
}}


void q_m2_nodes_3d(Eigen::MatrixXd &res) {
 res.resize(20, 3); res << 
0.0, 0.0, 0.0,
1.0, 0.0, 0.0,
1.0, 1.0, 0.0,
0.0, 1.0, 0.0,
0.0, 0.0, 1.0,
1.0, 0.0, 1.0,
1.0, 1.0, 1.0,
0.0, 1.0, 1.0,
0.5, 0.0, 0.0,
1.0, 0.5, 0.0,
0.5, 1.0, 0.0,
0.0, 0.5, 0.0,
0.0, 0.0, 0.5,
1.0, 0.0, 0.5,
1.0, 1.0, 0.5,
0.0, 1.0, 0.5,
0.5, 0.0, 1.0,
1.0, 0.5, 1.0,
0.5, 1.0, 1.0,
0.0, 0.5, 1.0;
}


}

void q_nodes_3d(const int q, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_nodes_3d(val); break;
	case 1: q_1_nodes_3d(val); break;
	case 2: q_2_nodes_3d(val); break;
	case -2: q_m2_nodes_3d(val); break;
	default: assert(false);
}}
void q_basis_value_3d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_basis_value_3d(local_index, uv, val); break;
	case 1: q_1_basis_value_3d(local_index, uv, val); break;
	case 2: q_2_basis_value_3d(local_index, uv, val); break;
	case -2: q_m2_basis_value_3d(local_index, uv, val); break;
	default: assert(false);
}}

void q_grad_basis_value_3d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_basis_grad_value_3d(local_index, uv, val); break;
	case 1: q_1_basis_grad_value_3d(local_index, uv, val); break;
	case 2: q_2_basis_grad_value_3d(local_index, uv, val); break;
	case -2: q_m2_basis_grad_value_3d(local_index, uv, val); break;
	default: assert(false);
}}

namespace {

}}}
