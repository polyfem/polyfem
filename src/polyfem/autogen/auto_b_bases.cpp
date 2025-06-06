#include "auto_b_bases.hpp"


namespace polyfem {
namespace autogen {
namespace {
void b_0_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

result_0.resize(x.size(),1);
switch(local_index){
	case 0: {result_0.setOnes();} break;
	default: assert(false);
}}
void b_0_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }} break;
	default: assert(false);
}}


void b_1_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

switch(local_index){
	case 0: {result_0 = -x - y + 1;} break;
	case 1: {result_0 = x;} break;
	case 2: {result_0 = y;} break;
	default: assert(false);
}}
void b_1_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

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


void b_2_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

switch(local_index){
	case 0: {result_0 = pow(x + y - 1, 2);} break;
	case 1: {result_0 = pow(x, 2);} break;
	case 2: {result_0 = pow(y, 2);} break;
	case 3: {result_0 = -2*x*(x + y - 1);} break;
	case 4: {result_0 = 2*x*y;} break;
	case 5: {result_0 = -2*y*(x + y - 1);} break;
	default: assert(false);
}}
void b_2_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = 2*(x + y - 1);val.col(0) = result_0; }{result_0 = 2*(x + y - 1);val.col(1) = result_0; }} break;
	case 1: {{result_0 = 2*x;val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }} break;
	case 2: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 2*y;val.col(1) = result_0; }} break;
	case 3: {{result_0 = 2*(-2*x - y + 1);val.col(0) = result_0; }{result_0 = -2*x;val.col(1) = result_0; }} break;
	case 4: {{result_0 = 2*y;val.col(0) = result_0; }{result_0 = 2*x;val.col(1) = result_0; }} break;
	case 5: {{result_0 = -2*y;val.col(0) = result_0; }{result_0 = 2*(-x - 2*y + 1);val.col(1) = result_0; }} break;
	default: assert(false);
}}


void b_3_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

switch(local_index){
	case 0: {result_0 = -pow(x + y - 1, 3);} break;
	case 1: {result_0 = pow(x, 3);} break;
	case 2: {result_0 = pow(y, 3);} break;
	case 3: {result_0 = 3*x*pow(x + y - 1, 2);} break;
	case 4: {result_0 = -3*pow(x, 2)*(x + y - 1);} break;
	case 5: {result_0 = 3*pow(x, 2)*y;} break;
	case 6: {result_0 = 3*x*pow(y, 2);} break;
	case 7: {result_0 = -3*pow(y, 2)*(x + y - 1);} break;
	case 8: {result_0 = 3*y*pow(x + y - 1, 2);} break;
	case 9: {result_0 = -6*x*y*(x + y - 1);} break;
	default: assert(false);
}}
void b_3_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = -3*pow(x + y - 1, 2);val.col(0) = result_0; }{result_0 = -3*pow(x + y - 1, 2);val.col(1) = result_0; }} break;
	case 1: {{result_0 = 3*pow(x, 2);val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }} break;
	case 2: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 3*pow(y, 2);val.col(1) = result_0; }} break;
	case 3: {{const auto helper_0 = y - 1;
result_0 = 3*(helper_0 + x)*(helper_0 + 3*x);val.col(0) = result_0; }{result_0 = 6*x*(x + y - 1);val.col(1) = result_0; }} break;
	case 4: {{const auto helper_0 = 3*x;
result_0 = -helper_0*(helper_0 + 2*y - 2);val.col(0) = result_0; }{result_0 = -3*pow(x, 2);val.col(1) = result_0; }} break;
	case 5: {{result_0 = 6*x*y;val.col(0) = result_0; }{result_0 = 3*pow(x, 2);val.col(1) = result_0; }} break;
	case 6: {{result_0 = 3*pow(y, 2);val.col(0) = result_0; }{result_0 = 6*x*y;val.col(1) = result_0; }} break;
	case 7: {{result_0 = -3*pow(y, 2);val.col(0) = result_0; }{const auto helper_0 = 3*y;
result_0 = -helper_0*(helper_0 + 2*x - 2);val.col(1) = result_0; }} break;
	case 8: {{result_0 = 6*y*(x + y - 1);val.col(0) = result_0; }{const auto helper_0 = x - 1;
result_0 = 3*(helper_0 + y)*(helper_0 + 3*y);val.col(1) = result_0; }} break;
	case 9: {{result_0 = -6*y*(2*x + y - 1);val.col(0) = result_0; }{result_0 = -6*x*(x + 2*y - 1);val.col(1) = result_0; }} break;
	default: assert(false);
}}


void b_4_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

switch(local_index){
	case 0: {result_0 = pow(x + y - 1, 4);} break;
	case 1: {result_0 = pow(x, 4);} break;
	case 2: {result_0 = pow(y, 4);} break;
	case 3: {result_0 = -4*x*pow(x + y - 1, 3);} break;
	case 4: {result_0 = 6*pow(x, 2)*pow(x + y - 1, 2);} break;
	case 5: {result_0 = -4*pow(x, 3)*(x + y - 1);} break;
	case 6: {result_0 = 4*pow(x, 3)*y;} break;
	case 7: {result_0 = 6*pow(x, 2)*pow(y, 2);} break;
	case 8: {result_0 = 4*x*pow(y, 3);} break;
	case 9: {result_0 = -4*pow(y, 3)*(x + y - 1);} break;
	case 10: {result_0 = 6*pow(y, 2)*pow(x + y - 1, 2);} break;
	case 11: {result_0 = -4*y*pow(x + y - 1, 3);} break;
	case 12: {result_0 = 12*x*y*pow(x + y - 1, 2);} break;
	case 13: {result_0 = -12*x*pow(y, 2)*(x + y - 1);} break;
	case 14: {result_0 = -12*pow(x, 2)*y*(x + y - 1);} break;
	default: assert(false);
}}
void b_4_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = 4*pow(x + y - 1, 3);val.col(0) = result_0; }{result_0 = 4*pow(x + y - 1, 3);val.col(1) = result_0; }} break;
	case 1: {{result_0 = 4*pow(x, 3);val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }} break;
	case 2: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 4*pow(y, 3);val.col(1) = result_0; }} break;
	case 3: {{const auto helper_0 = y - 1;
result_0 = -4*pow(helper_0 + x, 2)*(helper_0 + 4*x);val.col(0) = result_0; }{result_0 = -12*x*pow(x + y - 1, 2);val.col(1) = result_0; }} break;
	case 4: {{const auto helper_0 = y - 1;
result_0 = 12*x*(helper_0 + x)*(helper_0 + 2*x);val.col(0) = result_0; }{result_0 = 12*pow(x, 2)*(x + y - 1);val.col(1) = result_0; }} break;
	case 5: {{result_0 = -4*pow(x, 2)*(4*x + 3*y - 3);val.col(0) = result_0; }{result_0 = -4*pow(x, 3);val.col(1) = result_0; }} break;
	case 6: {{result_0 = 12*pow(x, 2)*y;val.col(0) = result_0; }{result_0 = 4*pow(x, 3);val.col(1) = result_0; }} break;
	case 7: {{result_0 = 12*x*pow(y, 2);val.col(0) = result_0; }{result_0 = 12*pow(x, 2)*y;val.col(1) = result_0; }} break;
	case 8: {{result_0 = 4*pow(y, 3);val.col(0) = result_0; }{result_0 = 12*x*pow(y, 2);val.col(1) = result_0; }} break;
	case 9: {{result_0 = -4*pow(y, 3);val.col(0) = result_0; }{result_0 = -4*pow(y, 2)*(3*x + 4*y - 3);val.col(1) = result_0; }} break;
	case 10: {{result_0 = 12*pow(y, 2)*(x + y - 1);val.col(0) = result_0; }{const auto helper_0 = x - 1;
result_0 = 12*y*(helper_0 + y)*(helper_0 + 2*y);val.col(1) = result_0; }} break;
	case 11: {{result_0 = -12*y*pow(x + y - 1, 2);val.col(0) = result_0; }{const auto helper_0 = x - 1;
result_0 = -4*pow(helper_0 + y, 2)*(helper_0 + 4*y);val.col(1) = result_0; }} break;
	case 12: {{const auto helper_0 = y - 1;
result_0 = 12*y*(helper_0 + x)*(helper_0 + 3*x);val.col(0) = result_0; }{const auto helper_0 = x - 1;
result_0 = 12*x*(helper_0 + y)*(helper_0 + 3*y);val.col(1) = result_0; }} break;
	case 13: {{result_0 = -12*pow(y, 2)*(2*x + y - 1);val.col(0) = result_0; }{result_0 = -12*x*y*(2*x + 3*y - 2);val.col(1) = result_0; }} break;
	case 14: {{result_0 = -12*x*y*(3*x + 2*y - 2);val.col(0) = result_0; }{result_0 = -12*pow(x, 2)*(x + 2*y - 1);val.col(1) = result_0; }} break;
	default: assert(false);
}}


}


void b_basis_value_2d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

switch(b){
	case 0: b_0_basis_value_2d(local_index, uv, val); break;
	case 1: b_1_basis_value_2d(local_index, uv, val); break;
	case 2: b_2_basis_value_2d(local_index, uv, val); break;
	case 3: b_3_basis_value_2d(local_index, uv, val); break;
	case 4: b_4_basis_value_2d(local_index, uv, val); break;
	default: assert(false); 
}}

void b_grad_basis_value_2d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

switch(b){
	case 0: b_0_basis_grad_value_2d(local_index, uv, val); break;
	case 1: b_1_basis_grad_value_2d(local_index, uv, val); break;
	case 2: b_2_basis_grad_value_2d(local_index, uv, val); break;
	case 3: b_3_basis_grad_value_2d(local_index, uv, val); break;
	case 4: b_4_basis_grad_value_2d(local_index, uv, val); break;
	default: assert(false); 
}}

namespace {
void b_0_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

result_0.resize(x.size(),1);
switch(local_index){
	case 0: {result_0.setOnes();} break;
	default: assert(false);
}}
void b_0_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	default: assert(false);
}}


void b_1_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

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
void b_1_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

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


void b_2_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

switch(local_index){
	case 0: {result_0 = pow(x + y + z - 1, 2);} break;
	case 1: {result_0 = pow(x, 2);} break;
	case 2: {result_0 = pow(y, 2);} break;
	case 3: {result_0 = pow(z, 2);} break;
	case 4: {result_0 = -2*x*(x + y + z - 1);} break;
	case 5: {result_0 = 2*x*y;} break;
	case 6: {result_0 = -2*y*(x + y + z - 1);} break;
	case 7: {result_0 = -2*z*(x + y + z - 1);} break;
	case 8: {result_0 = 2*x*z;} break;
	case 9: {result_0 = 2*y*z;} break;
	default: assert(false);
}}
void b_2_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = 2*(x + y + z - 1);val.col(0) = result_0; }{result_0 = 2*(x + y + z - 1);val.col(1) = result_0; }{result_0 = 2*(x + y + z - 1);val.col(2) = result_0; }} break;
	case 1: {{result_0 = 2*x;val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 2: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 2*y;val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 3: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = 2*z;val.col(2) = result_0; }} break;
	case 4: {{result_0 = 2*(-2*x - y - z + 1);val.col(0) = result_0; }{result_0 = -2*x;val.col(1) = result_0; }{result_0 = -2*x;val.col(2) = result_0; }} break;
	case 5: {{result_0 = 2*y;val.col(0) = result_0; }{result_0 = 2*x;val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 6: {{result_0 = -2*y;val.col(0) = result_0; }{result_0 = 2*(-x - 2*y - z + 1);val.col(1) = result_0; }{result_0 = -2*y;val.col(2) = result_0; }} break;
	case 7: {{result_0 = -2*z;val.col(0) = result_0; }{result_0 = -2*z;val.col(1) = result_0; }{result_0 = 2*(-x - y - 2*z + 1);val.col(2) = result_0; }} break;
	case 8: {{result_0 = 2*z;val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = 2*x;val.col(2) = result_0; }} break;
	case 9: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 2*z;val.col(1) = result_0; }{result_0 = 2*y;val.col(2) = result_0; }} break;
	default: assert(false);
}}


void b_3_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

switch(local_index){
	case 0: {result_0 = -pow(x + y + z - 1, 3);} break;
	case 1: {result_0 = pow(x, 3);} break;
	case 2: {result_0 = pow(y, 3);} break;
	case 3: {result_0 = pow(z, 3);} break;
	case 4: {result_0 = 3*x*pow(x + y + z - 1, 2);} break;
	case 5: {result_0 = -3*pow(x, 2)*(x + y + z - 1);} break;
	case 6: {result_0 = 3*pow(x, 2)*y;} break;
	case 7: {result_0 = 3*x*pow(y, 2);} break;
	case 8: {result_0 = -3*pow(y, 2)*(x + y + z - 1);} break;
	case 9: {result_0 = 3*y*pow(x + y + z - 1, 2);} break;
	case 10: {result_0 = 3*z*pow(x + y + z - 1, 2);} break;
	case 11: {result_0 = -3*pow(z, 2)*(x + y + z - 1);} break;
	case 12: {result_0 = 3*pow(x, 2)*z;} break;
	case 13: {result_0 = 3*x*pow(z, 2);} break;
	case 14: {result_0 = 3*pow(y, 2)*z;} break;
	case 15: {result_0 = 3*y*pow(z, 2);} break;
	case 16: {result_0 = -6*x*y*(x + y + z - 1);} break;
	case 17: {result_0 = -6*x*z*(x + y + z - 1);} break;
	case 18: {result_0 = 6*x*y*z;} break;
	case 19: {result_0 = -6*y*z*(x + y + z - 1);} break;
	default: assert(false);
}}
void b_3_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = -3*pow(x + y + z - 1, 2);val.col(0) = result_0; }{result_0 = -3*pow(x + y + z - 1, 2);val.col(1) = result_0; }{result_0 = -3*pow(x + y + z - 1, 2);val.col(2) = result_0; }} break;
	case 1: {{result_0 = 3*pow(x, 2);val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 2: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 3*pow(y, 2);val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 3: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = 3*pow(z, 2);val.col(2) = result_0; }} break;
	case 4: {{const auto helper_0 = y + z - 1;
result_0 = 3*(helper_0 + x)*(helper_0 + 3*x);val.col(0) = result_0; }{result_0 = 6*x*(x + y + z - 1);val.col(1) = result_0; }{result_0 = 6*x*(x + y + z - 1);val.col(2) = result_0; }} break;
	case 5: {{const auto helper_0 = 3*x;
result_0 = -helper_0*(helper_0 + 2*y + 2*z - 2);val.col(0) = result_0; }{result_0 = -3*pow(x, 2);val.col(1) = result_0; }{result_0 = -3*pow(x, 2);val.col(2) = result_0; }} break;
	case 6: {{result_0 = 6*x*y;val.col(0) = result_0; }{result_0 = 3*pow(x, 2);val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 7: {{result_0 = 3*pow(y, 2);val.col(0) = result_0; }{result_0 = 6*x*y;val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 8: {{result_0 = -3*pow(y, 2);val.col(0) = result_0; }{const auto helper_0 = 3*y;
result_0 = -helper_0*(helper_0 + 2*x + 2*z - 2);val.col(1) = result_0; }{result_0 = -3*pow(y, 2);val.col(2) = result_0; }} break;
	case 9: {{result_0 = 6*y*(x + y + z - 1);val.col(0) = result_0; }{const auto helper_0 = x + z - 1;
result_0 = 3*(helper_0 + y)*(helper_0 + 3*y);val.col(1) = result_0; }{result_0 = 6*y*(x + y + z - 1);val.col(2) = result_0; }} break;
	case 10: {{result_0 = 6*z*(x + y + z - 1);val.col(0) = result_0; }{result_0 = 6*z*(x + y + z - 1);val.col(1) = result_0; }{const auto helper_0 = x + y - 1;
result_0 = 3*(helper_0 + z)*(helper_0 + 3*z);val.col(2) = result_0; }} break;
	case 11: {{result_0 = -3*pow(z, 2);val.col(0) = result_0; }{result_0 = -3*pow(z, 2);val.col(1) = result_0; }{const auto helper_0 = 3*z;
result_0 = -helper_0*(helper_0 + 2*x + 2*y - 2);val.col(2) = result_0; }} break;
	case 12: {{result_0 = 6*x*z;val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = 3*pow(x, 2);val.col(2) = result_0; }} break;
	case 13: {{result_0 = 3*pow(z, 2);val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = 6*x*z;val.col(2) = result_0; }} break;
	case 14: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 6*y*z;val.col(1) = result_0; }{result_0 = 3*pow(y, 2);val.col(2) = result_0; }} break;
	case 15: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 3*pow(z, 2);val.col(1) = result_0; }{result_0 = 6*y*z;val.col(2) = result_0; }} break;
	case 16: {{result_0 = -6*y*(2*x + y + z - 1);val.col(0) = result_0; }{result_0 = -6*x*(x + 2*y + z - 1);val.col(1) = result_0; }{result_0 = -6*x*y;val.col(2) = result_0; }} break;
	case 17: {{result_0 = -6*z*(2*x + y + z - 1);val.col(0) = result_0; }{result_0 = -6*x*z;val.col(1) = result_0; }{result_0 = -6*x*(x + y + 2*z - 1);val.col(2) = result_0; }} break;
	case 18: {{result_0 = 6*y*z;val.col(0) = result_0; }{result_0 = 6*x*z;val.col(1) = result_0; }{result_0 = 6*x*y;val.col(2) = result_0; }} break;
	case 19: {{result_0 = -6*y*z;val.col(0) = result_0; }{result_0 = -6*z*(x + 2*y + z - 1);val.col(1) = result_0; }{result_0 = -6*y*(x + y + 2*z - 1);val.col(2) = result_0; }} break;
	default: assert(false);
}}


void b_4_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

switch(local_index){
	case 0: {result_0 = pow(x + y + z - 1, 4);} break;
	case 1: {result_0 = pow(x, 4);} break;
	case 2: {result_0 = pow(y, 4);} break;
	case 3: {result_0 = pow(z, 4);} break;
	case 4: {result_0 = -4*x*pow(x + y + z - 1, 3);} break;
	case 5: {result_0 = 6*pow(x, 2)*pow(x + y + z - 1, 2);} break;
	case 6: {result_0 = -4*pow(x, 3)*(x + y + z - 1);} break;
	case 7: {result_0 = 4*pow(x, 3)*y;} break;
	case 8: {result_0 = 6*pow(x, 2)*pow(y, 2);} break;
	case 9: {result_0 = 4*x*pow(y, 3);} break;
	case 10: {result_0 = -4*pow(y, 3)*(x + y + z - 1);} break;
	case 11: {result_0 = 6*pow(y, 2)*pow(x + y + z - 1, 2);} break;
	case 12: {result_0 = -4*y*pow(x + y + z - 1, 3);} break;
	case 13: {result_0 = -4*z*pow(x + y + z - 1, 3);} break;
	case 14: {result_0 = 6*pow(z, 2)*pow(x + y + z - 1, 2);} break;
	case 15: {result_0 = -4*pow(z, 3)*(x + y + z - 1);} break;
	case 16: {result_0 = 4*pow(x, 3)*z;} break;
	case 17: {result_0 = 6*pow(x, 2)*pow(z, 2);} break;
	case 18: {result_0 = 4*x*pow(z, 3);} break;
	case 19: {result_0 = 4*pow(y, 3)*z;} break;
	case 20: {result_0 = 6*pow(y, 2)*pow(z, 2);} break;
	case 21: {result_0 = 4*y*pow(z, 3);} break;
	case 22: {result_0 = 12*x*y*pow(x + y + z - 1, 2);} break;
	case 23: {result_0 = -12*x*pow(y, 2)*(x + y + z - 1);} break;
	case 24: {result_0 = -12*pow(x, 2)*y*(x + y + z - 1);} break;
	case 25: {result_0 = 12*x*z*pow(x + y + z - 1, 2);} break;
	case 26: {result_0 = -12*x*pow(z, 2)*(x + y + z - 1);} break;
	case 27: {result_0 = -12*pow(x, 2)*z*(x + y + z - 1);} break;
	case 28: {result_0 = 12*pow(x, 2)*y*z;} break;
	case 29: {result_0 = 12*x*y*pow(z, 2);} break;
	case 30: {result_0 = 12*x*pow(y, 2)*z;} break;
	case 31: {result_0 = -12*pow(y, 2)*z*(x + y + z - 1);} break;
	case 32: {result_0 = -12*y*pow(z, 2)*(x + y + z - 1);} break;
	case 33: {result_0 = 12*y*z*pow(x + y + z - 1, 2);} break;
	case 34: {result_0 = -24*x*y*z*(x + y + z - 1);} break;
	default: assert(false);
}}
void b_4_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

auto x=uv.col(0).array();
auto y=uv.col(1).array();
auto z=uv.col(2).array();

val.resize(uv.rows(), uv.cols());
 Eigen::ArrayXd result_0(uv.rows());
switch(local_index){
	case 0: {{result_0 = 4*pow(x + y + z - 1, 3);val.col(0) = result_0; }{result_0 = 4*pow(x + y + z - 1, 3);val.col(1) = result_0; }{result_0 = 4*pow(x + y + z - 1, 3);val.col(2) = result_0; }} break;
	case 1: {{result_0 = 4*pow(x, 3);val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 2: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 4*pow(y, 3);val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 3: {{result_0.setZero();val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = 4*pow(z, 3);val.col(2) = result_0; }} break;
	case 4: {{const auto helper_0 = y + z - 1;
result_0 = -4*pow(helper_0 + x, 2)*(helper_0 + 4*x);val.col(0) = result_0; }{result_0 = -12*x*pow(x + y + z - 1, 2);val.col(1) = result_0; }{result_0 = -12*x*pow(x + y + z - 1, 2);val.col(2) = result_0; }} break;
	case 5: {{const auto helper_0 = y + z - 1;
result_0 = 12*x*(helper_0 + x)*(helper_0 + 2*x);val.col(0) = result_0; }{result_0 = 12*pow(x, 2)*(x + y + z - 1);val.col(1) = result_0; }{result_0 = 12*pow(x, 2)*(x + y + z - 1);val.col(2) = result_0; }} break;
	case 6: {{result_0 = -4*pow(x, 2)*(4*x + 3*y + 3*z - 3);val.col(0) = result_0; }{result_0 = -4*pow(x, 3);val.col(1) = result_0; }{result_0 = -4*pow(x, 3);val.col(2) = result_0; }} break;
	case 7: {{result_0 = 12*pow(x, 2)*y;val.col(0) = result_0; }{result_0 = 4*pow(x, 3);val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 8: {{result_0 = 12*x*pow(y, 2);val.col(0) = result_0; }{result_0 = 12*pow(x, 2)*y;val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 9: {{result_0 = 4*pow(y, 3);val.col(0) = result_0; }{result_0 = 12*x*pow(y, 2);val.col(1) = result_0; }{result_0.setZero();val.col(2) = result_0; }} break;
	case 10: {{result_0 = -4*pow(y, 3);val.col(0) = result_0; }{result_0 = -4*pow(y, 2)*(3*x + 4*y + 3*z - 3);val.col(1) = result_0; }{result_0 = -4*pow(y, 3);val.col(2) = result_0; }} break;
	case 11: {{result_0 = 12*pow(y, 2)*(x + y + z - 1);val.col(0) = result_0; }{const auto helper_0 = x + z - 1;
result_0 = 12*y*(helper_0 + y)*(helper_0 + 2*y);val.col(1) = result_0; }{result_0 = 12*pow(y, 2)*(x + y + z - 1);val.col(2) = result_0; }} break;
	case 12: {{result_0 = -12*y*pow(x + y + z - 1, 2);val.col(0) = result_0; }{const auto helper_0 = x + z - 1;
result_0 = -4*pow(helper_0 + y, 2)*(helper_0 + 4*y);val.col(1) = result_0; }{result_0 = -12*y*pow(x + y + z - 1, 2);val.col(2) = result_0; }} break;
	case 13: {{result_0 = -12*z*pow(x + y + z - 1, 2);val.col(0) = result_0; }{result_0 = -12*z*pow(x + y + z - 1, 2);val.col(1) = result_0; }{const auto helper_0 = x + y - 1;
result_0 = -4*pow(helper_0 + z, 2)*(helper_0 + 4*z);val.col(2) = result_0; }} break;
	case 14: {{result_0 = 12*pow(z, 2)*(x + y + z - 1);val.col(0) = result_0; }{result_0 = 12*pow(z, 2)*(x + y + z - 1);val.col(1) = result_0; }{const auto helper_0 = x + y - 1;
result_0 = 12*z*(helper_0 + z)*(helper_0 + 2*z);val.col(2) = result_0; }} break;
	case 15: {{result_0 = -4*pow(z, 3);val.col(0) = result_0; }{result_0 = -4*pow(z, 3);val.col(1) = result_0; }{result_0 = -4*pow(z, 2)*(3*x + 3*y + 4*z - 3);val.col(2) = result_0; }} break;
	case 16: {{result_0 = 12*pow(x, 2)*z;val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = 4*pow(x, 3);val.col(2) = result_0; }} break;
	case 17: {{result_0 = 12*x*pow(z, 2);val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = 12*pow(x, 2)*z;val.col(2) = result_0; }} break;
	case 18: {{result_0 = 4*pow(z, 3);val.col(0) = result_0; }{result_0.setZero();val.col(1) = result_0; }{result_0 = 12*x*pow(z, 2);val.col(2) = result_0; }} break;
	case 19: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 12*pow(y, 2)*z;val.col(1) = result_0; }{result_0 = 4*pow(y, 3);val.col(2) = result_0; }} break;
	case 20: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 12*y*pow(z, 2);val.col(1) = result_0; }{result_0 = 12*pow(y, 2)*z;val.col(2) = result_0; }} break;
	case 21: {{result_0.setZero();val.col(0) = result_0; }{result_0 = 4*pow(z, 3);val.col(1) = result_0; }{result_0 = 12*y*pow(z, 2);val.col(2) = result_0; }} break;
	case 22: {{const auto helper_0 = y + z - 1;
result_0 = 12*y*(helper_0 + x)*(helper_0 + 3*x);val.col(0) = result_0; }{const auto helper_0 = x + z - 1;
result_0 = 12*x*(helper_0 + y)*(helper_0 + 3*y);val.col(1) = result_0; }{result_0 = 24*x*y*(x + y + z - 1);val.col(2) = result_0; }} break;
	case 23: {{result_0 = -12*pow(y, 2)*(2*x + y + z - 1);val.col(0) = result_0; }{result_0 = -12*x*y*(2*x + 3*y + 2*z - 2);val.col(1) = result_0; }{result_0 = -12*x*pow(y, 2);val.col(2) = result_0; }} break;
	case 24: {{result_0 = -12*x*y*(3*x + 2*y + 2*z - 2);val.col(0) = result_0; }{result_0 = -12*pow(x, 2)*(x + 2*y + z - 1);val.col(1) = result_0; }{result_0 = -12*pow(x, 2)*y;val.col(2) = result_0; }} break;
	case 25: {{const auto helper_0 = y + z - 1;
result_0 = 12*z*(helper_0 + x)*(helper_0 + 3*x);val.col(0) = result_0; }{result_0 = 24*x*z*(x + y + z - 1);val.col(1) = result_0; }{const auto helper_0 = x + y - 1;
result_0 = 12*x*(helper_0 + z)*(helper_0 + 3*z);val.col(2) = result_0; }} break;
	case 26: {{result_0 = -12*pow(z, 2)*(2*x + y + z - 1);val.col(0) = result_0; }{result_0 = -12*x*pow(z, 2);val.col(1) = result_0; }{result_0 = -12*x*z*(2*x + 2*y + 3*z - 2);val.col(2) = result_0; }} break;
	case 27: {{result_0 = -12*x*z*(3*x + 2*y + 2*z - 2);val.col(0) = result_0; }{result_0 = -12*pow(x, 2)*z;val.col(1) = result_0; }{result_0 = -12*pow(x, 2)*(x + y + 2*z - 1);val.col(2) = result_0; }} break;
	case 28: {{result_0 = 24*x*y*z;val.col(0) = result_0; }{result_0 = 12*pow(x, 2)*z;val.col(1) = result_0; }{result_0 = 12*pow(x, 2)*y;val.col(2) = result_0; }} break;
	case 29: {{result_0 = 12*y*pow(z, 2);val.col(0) = result_0; }{result_0 = 12*x*pow(z, 2);val.col(1) = result_0; }{result_0 = 24*x*y*z;val.col(2) = result_0; }} break;
	case 30: {{result_0 = 12*pow(y, 2)*z;val.col(0) = result_0; }{result_0 = 24*x*y*z;val.col(1) = result_0; }{result_0 = 12*x*pow(y, 2);val.col(2) = result_0; }} break;
	case 31: {{result_0 = -12*pow(y, 2)*z;val.col(0) = result_0; }{result_0 = -12*y*z*(2*x + 3*y + 2*z - 2);val.col(1) = result_0; }{result_0 = -12*pow(y, 2)*(x + y + 2*z - 1);val.col(2) = result_0; }} break;
	case 32: {{result_0 = -12*y*pow(z, 2);val.col(0) = result_0; }{result_0 = -12*pow(z, 2)*(x + 2*y + z - 1);val.col(1) = result_0; }{result_0 = -12*y*z*(2*x + 2*y + 3*z - 2);val.col(2) = result_0; }} break;
	case 33: {{result_0 = 24*y*z*(x + y + z - 1);val.col(0) = result_0; }{const auto helper_0 = x + z - 1;
result_0 = 12*z*(helper_0 + y)*(helper_0 + 3*y);val.col(1) = result_0; }{const auto helper_0 = x + y - 1;
result_0 = 12*y*(helper_0 + z)*(helper_0 + 3*z);val.col(2) = result_0; }} break;
	case 34: {{result_0 = -24*y*z*(2*x + y + z - 1);val.col(0) = result_0; }{result_0 = -24*x*z*(x + 2*y + z - 1);val.col(1) = result_0; }{result_0 = -24*x*y*(x + y + 2*z - 1);val.col(2) = result_0; }} break;
	default: assert(false);
}}


}


void b_basis_value_3d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

switch(b){
	case 0: b_0_basis_value_3d(local_index, uv, val); break;
	case 1: b_1_basis_value_3d(local_index, uv, val); break;
	case 2: b_2_basis_value_3d(local_index, uv, val); break;
	case 3: b_3_basis_value_3d(local_index, uv, val); break;
	case 4: b_4_basis_value_3d(local_index, uv, val); break;
	default: assert(false); 
}}

void b_grad_basis_value_3d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

switch(b){
	case 0: b_0_basis_grad_value_3d(local_index, uv, val); break;
	case 1: b_1_basis_grad_value_3d(local_index, uv, val); break;
	case 2: b_2_basis_grad_value_3d(local_index, uv, val); break;
	case 3: b_3_basis_grad_value_3d(local_index, uv, val); break;
	case 4: b_4_basis_grad_value_3d(local_index, uv, val); break;
	default: assert(false); 
}}

namespace {

}}}
