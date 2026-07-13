#include "auto_p_bases_nodes.hpp"
#include "p_n_bases.hpp"

namespace polyfem {
namespace autogen {
namespace {
void p_0_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(1, 2); res << 
0.33333333333333331, 0.33333333333333331;
}


void p_1_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(3, 2); res << 
0, 0,
1, 0,
0, 1;
}


void p_2_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(6, 2); res << 
0, 0,
1, 0,
0, 1,
1.0/2.0, 0,
1.0/2.0, 1.0/2.0,
0, 1.0/2.0;
}


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

namespace {
void p_0_nodes_3d(Eigen::MatrixXd &res) {
 res.resize(1, 3); res << 
0.33333333333333331, 0.33333333333333331, 0.33333333333333331;
}


void p_1_nodes_3d(Eigen::MatrixXd &res) {
 res.resize(4, 3); res << 
0, 0, 0,
1, 0, 0,
0, 1, 0,
0, 0, 1;
}


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

namespace {

}}}
