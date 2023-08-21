#include "auto_q_bases_2d_nodes.hpp"


namespace polyfem {
namespace autogen {
namespace {
void q_0_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(1, 2); res << 
0.5, 0.5;
}

void q_1_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(4, 2); res << 
0, 0,
1, 0,
1, 1,
0, 1;
}

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

void q_3_nodes_2d(Eigen::MatrixXd &res) {
 res.resize(16, 2); res << 
0, 0,
1, 0,
1, 1,
0, 1,
1.0/3.0, 0,
2.0/3.0, 0,
1, 1.0/3.0,
1, 2.0/3.0,
2.0/3.0, 1,
1.0/3.0, 1,
0, 2.0/3.0,
0, 1.0/3.0,
1.0/3.0, 1.0/3.0,
1.0/3.0, 2.0/3.0,
2.0/3.0, 1.0/3.0,
2.0/3.0, 2.0/3.0;
}

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
	case 3: q_3_nodes_2d(val); break;
	case -2: q_m2_nodes_2d(val); break;
	default: assert(false);
}}
}}
