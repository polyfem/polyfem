#include "auto_q_bases_1d_nodes.hpp"


namespace polyfem {
namespace autogen {
namespace {
void q_0_nodes_1d(Eigen::MatrixXd &res) {
 res.resize(1, 1); res << 
0.5;
}

void q_1_nodes_1d(Eigen::MatrixXd &res) {
 res.resize(2, 1); res << 
0,
1;
}

void q_2_nodes_1d(Eigen::MatrixXd &res) {
 res.resize(3, 1); res << 
0,
1.0/2.0,
1;
}

void q_3_nodes_1d(Eigen::MatrixXd &res) {
 res.resize(4, 1); res << 
0,
1.0/3.0,
2.0/3.0,
1;
}

void q_m2_nodes_1d(Eigen::MatrixXd &res) {
 res.resize(3, 1); res << 
0,
1.0/2.0,
1;
}

void q_4_nodes_1d(Eigen::MatrixXd &res) {
 res.resize(5, 1); res << 
0,
1.0/4.0,
1.0/2.0,
3.0/4.0,
1;
}

void q_5_nodes_1d(Eigen::MatrixXd &res) {
 res.resize(6, 1); res << 
0,
1.0/5.0,
2.0/5.0,
3.0/5.0,
4.0/5.0,
1;
}

void q_6_nodes_1d(Eigen::MatrixXd &res) {
 res.resize(7, 1); res << 
0,
1.0/6.0,
1.0/3.0,
1.0/2.0,
2.0/3.0,
5.0/6.0,
1;
}

}

void q_nodes_1d(const int q, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_nodes_1d(val); break;
	case 1: q_1_nodes_1d(val); break;
	case 2: q_2_nodes_1d(val); break;
	case 3: q_3_nodes_1d(val); break;
	case -2: q_m2_nodes_1d(val); break;
	case 4: q_4_nodes_1d(val); break;
	case 5: q_5_nodes_1d(val); break;
	case 6: q_6_nodes_1d(val); break;
	default: assert(false);
}}
}}
