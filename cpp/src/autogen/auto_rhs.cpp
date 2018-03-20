#include "auto_rhs.hpp"


namespace poly_fem {
namespace autogen {
void saint_venenant_2d_function(const AutodiffHessianPt &pt, const ElasticityTensor &C, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res) {
res.resize(2);
const auto // Not supported in C:
// f0
helper_0 = pt(0);
const auto // Not supported in C:
// 
helper_1 = helper_0.getHessian()(0,0);
const auto // Not supported in C:
// C
helper_2 = C(0, 0);
const auto // Not supported in C:
// 
helper_3 = helper_0.getGradient()(0);
const auto // Not supported in C:
// f1
helper_4 = pt(1);
const auto // Not supported in C:
// 
helper_5 = helper_4.getGradient()(0);
const auto helper_6 = 0.5*pow(helper_3, 2) + 1.0*helper_3 + 0.5*pow(helper_5, 2);
const auto // Not supported in C:
// C
helper_7 = C(0, 1);
const auto // Not supported in C:
// 
helper_8 = helper_4.getGradient()(1);
const auto // Not supported in C:
// 
helper_9 = helper_0.getGradient()(1);
const auto helper_10 = 0.5*pow(helper_8, 2) + 1.0*helper_8 + 0.5*pow(helper_9, 2);
const auto // Not supported in C:
// C
helper_11 = C(0, 2);
const auto helper_12 = helper_3*helper_9 + helper_5*helper_8 + helper_5 + helper_9;
const auto helper_13 = helper_10*helper_7 + helper_11*helper_12 + helper_2*helper_6;
const auto // Not supported in C:
// 
helper_14 = helper_0.getHessian()(1,1);
const auto // Not supported in C:
// C
helper_15 = C(1, 0);
const auto // Not supported in C:
// C
helper_16 = C(1, 1);
const auto // Not supported in C:
// C
helper_17 = C(1, 2);
const auto helper_18 = helper_10*helper_16 + helper_12*helper_17 + helper_15*helper_6;
const auto // Not supported in C:
// 
helper_19 = helper_0.getHessian()(0,1);
const auto // Not supported in C:
// C
helper_20 = C(2, 0);
const auto // Not supported in C:
// C
helper_21 = C(2, 1);
const auto // Not supported in C:
// C
helper_22 = C(2, 2);
const auto helper_23 = 2*helper_10*helper_21 + 2*helper_12*helper_22 + 2*helper_20*helper_6;
const auto // Not supported in C:
// 
helper_24 = helper_4.getHessian()(0,0);
const auto helper_25 = 1.0*helper_1*helper_3 + 1.0*helper_1 + 1.0*helper_24*helper_5;
const auto // Not supported in C:
// 
helper_26 = helper_4.getHessian()(0,1);
const auto helper_27 = 1.0*helper_19*helper_9 + 1.0*helper_26*helper_8 + 1.0*helper_26;
const auto helper_28 = helper_1*helper_9 + helper_19*helper_3 + helper_19 + helper_24*helper_8 + helper_24 + helper_26*helper_5;
const auto helper_29 = helper_20*helper_25 + helper_21*helper_27 + helper_22*helper_28;
const auto helper_30 = 1.0*helper_19*helper_3 + 1.0*helper_19 + 1.0*helper_26*helper_5;
const auto // Not supported in C:
// 
helper_31 = helper_4.getHessian()(1,1);
const auto helper_32 = 1.0*helper_14*helper_9 + 1.0*helper_31*helper_8 + 1.0*helper_31;
const auto helper_33 = helper_14*helper_3 + helper_14 + helper_19*helper_9 + helper_26*helper_8 + helper_26 + helper_31*helper_5;
const auto helper_34 = helper_15*helper_30 + helper_16*helper_32 + helper_17*helper_33;
const auto helper_35 = helper_3 + 1;
const auto helper_36 = helper_11*helper_28 + helper_2*helper_25 + helper_27*helper_7;
const auto helper_37 = helper_20*helper_30 + helper_21*helper_32 + helper_22*helper_33;
const auto helper_38 = helper_8 + 1;
res(0) = helper_1*helper_13 + helper_14*helper_18 + helper_19*helper_23 + helper_29*helper_9 + helper_34*helper_9 + helper_35*helper_36 + helper_35*helper_37;
res(1) = helper_13*helper_24 + helper_18*helper_31 + helper_23*helper_26 + helper_29*helper_38 + helper_34*helper_38 + helper_36*helper_5 + helper_37*helper_5;
}

void saint_venenant_3d_function(const AutodiffHessianPt &pt, const ElasticityTensor &C, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res) {
res.resize(3);
const auto // Not supported in C:
// f0
helper_0 = pt(0);
const auto // Not supported in C:
// 
helper_1 = helper_0.getHessian()(0,0);
const auto // Not supported in C:
// C
helper_2 = C(0, 0);
const auto // Not supported in C:
// 
helper_3 = helper_0.getGradient()(0);
const auto // Not supported in C:
// f1
helper_4 = pt(1);
const auto // Not supported in C:
// 
helper_5 = helper_4.getGradient()(0);
const auto // Not supported in C:
// f2
helper_6 = pt(2);
const auto // Not supported in C:
// 
helper_7 = helper_6.getGradient()(0);
const auto helper_8 = 0.5*pow(helper_3, 2) + 1.0*helper_3 + 0.5*pow(helper_5, 2) + 0.5*pow(helper_7, 2);
const auto // Not supported in C:
// C
helper_9 = C(0, 1);
const auto // Not supported in C:
// 
helper_10 = helper_4.getGradient()(1);
const auto // Not supported in C:
// 
helper_11 = helper_0.getGradient()(1);
const auto // Not supported in C:
// 
helper_12 = helper_6.getGradient()(1);
const auto helper_13 = 0.5*pow(helper_10, 2) + 1.0*helper_10 + 0.5*pow(helper_11, 2) + 0.5*pow(helper_12, 2);
const auto // Not supported in C:
// C
helper_14 = C(0, 2);
const auto // Not supported in C:
// 
helper_15 = helper_6.getGradient()(2);
const auto // Not supported in C:
// 
helper_16 = helper_0.getGradient()(2);
const auto // Not supported in C:
// 
helper_17 = helper_4.getGradient()(2);
const auto helper_18 = 0.5*pow(helper_15, 2) + 1.0*helper_15 + 0.5*pow(helper_16, 2) + 0.5*pow(helper_17, 2);
const auto // Not supported in C:
// C
helper_19 = C(0, 5);
const auto helper_20 = helper_10*helper_5 + helper_11*helper_3 + helper_11 + helper_12*helper_7 + helper_5;
const auto // Not supported in C:
// C
helper_21 = C(0, 4);
const auto helper_22 = helper_15*helper_7 + helper_16*helper_3 + helper_16 + helper_17*helper_5 + helper_7;
const auto // Not supported in C:
// C
helper_23 = C(0, 3);
const auto helper_24 = helper_10*helper_17 + helper_11*helper_16 + helper_12*helper_15 + helper_12 + helper_17;
const auto helper_25 = helper_13*helper_9 + helper_14*helper_18 + helper_19*helper_20 + helper_2*helper_8 + helper_21*helper_22 + helper_23*helper_24;
const auto // Not supported in C:
// 
helper_26 = helper_0.getHessian()(1,1);
const auto // Not supported in C:
// C
helper_27 = C(1, 0);
const auto // Not supported in C:
// C
helper_28 = C(1, 1);
const auto // Not supported in C:
// C
helper_29 = C(1, 2);
const auto // Not supported in C:
// C
helper_30 = C(1, 5);
const auto // Not supported in C:
// C
helper_31 = C(1, 4);
const auto // Not supported in C:
// C
helper_32 = C(1, 3);
const auto helper_33 = helper_13*helper_28 + helper_18*helper_29 + helper_20*helper_30 + helper_22*helper_31 + helper_24*helper_32 + helper_27*helper_8;
const auto // Not supported in C:
// 
helper_34 = helper_0.getHessian()(2,2);
const auto // Not supported in C:
// C
helper_35 = C(2, 0);
const auto // Not supported in C:
// C
helper_36 = C(2, 1);
const auto // Not supported in C:
// C
helper_37 = C(2, 2);
const auto // Not supported in C:
// C
helper_38 = C(2, 5);
const auto // Not supported in C:
// C
helper_39 = C(2, 4);
const auto // Not supported in C:
// C
helper_40 = C(2, 3);
const auto helper_41 = helper_13*helper_36 + helper_18*helper_37 + helper_20*helper_38 + helper_22*helper_39 + helper_24*helper_40 + helper_35*helper_8;
const auto // Not supported in C:
// 
helper_42 = helper_0.getHessian()(1,2);
const auto // Not supported in C:
// C
helper_43 = C(3, 0);
const auto // Not supported in C:
// C
helper_44 = C(3, 1);
const auto // Not supported in C:
// C
helper_45 = C(3, 2);
const auto // Not supported in C:
// C
helper_46 = C(3, 5);
const auto // Not supported in C:
// C
helper_47 = C(3, 4);
const auto // Not supported in C:
// C
helper_48 = C(3, 3);
const auto helper_49 = 2*helper_13*helper_44 + 2*helper_18*helper_45 + 2*helper_20*helper_46 + 2*helper_22*helper_47 + 2*helper_24*helper_48 + 2*helper_43*helper_8;
const auto // Not supported in C:
// 
helper_50 = helper_0.getHessian()(0,2);
const auto // Not supported in C:
// C
helper_51 = C(4, 0);
const auto // Not supported in C:
// C
helper_52 = C(4, 1);
const auto // Not supported in C:
// C
helper_53 = C(4, 2);
const auto // Not supported in C:
// C
helper_54 = C(4, 5);
const auto // Not supported in C:
// C
helper_55 = C(4, 4);
const auto // Not supported in C:
// C
helper_56 = C(4, 3);
const auto helper_57 = 2*helper_13*helper_52 + 2*helper_18*helper_53 + 2*helper_20*helper_54 + 2*helper_22*helper_55 + 2*helper_24*helper_56 + 2*helper_51*helper_8;
const auto // Not supported in C:
// 
helper_58 = helper_0.getHessian()(0,1);
const auto // Not supported in C:
// C
helper_59 = C(5, 0);
const auto // Not supported in C:
// C
helper_60 = C(5, 1);
const auto // Not supported in C:
// C
helper_61 = C(5, 2);
const auto // Not supported in C:
// C
helper_62 = C(5, 5);
const auto // Not supported in C:
// C
helper_63 = C(5, 4);
const auto // Not supported in C:
// C
helper_64 = C(5, 3);
const auto helper_65 = 2*helper_13*helper_60 + 2*helper_18*helper_61 + 2*helper_20*helper_62 + 2*helper_22*helper_63 + 2*helper_24*helper_64 + 2*helper_59*helper_8;
const auto // Not supported in C:
// 
helper_66 = helper_4.getHessian()(0,0);
const auto // Not supported in C:
// 
helper_67 = helper_6.getHessian()(0,0);
const auto helper_68 = 1.0*helper_1*helper_3 + 1.0*helper_1 + 1.0*helper_5*helper_66 + 1.0*helper_67*helper_7;
const auto // Not supported in C:
// 
helper_69 = helper_4.getHessian()(0,1);
const auto // Not supported in C:
// 
helper_70 = helper_6.getHessian()(0,1);
const auto helper_71 = 1.0*helper_10*helper_69 + 1.0*helper_11*helper_58 + 1.0*helper_12*helper_70 + 1.0*helper_69;
const auto // Not supported in C:
// 
helper_72 = helper_6.getHessian()(0,2);
const auto // Not supported in C:
// 
helper_73 = helper_4.getHessian()(0,2);
const auto helper_74 = 1.0*helper_15*helper_72 + 1.0*helper_16*helper_50 + 1.0*helper_17*helper_73 + 1.0*helper_72;
const auto helper_75 = helper_1*helper_11 + helper_10*helper_66 + helper_12*helper_67 + helper_3*helper_58 + helper_5*helper_69 + helper_58 + helper_66 + helper_7*helper_70;
const auto helper_76 = helper_1*helper_16 + helper_15*helper_67 + helper_17*helper_66 + helper_3*helper_50 + helper_5*helper_73 + helper_50 + helper_67 + helper_7*helper_72;
const auto helper_77 = helper_10*helper_73 + helper_11*helper_50 + helper_12*helper_72 + helper_73;
const auto helper_78 = helper_15*helper_70 + helper_16*helper_58 + helper_17*helper_69 + helper_70;
const auto helper_79 = helper_77 + helper_78;
const auto helper_80 = helper_51*helper_68 + helper_52*helper_71 + helper_53*helper_74 + helper_54*helper_75 + helper_55*helper_76 + helper_56*helper_79;
const auto helper_81 = helper_59*helper_68 + helper_60*helper_71 + helper_61*helper_74 + helper_62*helper_75 + helper_63*helper_76 + helper_64*helper_79;
const auto helper_82 = 1.0*helper_3*helper_58 + 1.0*helper_5*helper_69 + 1.0*helper_58 + 1.0*helper_7*helper_70;
const auto // Not supported in C:
// 
helper_83 = helper_4.getHessian()(1,1);
const auto // Not supported in C:
// 
helper_84 = helper_6.getHessian()(1,1);
const auto helper_85 = 1.0*helper_10*helper_83 + 1.0*helper_11*helper_26 + 1.0*helper_12*helper_84 + 1.0*helper_83;
const auto // Not supported in C:
// 
helper_86 = helper_6.getHessian()(1,2);
const auto // Not supported in C:
// 
helper_87 = helper_4.getHessian()(1,2);
const auto helper_88 = 1.0*helper_15*helper_86 + 1.0*helper_16*helper_42 + 1.0*helper_17*helper_87 + 1.0*helper_86;
const auto helper_89 = helper_10*helper_69 + helper_11*helper_58 + helper_12*helper_70 + helper_26*helper_3 + helper_26 + helper_5*helper_83 + helper_69 + helper_7*helper_84;
const auto helper_90 = helper_3*helper_42 + helper_42 + helper_5*helper_87 + helper_7*helper_86;
const auto helper_91 = helper_78 + helper_90;
const auto helper_92 = helper_10*helper_87 + helper_11*helper_42 + helper_12*helper_86 + helper_15*helper_84 + helper_16*helper_26 + helper_17*helper_83 + helper_84 + helper_87;
const auto helper_93 = helper_27*helper_82 + helper_28*helper_85 + helper_29*helper_88 + helper_30*helper_89 + helper_31*helper_91 + helper_32*helper_92;
const auto helper_94 = helper_43*helper_82 + helper_44*helper_85 + helper_45*helper_88 + helper_46*helper_89 + helper_47*helper_91 + helper_48*helper_92;
const auto helper_95 = 1.0*helper_3*helper_50 + 1.0*helper_5*helper_73 + 1.0*helper_50 + 1.0*helper_7*helper_72;
const auto helper_96 = 1.0*helper_10*helper_87 + 1.0*helper_11*helper_42 + 1.0*helper_12*helper_86 + 1.0*helper_87;
const auto // Not supported in C:
// 
helper_97 = helper_6.getHessian()(2,2);
const auto // Not supported in C:
// 
helper_98 = helper_4.getHessian()(2,2);
const auto helper_99 = 1.0*helper_15*helper_97 + 1.0*helper_16*helper_34 + 1.0*helper_17*helper_98 + 1.0*helper_97;
const auto helper_100 = helper_77 + helper_90;
const auto helper_101 = helper_15*helper_72 + helper_16*helper_50 + helper_17*helper_73 + helper_3*helper_34 + helper_34 + helper_5*helper_98 + helper_7*helper_97 + helper_72;
const auto helper_102 = helper_10*helper_98 + helper_11*helper_34 + helper_12*helper_97 + helper_15*helper_86 + helper_16*helper_42 + helper_17*helper_87 + helper_86 + helper_98;
const auto helper_103 = helper_100*helper_38 + helper_101*helper_39 + helper_102*helper_40 + helper_35*helper_95 + helper_36*helper_96 + helper_37*helper_99;
const auto helper_104 = helper_100*helper_46 + helper_101*helper_47 + helper_102*helper_48 + helper_43*helper_95 + helper_44*helper_96 + helper_45*helper_99;
const auto helper_105 = helper_3 + 1;
const auto helper_106 = helper_14*helper_74 + helper_19*helper_75 + helper_2*helper_68 + helper_21*helper_76 + helper_23*helper_79 + helper_71*helper_9;
const auto helper_107 = helper_59*helper_82 + helper_60*helper_85 + helper_61*helper_88 + helper_62*helper_89 + helper_63*helper_91 + helper_64*helper_92;
const auto helper_108 = helper_100*helper_54 + helper_101*helper_55 + helper_102*helper_56 + helper_51*helper_95 + helper_52*helper_96 + helper_53*helper_99;
const auto helper_109 = helper_10 + 1;
const auto helper_110 = helper_15 + 1;
res(0) = helper_1*helper_25 + helper_103*helper_16 + helper_104*helper_11 + helper_105*helper_106 + helper_105*helper_107 + helper_105*helper_108 + helper_11*helper_81 + helper_11*helper_93 + helper_16*helper_80 + helper_16*helper_94 + helper_26*helper_33 + helper_34*helper_41 + helper_42*helper_49 + helper_50*helper_57 + helper_58*helper_65;
res(1) = helper_103*helper_17 + helper_104*helper_109 + helper_106*helper_5 + helper_107*helper_5 + helper_108*helper_5 + helper_109*helper_81 + helper_109*helper_93 + helper_17*helper_80 + helper_17*helper_94 + helper_25*helper_66 + helper_33*helper_83 + helper_41*helper_98 + helper_49*helper_87 + helper_57*helper_73 + helper_65*helper_69;
res(2) = helper_103*helper_110 + helper_104*helper_12 + helper_106*helper_7 + helper_107*helper_7 + helper_108*helper_7 + helper_110*helper_80 + helper_110*helper_94 + helper_12*helper_81 + helper_12*helper_93 + helper_25*helper_67 + helper_33*helper_84 + helper_41*helper_97 + helper_49*helper_86 + helper_57*helper_72 + helper_65*helper_70;
}


void neo_hookean_2d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res) {
res.resize(2);
const auto // Not supported in C:
// f1
helper_0 = pt(1);
const auto // Not supported in C:
// 
helper_1 = helper_0.getGradient()(0);
const auto // Not supported in C:
// f0
helper_2 = pt(0);
const auto // Not supported in C:
// 
helper_3 = helper_2.getGradient()(1);
const auto // Not supported in C:
// 
helper_4 = helper_2.getGradient()(0) + 1;
const auto // Not supported in C:
// 
helper_5 = helper_0.getGradient()(1) + 1;
const auto helper_6 = -helper_1*helper_3 + helper_4*helper_5;
const auto helper_7 = pow(helper_6, -2);
const auto // Not supported in C:
// 
helper_8 = helper_0.getHessian()(0,1);
const auto helper_9 = helper_3*helper_8;
const auto // Not supported in C:
// 
helper_10 = helper_2.getHessian()(1,1);
const auto helper_11 = helper_1*helper_10;
const auto // Not supported in C:
// 
helper_12 = helper_0.getHessian()(1,1);
const auto helper_13 = helper_12*helper_4;
const auto // Not supported in C:
// 
helper_14 = helper_2.getHessian()(0,1);
const auto helper_15 = helper_14*helper_5;
const auto helper_16 = -helper_11 + helper_13 + helper_15 - helper_9;
const auto helper_17 = helper_1*helper_16*helper_7;
const auto helper_18 = helper_17*lambda;
const auto helper_19 = helper_5*helper_7;
const auto // Not supported in C:
// 
helper_20 = helper_0.getHessian()(0,0);
const auto helper_21 = helper_20*helper_3;
const auto helper_22 = helper_1*helper_14;
const auto helper_23 = helper_4*helper_8;
const auto // Not supported in C:
// 
helper_24 = helper_2.getHessian()(0,0);
const auto helper_25 = helper_24*helper_5;
const auto helper_26 = -helper_21 - helper_22 + helper_23 + helper_25;
const auto helper_27 = helper_19*helper_26*lambda;
const auto helper_28 = log(helper_6);
const auto helper_29 = 1.0/helper_6;
const auto helper_30 = helper_29*helper_8;
const auto helper_31 = helper_26*helper_3*helper_7;
const auto helper_32 = helper_31*lambda;
const auto helper_33 = helper_4*helper_7;
const auto helper_34 = helper_16*helper_33*lambda;
const auto helper_35 = helper_14*helper_29;
res(0) = helper_18*helper_28 - helper_18 - helper_27*helper_28 + helper_27 + mu*(helper_10 - helper_17 + helper_30) - mu*(helper_19*(helper_21 + helper_22 - helper_23 - helper_25) - helper_24 + helper_30);
res(1) = helper_28*helper_32 - helper_28*helper_34 - helper_32 + helper_34 - mu*(-helper_12 + helper_33*(helper_11 - helper_13 - helper_15 + helper_9) + helper_35) + mu*(helper_20 - helper_31 + helper_35);
}

void neo_hookean_3d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res) {
res.resize(3);
const auto // Not supported in C:
// f1
helper_0 = pt(1);
const auto // Not supported in C:
// 
helper_1 = helper_0.getGradient()(2);
const auto // Not supported in C:
// f0
helper_2 = pt(0);
const auto // Not supported in C:
// 
helper_3 = helper_2.getGradient()(1);
const auto // Not supported in C:
// f2
helper_4 = pt(2);
const auto // Not supported in C:
// 
helper_5 = helper_4.getGradient()(0);
const auto helper_6 = helper_3*helper_5;
const auto // Not supported in C:
// 
helper_7 = helper_4.getGradient()(1);
const auto // Not supported in C:
// 
helper_8 = helper_2.getGradient()(2);
const auto // Not supported in C:
// 
helper_9 = helper_0.getGradient()(0);
const auto helper_10 = helper_8*helper_9;
const auto // Not supported in C:
// 
helper_11 = helper_2.getGradient()(0) + 1;
const auto helper_12 = helper_1*helper_11;
const auto // Not supported in C:
// 
helper_13 = helper_0.getGradient()(1) + 1;
const auto helper_14 = helper_5*helper_8;
const auto // Not supported in C:
// 
helper_15 = helper_4.getGradient()(2) + 1;
const auto helper_16 = helper_3*helper_9;
const auto helper_17 = helper_11*helper_13;
const auto helper_18 = helper_1*helper_6 + helper_10*helper_7 - helper_12*helper_7 - helper_13*helper_14 - helper_15*helper_16 + helper_15*helper_17;
const auto helper_19 = log(helper_18);
const auto helper_20 = helper_19*lambda;
const auto helper_21 = -helper_10 + helper_12;
const auto helper_22 = helper_11*helper_7;
const auto helper_23 = helper_22 - helper_6;
const auto helper_24 = -helper_16 + helper_17;
const auto helper_25 = helper_11*helper_15 - helper_14;
const auto helper_26 = -helper_21*helper_23 + helper_24*helper_25;
const auto helper_27 = 1.0/helper_26;
const auto // Not supported in C:
// 
helper_28 = helper_0.getHessian()(0,2);
const auto // Not supported in C:
// 
helper_29 = helper_4.getHessian()(0,2);
const auto // Not supported in C:
// 
helper_30 = helper_2.getHessian()(0,2);
const auto helper_31 = helper_29*helper_3;
const auto // Not supported in C:
// 
helper_32 = helper_2.getHessian()(1,2);
const auto helper_33 = helper_32*helper_5;
const auto // Not supported in C:
// 
helper_34 = helper_4.getHessian()(1,2);
const auto helper_35 = helper_11*helper_34;
const auto helper_36 = -helper_33 + helper_35;
const auto helper_37 = helper_30*helper_7 - helper_31 + helper_36;
const auto helper_38 = helper_32*helper_9;
const auto // Not supported in C:
// 
helper_39 = helper_0.getHessian()(1,2);
const auto helper_40 = helper_11*helper_39;
const auto helper_41 = -helper_38 + helper_40;
const auto helper_42 = helper_28*helper_3;
const auto helper_43 = helper_13*helper_30;
const auto helper_44 = helper_41 - helper_42 + helper_43;
const auto helper_45 = helper_27*(-helper_23*helper_28 + helper_24*helper_29 - helper_37*helper_9 + helper_44*helper_5);
const auto helper_46 = helper_19*helper_27*lambda;
const auto // Not supported in C:
// 
helper_47 = helper_2.getHessian()(0,1);
const auto helper_48 = 1.0/helper_11;
const auto helper_49 = 1.0/helper_24;
const auto helper_50 = -helper_23*helper_9 + helper_24*helper_5;
const auto helper_51 = helper_21*helper_50;
const auto helper_52 = helper_26*helper_9;
const auto helper_53 = -helper_51 + helper_52;
const auto helper_54 = pow(helper_24, -2);
const auto // Not supported in C:
// 
helper_55 = helper_0.getHessian()(0,1);
const auto helper_56 = helper_3*helper_55;
const auto // Not supported in C:
// 
helper_57 = helper_2.getHessian()(1,1);
const auto helper_58 = helper_57*helper_9;
const auto // Not supported in C:
// 
helper_59 = helper_0.getHessian()(1,1);
const auto helper_60 = helper_11*helper_59;
const auto helper_61 = helper_13*helper_47;
const auto helper_62 = -helper_56 - helper_58 + helper_60 + helper_61;
const auto // Not supported in C:
// 
helper_63 = helper_2.getHessian()(0,0);
const auto helper_64 = helper_49*helper_63/pow(helper_11, 2);
const auto helper_65 = helper_17*helper_26;
const auto helper_66 = helper_21*helper_3;
const auto helper_67 = helper_24*helper_8;
const auto helper_68 = -helper_66 + helper_67;
const auto helper_69 = helper_50*helper_68 + helper_65;
const auto // Not supported in C:
// 
helper_70 = helper_0.getHessian()(0,0);
const auto helper_71 = helper_3*helper_70;
const auto helper_72 = helper_47*helper_9;
const auto helper_73 = helper_11*helper_55;
const auto helper_74 = helper_13*helper_63;
const auto helper_75 = -helper_71 - helper_72 + helper_73 + helper_74;
const auto helper_76 = pow(helper_26, -2);
const auto helper_77 = helper_21*helper_37;
const auto helper_78 = helper_1*helper_30;
const auto helper_79 = helper_28*helper_8;
const auto // Not supported in C:
// 
helper_80 = helper_2.getHessian()(2,2);
const auto helper_81 = helper_80*helper_9;
const auto // Not supported in C:
// 
helper_82 = helper_0.getHessian()(2,2);
const auto helper_83 = helper_23*(helper_11*helper_82 + helper_78 - helper_79 - helper_81);
const auto helper_84 = helper_29*helper_8;
const auto helper_85 = helper_5*helper_80;
const auto // Not supported in C:
// 
helper_86 = helper_4.getHessian()(2,2);
const auto helper_87 = helper_11*helper_86;
const auto helper_88 = helper_24*(helper_15*helper_30 - helper_84 - helper_85 + helper_87);
const auto helper_89 = helper_25*helper_44;
const auto helper_90 = -helper_77 - helper_83 + helper_88 + helper_89;
const auto helper_91 = helper_50*helper_76*helper_90;
const auto helper_92 = 1.0/helper_18;
const auto helper_93 = helper_27*helper_92*lambda;
const auto helper_94 = helper_1*helper_31 + helper_1*helper_33 + helper_10*helper_34 - helper_12*helper_34 - helper_13*helper_84 - helper_13*helper_85 + helper_13*helper_87 - helper_14*helper_39 - helper_15*helper_38 + helper_15*helper_40 - helper_15*helper_42 + helper_15*helper_43 - helper_16*helper_86 - helper_22*helper_82 + helper_6*helper_82 - helper_7*helper_78 + helper_7*helper_79 + helper_7*helper_81;
const auto helper_95 = helper_49*helper_76;
const auto // Not supported in C:
// 
helper_96 = helper_4.getHessian()(0,1);
const auto helper_97 = helper_3*helper_96;
const auto helper_98 = helper_5*helper_57;
const auto // Not supported in C:
// 
helper_99 = helper_4.getHessian()(1,1);
const auto helper_100 = helper_11*helper_99 + helper_47*helper_7 - helper_97 - helper_98;
const auto helper_101 = helper_100*helper_21;
const auto helper_102 = helper_1*helper_47;
const auto helper_103 = helper_55*helper_8;
const auto helper_104 = helper_102 - helper_103 + helper_41;
const auto helper_105 = helper_104*helper_23;
const auto helper_106 = helper_8*helper_96;
const auto helper_107 = helper_24*(-helper_106 + helper_15*helper_47 + helper_36);
const auto helper_108 = helper_25*helper_62;
const auto helper_109 = -helper_101 - helper_105 + helper_107 + helper_108;
const auto helper_110 = helper_1*helper_97 + helper_1*helper_98 + helper_10*helper_99 - helper_102*helper_7 + helper_103*helper_7 - helper_106*helper_13 - helper_12*helper_99 - helper_13*helper_33 + helper_13*helper_35 - helper_14*helper_59 - helper_15*helper_56 - helper_15*helper_58 + helper_15*helper_60 + helper_15*helper_61 - helper_16*helper_34 - helper_22*helper_39 + helper_38*helper_7 + helper_39*helper_6;
const auto // Not supported in C:
// 
helper_111 = helper_4.getHessian()(0,0);
const auto helper_112 = helper_111*helper_3;
const auto helper_113 = helper_11*helper_96 - helper_112 - helper_47*helper_5 + helper_63*helper_7;
const auto helper_114 = helper_113*helper_21;
const auto helper_115 = helper_1*helper_63;
const auto helper_116 = helper_70*helper_8;
const auto helper_117 = helper_30*helper_9;
const auto helper_118 = helper_11*helper_28 + helper_115 - helper_116 - helper_117;
const auto helper_119 = helper_118*helper_23;
const auto helper_120 = helper_111*helper_8;
const auto helper_121 = helper_30*helper_5;
const auto helper_122 = helper_11*helper_29;
const auto helper_123 = helper_24*(-helper_120 - helper_121 + helper_122 + helper_15*helper_63);
const auto helper_124 = helper_25*helper_75;
const auto helper_125 = -helper_114 - helper_119 + helper_123 + helper_124;
const auto helper_126 = helper_125*helper_49*helper_76;
const auto helper_127 = helper_48*helper_49;
const auto helper_128 = helper_1*helper_112 + helper_10*helper_96 + helper_102*helper_5 - helper_115*helper_7 + helper_116*helper_7 + helper_117*helper_7 - helper_12*helper_96 - helper_120*helper_13 - helper_121*helper_13 + helper_122*helper_13 - helper_14*helper_55 - helper_15*helper_71 - helper_15*helper_72 + helper_15*helper_73 + helper_15*helper_74 - helper_16*helper_29 - helper_22*helper_28 + helper_28*helper_6;
const auto helper_129 = helper_19*helper_27*helper_48*helper_49*lambda;
const auto helper_130 = helper_21*helper_47;
const auto helper_131 = helper_130*helper_50;
const auto helper_132 = helper_26*helper_72;
const auto helper_133 = helper_26*helper_73;
const auto helper_134 = helper_104*helper_11;
const auto helper_135 = helper_134*helper_50;
const auto helper_136 = helper_11*helper_21;
const auto helper_137 = helper_136*(-helper_100*helper_9 - helper_23*helper_55 + helper_24*helper_96 + helper_5*helper_62);
const auto helper_138 = helper_109*helper_11;
const auto helper_139 = helper_138*helper_9;
const auto helper_140 = helper_111*helper_24 - helper_113*helper_9 - helper_23*helper_70 + helper_5*helper_75;
const auto helper_141 = helper_24*helper_30;
const auto helper_142 = helper_118*helper_3;
const auto helper_143 = helper_75*helper_8;
const auto helper_144 = -helper_130 + helper_141 - helper_142 + helper_143;
const auto helper_145 = helper_125*helper_17 + helper_133 + helper_26*helper_74;
const auto helper_146 = helper_51 - helper_52;
const auto helper_147 = helper_146*helper_27;
const auto helper_148 = helper_56 + helper_58 - helper_60 - helper_61;
const auto helper_149 = helper_101 + helper_105 - helper_107 - helper_108;
const auto helper_150 = helper_27*helper_48*helper_49;
const auto helper_151 = helper_66 - helper_67;
const auto helper_152 = -helper_151*helper_50 + helper_65;
const auto helper_153 = helper_114 + helper_119 - helper_123 - helper_124;
const auto helper_154 = helper_130 - helper_141 + helper_142 - helper_143;
const auto helper_155 = helper_23*helper_27*helper_30;
const auto helper_156 = helper_11*helper_27;
const auto helper_157 = helper_156*helper_37;
const auto helper_158 = helper_151*helper_23 + helper_26*helper_3;
const auto helper_159 = helper_158*helper_27*helper_54*helper_75;
const auto helper_160 = helper_11*helper_76;
const auto helper_161 = helper_160*helper_23*helper_90;
const auto helper_162 = helper_11*helper_27*helper_92*helper_94*lambda;
const auto helper_163 = helper_11*helper_110*helper_27*helper_92*lambda;
const auto helper_164 = helper_158*helper_49;
const auto helper_165 = helper_125*helper_19*helper_76*lambda;
const auto helper_166 = helper_128*helper_27*helper_92*lambda;
const auto helper_167 = helper_19*helper_27*helper_49*lambda;
const auto helper_168 = helper_26*helper_47;
const auto helper_169 = helper_101*helper_11 + helper_105*helper_11 + helper_130*helper_23 + helper_138 + helper_168;
const auto helper_170 = helper_113*helper_151 + helper_125*helper_3 + helper_154*helper_23 + helper_168;
const auto helper_171 = helper_27*helper_49;
const auto helper_172 = helper_130*helper_27;
const auto helper_173 = helper_141*helper_27;
const auto helper_174 = helper_134*helper_27;
const auto helper_175 = helper_156*helper_44;
const auto helper_176 = helper_109*helper_136*helper_76;
const auto helper_177 = helper_11*helper_24*helper_76;
res(0) = helper_109*helper_19*helper_53*helper_95*lambda - helper_110*helper_49*helper_53*helper_93 - helper_126*helper_19*helper_48*helper_69*lambda + helper_127*helper_128*helper_27*helper_69*helper_92*lambda + helper_129*(helper_140*helper_68 + helper_144*helper_50 + helper_145) - helper_129*(-helper_131 + helper_132 + helper_133 - helper_135 - helper_137 + helper_139) - helper_19*helper_27*helper_48*helper_54*helper_69*helper_75*lambda + helper_19*helper_27*helper_53*helper_54*helper_62*lambda - helper_19*helper_27*helper_64*helper_69*lambda - helper_20*helper_45 + helper_20*helper_91 + helper_46*helper_47*helper_48*helper_49*helper_53 - helper_50*helper_93*helper_94 + mu*(helper_45 + helper_80 - helper_91) - mu*(helper_146*helper_149*helper_95 + helper_147*helper_148*helper_54 - helper_147*helper_47*helper_48*helper_49 + helper_150*(helper_131 - helper_132 - helper_133 + helper_135 + helper_137 - helper_139) - helper_57) - mu*(helper_127*helper_152*helper_153*helper_76 + helper_150*(-helper_140*helper_151 + helper_145 - helper_154*helper_50) + helper_152*helper_27*helper_48*helper_54*(helper_71 + helper_72 - helper_73 - helper_74) - helper_152*helper_27*helper_64 - helper_63);
res(1) = -helper_109*helper_11*helper_19*helper_25*helper_76*lambda - helper_11*helper_20*helper_25*helper_27*helper_49*helper_62 - helper_155*helper_20 - helper_157*helper_20 + helper_159*helper_20 + helper_161*helper_20 - helper_162*helper_23 + helper_163*helper_25 + helper_164*helper_165 - helper_164*helper_166 + helper_167*helper_169 - helper_167*helper_170 + mu*(helper_155 + helper_157 - helper_161 + helper_82) + mu*(-helper_126*helper_158 - helper_159 + helper_170*helper_171 + helper_70) - mu*(helper_11*helper_148*helper_25*helper_27*helper_49 + helper_149*helper_160*helper_25 + helper_169*helper_171 - helper_59);
res(2) = -helper_144*helper_46 + helper_162*helper_24 - helper_163*helper_21 + helper_165*helper_68 - helper_166*helper_68 - helper_172*helper_20 + helper_173*helper_20 - helper_174*helper_20 + helper_175*helper_20 + helper_176*helper_20 - helper_177*helper_19*helper_90*lambda - mu*(-helper_111 + helper_151*helper_153*helper_76 + helper_154*helper_27) + mu*(helper_172 + helper_174 - helper_176 + helper_99) - mu*(helper_173 + helper_175 + helper_177*(helper_77 + helper_83 - helper_88 - helper_89) - helper_86);
}



}}
