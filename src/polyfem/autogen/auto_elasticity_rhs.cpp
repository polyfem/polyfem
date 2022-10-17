#include <polyfem/autogen/auto_elasticity_rhs.hpp>

namespace polyfem
{
	namespace autogen
	{
		void linear_elasticity_2d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res)
		{
			res.resize(2);
			const auto // Not supported in C:
					   // f0
				helper_0 = pt(0);
			const auto // Not supported in C:
					   //
				helper_1 = helper_0.getHessian()(0, 0);
			const auto helper_2 = 2.0 * mu;
			const auto // Not supported in C:
					   // f1
				helper_3 = pt(1);
			const auto // Not supported in C:
					   //
				helper_4 = helper_3.getHessian()(0, 1);
			const auto helper_5 = 1.0 * lambda;
			const auto // Not supported in C:
					   //
				helper_6 = helper_3.getHessian()(1, 1);
			const auto // Not supported in C:
					   //
				helper_7 = helper_0.getHessian()(0, 1);
			// Not supported in C:
			//
			//
			res(0) = helper_1 * helper_2 + helper_5 * (helper_1 + helper_4) + mu * (helper_4 + helper_0.getHessian()(1, 1));
			res(1) = helper_2 * helper_6 + helper_5 * (helper_6 + helper_7) + mu * (helper_7 + helper_3.getHessian()(0, 0));
		}

		void linear_elasticity_3d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res)
		{
			res.resize(3);
			const auto // Not supported in C:
					   // f0
				helper_0 = pt(0);
			const auto // Not supported in C:
					   //
				helper_1 = helper_0.getHessian()(0, 0);
			const auto helper_2 = 2.0 * mu;
			const auto // Not supported in C:
					   // f1
				helper_3 = pt(1);
			const auto // Not supported in C:
					   //
				helper_4 = helper_3.getHessian()(0, 1);
			const auto // Not supported in C:
					   // f2
				helper_5 = pt(2);
			const auto // Not supported in C:
					   //
				helper_6 = helper_5.getHessian()(0, 2);
			const auto helper_7 = 1.0 * lambda;
			const auto // Not supported in C:
					   //
				helper_8 = helper_3.getHessian()(1, 1);
			const auto // Not supported in C:
					   //
				helper_9 = helper_0.getHessian()(0, 1);
			const auto // Not supported in C:
					   //
				helper_10 = helper_5.getHessian()(1, 2);
			const auto // Not supported in C:
					   //
				helper_11 = helper_5.getHessian()(2, 2);
			const auto // Not supported in C:
					   //
				helper_12 = helper_0.getHessian()(0, 2);
			const auto // Not supported in C:
					   //
				helper_13 = helper_3.getHessian()(1, 2);
			// Not supported in C:
			//
			//
			//
			//
			//
			//
			res(0) = helper_1 * helper_2 + helper_7 * (helper_1 + helper_4 + helper_6) + mu * (helper_4 + helper_0.getHessian()(1, 1)) + mu * (helper_6 + helper_0.getHessian()(2, 2));
			res(1) = helper_2 * helper_8 + helper_7 * (helper_10 + helper_8 + helper_9) + mu * (helper_10 + helper_3.getHessian()(2, 2)) + mu * (helper_9 + helper_3.getHessian()(0, 0));
			res(2) = helper_11 * helper_2 + helper_7 * (helper_11 + helper_12 + helper_13) + mu * (helper_12 + helper_5.getHessian()(0, 0)) + mu * (helper_13 + helper_5.getHessian()(1, 1));
		}

		void hooke_2d_function(const AutodiffHessianPt &pt, const assembler::ElasticityTensor &C, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res)
		{
			res.resize(2);
			const auto // Not supported in C:
					   // f0
				helper_0 = pt(0);
			const auto // Not supported in C:
					   //
				helper_1 = 1.0 * helper_0.getHessian()(0, 0);
			const auto // Not supported in C:
					   // C
				helper_2 = C(2, 1);
			const auto // Not supported in C:
					   // f1
				helper_3 = pt(1);
			const auto // Not supported in C:
					   //
				helper_4 = 1.0 * helper_3.getHessian()(1, 1);
			const auto // Not supported in C:
					   //
				helper_5 = helper_3.getHessian()(0, 1);
			const auto helper_6 = 1.0 * helper_5;
			const auto // Not supported in C:
					   // C
				helper_7 = C(2, 0);
			const auto // Not supported in C:
					   //
				helper_8 = helper_0.getHessian()(0, 1);
			const auto helper_9 = 1.0 * helper_8;
			const auto // Not supported in C:
					   // C
				helper_10 = C(2, 2);
			const auto // Not supported in C:
					   //
				helper_11 = helper_5 + helper_0.getHessian()(1, 1);
			const auto // Not supported in C:
					   //
				helper_12 = helper_8 + helper_3.getHessian()(0, 0);
			// Not supported in C:
			// C
			// C
			// C
			// C
			// C
			// C
			res(0) = helper_1 * C(0, 0) + helper_10 * helper_11 + helper_12 * C(0, 2) + helper_2 * helper_4 + helper_6 * C(0, 1) + helper_7 * helper_9;
			res(1) = helper_1 * helper_7 + helper_10 * helper_12 + helper_11 * C(1, 2) + helper_2 * helper_6 + helper_4 * C(1, 1) + helper_9 * C(1, 0);
		}

		void hooke_3d_function(const AutodiffHessianPt &pt, const assembler::ElasticityTensor &C, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res)
		{
			res.resize(3);
			const auto // Not supported in C:
					   // f0
				helper_0 = pt(0);
			const auto // Not supported in C:
					   //
				helper_1 = 1.0 * helper_0.getHessian()(0, 0);
			const auto // Not supported in C:
					   // C
				helper_2 = C(4, 2);
			const auto // Not supported in C:
					   // f2
				helper_3 = pt(2);
			const auto // Not supported in C:
					   //
				helper_4 = 1.0 * helper_3.getHessian()(2, 2);
			const auto // Not supported in C:
					   // C
				helper_5 = C(5, 1);
			const auto // Not supported in C:
					   // f1
				helper_6 = pt(1);
			const auto // Not supported in C:
					   //
				helper_7 = 1.0 * helper_6.getHessian()(1, 1);
			const auto // Not supported in C:
					   //
				helper_8 = helper_6.getHessian()(0, 1);
			const auto helper_9 = 1.0 * helper_8;
			const auto // Not supported in C:
					   //
				helper_10 = helper_3.getHessian()(0, 2);
			const auto helper_11 = 1.0 * helper_10;
			const auto // Not supported in C:
					   // C
				helper_12 = C(4, 0);
			const auto // Not supported in C:
					   //
				helper_13 = helper_0.getHessian()(0, 2);
			const auto helper_14 = 1.0 * helper_13;
			const auto // Not supported in C:
					   // C
				helper_15 = C(4, 1);
			const auto // Not supported in C:
					   //
				helper_16 = helper_6.getHessian()(1, 2);
			const auto helper_17 = 1.0 * helper_16;
			const auto // Not supported in C:
					   // C
				helper_18 = C(5, 0);
			const auto // Not supported in C:
					   //
				helper_19 = helper_0.getHessian()(0, 1);
			const auto helper_20 = 1.0 * helper_19;
			const auto // Not supported in C:
					   // C
				helper_21 = C(5, 2);
			const auto // Not supported in C:
					   //
				helper_22 = helper_3.getHessian()(1, 2);
			const auto helper_23 = 1.0 * helper_22;
			const auto // Not supported in C:
					   // C
				helper_24 = C(5, 5);
			const auto // Not supported in C:
					   //
				helper_25 = helper_8 + helper_0.getHessian()(1, 1);
			const auto // Not supported in C:
					   // C
				helper_26 = C(4, 4);
			const auto // Not supported in C:
					   //
				helper_27 = helper_10 + helper_0.getHessian()(2, 2);
			const auto // Not supported in C:
					   //
				helper_28 = helper_19 + helper_6.getHessian()(0, 0);
			const auto // Not supported in C:
					   // C
				helper_29 = C(4, 3);
			const auto // Not supported in C:
					   //
				helper_30 = helper_22 + helper_6.getHessian()(2, 2);
			const auto // Not supported in C:
					   //
				helper_31 = helper_13 + helper_3.getHessian()(0, 0);
			const auto // Not supported in C:
					   // C
				helper_32 = C(5, 3);
			const auto // Not supported in C:
					   //
				helper_33 = helper_16 + helper_3.getHessian()(1, 1);
			const auto // Not supported in C:
					   // C
				helper_34 = C(4, 5);
			const auto // Not supported in C:
					   //
				helper_35 = helper_0.getHessian()(1, 2);
			const auto // Not supported in C:
					   //
				helper_36 = helper_6.getHessian()(0, 2);
			const auto helper_37 = helper_35 + helper_36;
			const auto // Not supported in C:
					   // C
				helper_38 = C(5, 4);
			const auto // Not supported in C:
					   //
				helper_39 = helper_3.getHessian()(0, 1);
			const auto helper_40 = helper_35 + helper_39;
			const auto helper_41 = helper_36 + helper_39;
			const auto // Not supported in C:
					   // C
				helper_42 = C(3, 2);
			const auto // Not supported in C:
					   // C
				helper_43 = C(3, 0);
			const auto // Not supported in C:
					   // C
				helper_44 = C(3, 1);
			const auto // Not supported in C:
					   // C
				helper_45 = C(3, 4);
			const auto // Not supported in C:
					   // C
				helper_46 = C(3, 3);
			const auto // Not supported in C:
					   // C
				helper_47 = C(3, 5);
			// Not supported in C:
			// C
			// C
			// C
			// C
			// C
			// C
			// C
			// C
			// C
			// C
			// C
			// C
			// C
			// C
			// C
			// C
			// C
			// C
			res(0) = helper_1 * C(0, 0) + helper_11 * C(0, 2) + helper_12 * helper_14 + helper_15 * helper_17 + helper_18 * helper_20 + helper_2 * helper_4 + helper_21 * helper_23 + helper_24 * helper_25 + helper_26 * helper_27 + helper_28 * C(0, 5) + helper_29 * helper_30 + helper_31 * C(0, 4) + helper_32 * helper_33 + helper_34 * helper_37 + helper_38 * helper_40 + helper_41 * C(0, 3) + helper_5 * helper_7 + helper_9 * C(0, 1);
			res(1) = helper_1 * helper_18 + helper_11 * helper_21 + helper_14 * helper_43 + helper_17 * helper_44 + helper_20 * C(1, 0) + helper_23 * C(1, 2) + helper_24 * helper_28 + helper_25 * C(1, 5) + helper_27 * helper_45 + helper_30 * helper_46 + helper_31 * helper_38 + helper_32 * helper_41 + helper_33 * C(1, 3) + helper_37 * helper_47 + helper_4 * helper_42 + helper_40 * C(1, 4) + helper_5 * helper_9 + helper_7 * C(1, 1);
			res(2) = helper_1 * helper_12 + helper_11 * helper_2 + helper_14 * C(2, 0) + helper_15 * helper_9 + helper_17 * C(2, 1) + helper_20 * helper_43 + helper_23 * helper_42 + helper_25 * helper_47 + helper_26 * helper_31 + helper_27 * C(2, 4) + helper_28 * helper_34 + helper_29 * helper_41 + helper_30 * C(2, 3) + helper_33 * helper_46 + helper_37 * C(2, 5) + helper_4 * C(2, 2) + helper_40 * helper_45 + helper_44 * helper_7;
		}

		void saint_venant_2d_function(const AutodiffHessianPt &pt, const assembler::ElasticityTensor &C, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res)
		{
			res.resize(2);
			const auto // Not supported in C:
					   // f0
				helper_0 = pt(0);
			const auto // Not supported in C:
					   //
				helper_1 = helper_0.getHessian()(0, 0);
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
			const auto helper_6 = 0.5 * pow(helper_3, 2) + 1.0 * helper_3 + 0.5 * pow(helper_5, 2);
			const auto // Not supported in C:
					   // C
				helper_7 = C(0, 1);
			const auto // Not supported in C:
					   //
				helper_8 = helper_4.getGradient()(1);
			const auto // Not supported in C:
					   //
				helper_9 = helper_0.getGradient()(1);
			const auto helper_10 = 0.5 * pow(helper_8, 2) + 1.0 * helper_8 + 0.5 * pow(helper_9, 2);
			const auto // Not supported in C:
					   // C
				helper_11 = C(0, 2);
			const auto helper_12 = helper_3 * helper_9 + helper_5 * helper_8 + helper_5 + helper_9;
			const auto helper_13 = helper_10 * helper_7 + helper_11 * helper_12 + helper_2 * helper_6;
			const auto // Not supported in C:
					   //
				helper_14 = helper_0.getHessian()(1, 1);
			const auto // Not supported in C:
					   // C
				helper_15 = C(1, 0);
			const auto // Not supported in C:
					   // C
				helper_16 = C(1, 1);
			const auto // Not supported in C:
					   // C
				helper_17 = C(1, 2);
			const auto helper_18 = helper_10 * helper_16 + helper_12 * helper_17 + helper_15 * helper_6;
			const auto // Not supported in C:
					   //
				helper_19 = helper_0.getHessian()(0, 1);
			const auto // Not supported in C:
					   // C
				helper_20 = C(2, 0);
			const auto // Not supported in C:
					   // C
				helper_21 = C(2, 1);
			const auto // Not supported in C:
					   // C
				helper_22 = C(2, 2);
			const auto helper_23 = 2 * helper_10 * helper_21 + 2 * helper_12 * helper_22 + 2 * helper_20 * helper_6;
			const auto // Not supported in C:
					   //
				helper_24 = helper_4.getHessian()(0, 0);
			const auto helper_25 = 1.0 * helper_1 * helper_3 + 1.0 * helper_1 + 1.0 * helper_24 * helper_5;
			const auto // Not supported in C:
					   //
				helper_26 = helper_4.getHessian()(0, 1);
			const auto helper_27 = helper_19 * helper_9 + helper_26 * helper_8 + helper_26;
			const auto helper_28 = 1.0 * helper_27;
			const auto helper_29 = helper_19 * helper_3 + helper_19 + helper_26 * helper_5;
			const auto helper_30 = helper_1 * helper_9 + helper_24 * helper_8 + helper_24 + helper_29;
			const auto helper_31 = helper_20 * helper_25 + helper_21 * helper_28 + helper_22 * helper_30;
			const auto // Not supported in C:
					   //
				helper_32 = helper_4.getHessian()(1, 1);
			const auto helper_33 = 1.0 * helper_14 * helper_9 + 1.0 * helper_32 * helper_8 + 1.0 * helper_32;
			const auto helper_34 = 1.0 * helper_29;
			const auto helper_35 = helper_14 * helper_3 + helper_14 + helper_27 + helper_32 * helper_5;
			const auto helper_36 = helper_15 * helper_34 + helper_16 * helper_33 + helper_17 * helper_35;
			const auto helper_37 = helper_3 + 1;
			const auto helper_38 = helper_11 * helper_30 + helper_2 * helper_25 + helper_28 * helper_7;
			const auto helper_39 = helper_20 * helper_34 + helper_21 * helper_33 + helper_22 * helper_35;
			const auto helper_40 = helper_8 + 1;
			res(0) = helper_1 * helper_13 + helper_14 * helper_18 + helper_19 * helper_23 + helper_31 * helper_9 + helper_36 * helper_9 + helper_37 * helper_38 + helper_37 * helper_39;
			res(1) = helper_13 * helper_24 + helper_18 * helper_32 + helper_23 * helper_26 + helper_31 * helper_40 + helper_36 * helper_40 + helper_38 * helper_5 + helper_39 * helper_5;
		}

		void saint_venant_3d_function(const AutodiffHessianPt &pt, const assembler::ElasticityTensor &C, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res)
		{
			res.resize(3);
			const auto // Not supported in C:
					   // f0
				helper_0 = pt(0);
			const auto // Not supported in C:
					   //
				helper_1 = helper_0.getHessian()(0, 0);
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
			const auto helper_8 = 0.5 * pow(helper_3, 2) + 1.0 * helper_3 + 0.5 * pow(helper_5, 2) + 0.5 * pow(helper_7, 2);
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
			const auto helper_13 = 0.5 * pow(helper_10, 2) + 1.0 * helper_10 + 0.5 * pow(helper_11, 2) + 0.5 * pow(helper_12, 2);
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
			const auto helper_18 = 0.5 * pow(helper_15, 2) + 1.0 * helper_15 + 0.5 * pow(helper_16, 2) + 0.5 * pow(helper_17, 2);
			const auto // Not supported in C:
					   // C
				helper_19 = C(0, 5);
			const auto helper_20 = helper_10 * helper_5 + helper_11 * helper_3 + helper_11 + helper_12 * helper_7 + helper_5;
			const auto // Not supported in C:
					   // C
				helper_21 = C(0, 4);
			const auto helper_22 = helper_15 * helper_7 + helper_16 * helper_3 + helper_16 + helper_17 * helper_5 + helper_7;
			const auto // Not supported in C:
					   // C
				helper_23 = C(0, 3);
			const auto helper_24 = helper_10 * helper_17 + helper_11 * helper_16 + helper_12 * helper_15 + helper_12 + helper_17;
			const auto helper_25 = helper_13 * helper_9 + helper_14 * helper_18 + helper_19 * helper_20 + helper_2 * helper_8 + helper_21 * helper_22 + helper_23 * helper_24;
			const auto // Not supported in C:
					   //
				helper_26 = helper_0.getHessian()(1, 1);
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
			const auto helper_33 = helper_13 * helper_28 + helper_18 * helper_29 + helper_20 * helper_30 + helper_22 * helper_31 + helper_24 * helper_32 + helper_27 * helper_8;
			const auto // Not supported in C:
					   //
				helper_34 = helper_0.getHessian()(2, 2);
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
			const auto helper_41 = helper_13 * helper_36 + helper_18 * helper_37 + helper_20 * helper_38 + helper_22 * helper_39 + helper_24 * helper_40 + helper_35 * helper_8;
			const auto // Not supported in C:
					   //
				helper_42 = helper_0.getHessian()(1, 2);
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
			const auto helper_49 = 2 * helper_13 * helper_44 + 2 * helper_18 * helper_45 + 2 * helper_20 * helper_46 + 2 * helper_22 * helper_47 + 2 * helper_24 * helper_48 + 2 * helper_43 * helper_8;
			const auto // Not supported in C:
					   //
				helper_50 = helper_0.getHessian()(0, 2);
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
			const auto helper_57 = 2 * helper_13 * helper_52 + 2 * helper_18 * helper_53 + 2 * helper_20 * helper_54 + 2 * helper_22 * helper_55 + 2 * helper_24 * helper_56 + 2 * helper_51 * helper_8;
			const auto // Not supported in C:
					   //
				helper_58 = helper_0.getHessian()(0, 1);
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
			const auto helper_65 = 2 * helper_13 * helper_60 + 2 * helper_18 * helper_61 + 2 * helper_20 * helper_62 + 2 * helper_22 * helper_63 + 2 * helper_24 * helper_64 + 2 * helper_59 * helper_8;
			const auto // Not supported in C:
					   //
				helper_66 = helper_4.getHessian()(0, 0);
			const auto // Not supported in C:
					   //
				helper_67 = helper_6.getHessian()(0, 0);
			const auto helper_68 = 1.0 * helper_1 * helper_3 + 1.0 * helper_1 + 1.0 * helper_5 * helper_66 + 1.0 * helper_67 * helper_7;
			const auto // Not supported in C:
					   //
				helper_69 = helper_4.getHessian()(0, 1);
			const auto // Not supported in C:
					   //
				helper_70 = helper_6.getHessian()(0, 1);
			const auto helper_71 = helper_10 * helper_69 + helper_11 * helper_58 + helper_12 * helper_70 + helper_69;
			const auto helper_72 = 1.0 * helper_71;
			const auto // Not supported in C:
					   //
				helper_73 = helper_6.getHessian()(0, 2);
			const auto // Not supported in C:
					   //
				helper_74 = helper_4.getHessian()(0, 2);
			const auto helper_75 = helper_15 * helper_73 + helper_16 * helper_50 + helper_17 * helper_74 + helper_73;
			const auto helper_76 = 1.0 * helper_75;
			const auto helper_77 = helper_3 * helper_58 + helper_5 * helper_69 + helper_58 + helper_7 * helper_70;
			const auto helper_78 = helper_1 * helper_11 + helper_10 * helper_66 + helper_12 * helper_67 + helper_66 + helper_77;
			const auto helper_79 = helper_3 * helper_50 + helper_5 * helper_74 + helper_50 + helper_7 * helper_73;
			const auto helper_80 = helper_1 * helper_16 + helper_15 * helper_67 + helper_17 * helper_66 + helper_67 + helper_79;
			const auto helper_81 = helper_15 * helper_70 + helper_16 * helper_58 + helper_17 * helper_69 + helper_70;
			const auto helper_82 = helper_10 * helper_74 + helper_11 * helper_50 + helper_12 * helper_73 + helper_74;
			const auto helper_83 = helper_81 + helper_82;
			const auto helper_84 = helper_51 * helper_68 + helper_52 * helper_72 + helper_53 * helper_76 + helper_54 * helper_78 + helper_55 * helper_80 + helper_56 * helper_83;
			const auto helper_85 = helper_59 * helper_68 + helper_60 * helper_72 + helper_61 * helper_76 + helper_62 * helper_78 + helper_63 * helper_80 + helper_64 * helper_83;
			const auto // Not supported in C:
					   //
				helper_86 = helper_4.getHessian()(1, 1);
			const auto // Not supported in C:
					   //
				helper_87 = helper_6.getHessian()(1, 1);
			const auto helper_88 = 1.0 * helper_10 * helper_86 + 1.0 * helper_11 * helper_26 + 1.0 * helper_12 * helper_87 + 1.0 * helper_86;
			const auto helper_89 = 1.0 * helper_77;
			const auto // Not supported in C:
					   //
				helper_90 = helper_6.getHessian()(1, 2);
			const auto // Not supported in C:
					   //
				helper_91 = helper_4.getHessian()(1, 2);
			const auto helper_92 = helper_15 * helper_90 + helper_16 * helper_42 + helper_17 * helper_91 + helper_90;
			const auto helper_93 = 1.0 * helper_92;
			const auto helper_94 = helper_26 * helper_3 + helper_26 + helper_5 * helper_86 + helper_7 * helper_87 + helper_71;
			const auto helper_95 = helper_10 * helper_91 + helper_11 * helper_42 + helper_12 * helper_90 + helper_91;
			const auto helper_96 = helper_15 * helper_87 + helper_16 * helper_26 + helper_17 * helper_86 + helper_87 + helper_95;
			const auto helper_97 = helper_3 * helper_42 + helper_42 + helper_5 * helper_91 + helper_7 * helper_90;
			const auto helper_98 = helper_81 + helper_97;
			const auto helper_99 = helper_27 * helper_89 + helper_28 * helper_88 + helper_29 * helper_93 + helper_30 * helper_94 + helper_31 * helper_98 + helper_32 * helper_96;
			const auto helper_100 = helper_43 * helper_89 + helper_44 * helper_88 + helper_45 * helper_93 + helper_46 * helper_94 + helper_47 * helper_98 + helper_48 * helper_96;
			const auto // Not supported in C:
					   //
				helper_101 = helper_6.getHessian()(2, 2);
			const auto // Not supported in C:
					   //
				helper_102 = helper_4.getHessian()(2, 2);
			const auto helper_103 = 1.0 * helper_101 * helper_15 + 1.0 * helper_101 + 1.0 * helper_102 * helper_17 + 1.0 * helper_16 * helper_34;
			const auto helper_104 = 1.0 * helper_79;
			const auto helper_105 = 1.0 * helper_95;
			const auto helper_106 = helper_101 * helper_7 + helper_102 * helper_5 + helper_3 * helper_34 + helper_34 + helper_75;
			const auto helper_107 = helper_10 * helper_102 + helper_101 * helper_12 + helper_102 + helper_11 * helper_34 + helper_92;
			const auto helper_108 = helper_82 + helper_97;
			const auto helper_109 = helper_103 * helper_37 + helper_104 * helper_35 + helper_105 * helper_36 + helper_106 * helper_39 + helper_107 * helper_40 + helper_108 * helper_38;
			const auto helper_110 = helper_103 * helper_45 + helper_104 * helper_43 + helper_105 * helper_44 + helper_106 * helper_47 + helper_107 * helper_48 + helper_108 * helper_46;
			const auto helper_111 = helper_3 + 1;
			const auto helper_112 = helper_14 * helper_76 + helper_19 * helper_78 + helper_2 * helper_68 + helper_21 * helper_80 + helper_23 * helper_83 + helper_72 * helper_9;
			const auto helper_113 = helper_59 * helper_89 + helper_60 * helper_88 + helper_61 * helper_93 + helper_62 * helper_94 + helper_63 * helper_98 + helper_64 * helper_96;
			const auto helper_114 = helper_103 * helper_53 + helper_104 * helper_51 + helper_105 * helper_52 + helper_106 * helper_55 + helper_107 * helper_56 + helper_108 * helper_54;
			const auto helper_115 = helper_10 + 1;
			const auto helper_116 = helper_15 + 1;
			res(0) = helper_1 * helper_25 + helper_100 * helper_16 + helper_109 * helper_16 + helper_11 * helper_110 + helper_11 * helper_85 + helper_11 * helper_99 + helper_111 * helper_112 + helper_111 * helper_113 + helper_111 * helper_114 + helper_16 * helper_84 + helper_26 * helper_33 + helper_34 * helper_41 + helper_42 * helper_49 + helper_50 * helper_57 + helper_58 * helper_65;
			res(1) = helper_100 * helper_17 + helper_102 * helper_41 + helper_109 * helper_17 + helper_110 * helper_115 + helper_112 * helper_5 + helper_113 * helper_5 + helper_114 * helper_5 + helper_115 * helper_85 + helper_115 * helper_99 + helper_17 * helper_84 + helper_25 * helper_66 + helper_33 * helper_86 + helper_49 * helper_91 + helper_57 * helper_74 + helper_65 * helper_69;
			res(2) = helper_100 * helper_116 + helper_101 * helper_41 + helper_109 * helper_116 + helper_110 * helper_12 + helper_112 * helper_7 + helper_113 * helper_7 + helper_114 * helper_7 + helper_116 * helper_84 + helper_12 * helper_85 + helper_12 * helper_99 + helper_25 * helper_67 + helper_33 * helper_87 + helper_49 * helper_90 + helper_57 * helper_73 + helper_65 * helper_70;
		}

		void neo_hookean_2d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res)
		{
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
			const auto helper_6 = -helper_1 * helper_3 + helper_4 * helper_5;
			const auto helper_7 = pow(helper_6, -2);
			const auto // Not supported in C:
					   //
				helper_8 = helper_2.getHessian()(1, 1);
			const auto helper_9 = helper_1 * helper_8;
			const auto // Not supported in C:
					   //
				helper_10 = helper_0.getHessian()(1, 1);
			const auto helper_11 = helper_10 * helper_4;
			const auto // Not supported in C:
					   //
				helper_12 = helper_0.getHessian()(0, 1);
			const auto helper_13 = helper_12 * helper_3;
			const auto // Not supported in C:
					   //
				helper_14 = helper_2.getHessian()(0, 1);
			const auto helper_15 = helper_14 * helper_5;
			const auto helper_16 = helper_7 * (helper_11 - helper_13 + helper_15 - helper_9);
			const auto helper_17 = helper_1 * helper_16;
			const auto helper_18 = helper_17 * lambda;
			const auto // Not supported in C:
					   //
				helper_19 = helper_0.getHessian()(0, 0);
			const auto helper_20 = helper_19 * helper_3;
			const auto // Not supported in C:
					   //
				helper_21 = helper_2.getHessian()(0, 0);
			const auto helper_22 = helper_21 * helper_5;
			const auto helper_23 = helper_1 * helper_14;
			const auto helper_24 = helper_12 * helper_4;
			const auto helper_25 = -helper_20 + helper_22 - helper_23 + helper_24;
			const auto helper_26 = helper_5 * helper_7;
			const auto helper_27 = helper_25 * helper_26 * lambda;
			const auto helper_28 = log(helper_6);
			const auto helper_29 = 1.0 / helper_6;
			const auto helper_30 = helper_12 * helper_29;
			const auto helper_31 = helper_25 * helper_3 * helper_7;
			const auto helper_32 = helper_31 * lambda;
			const auto helper_33 = helper_16 * helper_4 * lambda;
			const auto helper_34 = helper_14 * helper_29;
			res(0) = helper_18 * helper_28 - helper_18 - helper_27 * helper_28 + helper_27 + mu * (-helper_17 + helper_30 + helper_8) - mu * (-helper_21 + helper_26 * (helper_20 - helper_22 + helper_23 - helper_24) + helper_30);
			res(1) = helper_28 * helper_32 - helper_28 * helper_33 - helper_32 + helper_33 - mu * (-helper_10 + helper_34 + helper_4 * helper_7 * (-helper_11 + helper_13 - helper_15 + helper_9)) + mu * (helper_19 - helper_31 + helper_34);
		}

		void neo_hookean_3d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res)
		{
			res.resize(3);
			const auto // Not supported in C:
					   // f0
				helper_0 = pt(0);
			const auto // Not supported in C:
					   //
				helper_1 = helper_0.getGradient()(2);
			const auto // Not supported in C:
					   // f1
				helper_2 = pt(1);
			const auto // Not supported in C:
					   //
				helper_3 = helper_2.getGradient()(0);
			const auto helper_4 = helper_1 * helper_3;
			const auto // Not supported in C:
					   //
				helper_5 = helper_2.getGradient()(2);
			const auto // Not supported in C:
					   //
				helper_6 = helper_0.getGradient()(0) + 1;
			const auto helper_7 = helper_5 * helper_6;
			const auto helper_8 = -helper_4 + helper_7;
			const auto // Not supported in C:
					   //
				helper_9 = helper_0.getGradient()(1);
			const auto // Not supported in C:
					   // f2
				helper_10 = pt(2);
			const auto // Not supported in C:
					   //
				helper_11 = helper_10.getGradient()(0);
			const auto helper_12 = helper_11 * helper_9;
			const auto // Not supported in C:
					   //
				helper_13 = helper_10.getGradient()(1);
			const auto helper_14 = helper_13 * helper_6;
			const auto helper_15 = -helper_12 + helper_14;
			const auto helper_16 = helper_15 * helper_8;
			const auto helper_17 = helper_3 * helper_9;
			const auto // Not supported in C:
					   //
				helper_18 = helper_2.getGradient()(1) + 1;
			const auto helper_19 = helper_18 * helper_6;
			const auto helper_20 = -helper_17 + helper_19;
			const auto helper_21 = helper_1 * helper_11;
			const auto // Not supported in C:
					   //
				helper_22 = helper_10.getGradient()(2) + 1;
			const auto helper_23 = helper_22 * helper_6;
			const auto helper_24 = -helper_21 + helper_23;
			const auto helper_25 = helper_20 * helper_24;
			const auto helper_26 = -helper_16 + helper_25;
			const auto helper_27 = 1.0 / helper_26;
			const auto // Not supported in C:
					   //
				helper_28 = helper_2.getHessian()(0, 2);
			const auto // Not supported in C:
					   //
				helper_29 = helper_10.getHessian()(0, 2);
			const auto // Not supported in C:
					   //
				helper_30 = helper_0.getHessian()(0, 2);
			const auto helper_31 = helper_13 * helper_30;
			const auto helper_32 = helper_29 * helper_9;
			const auto // Not supported in C:
					   //
				helper_33 = helper_10.getHessian()(1, 2);
			const auto // Not supported in C:
					   //
				helper_34 = helper_0.getHessian()(1, 2);
			const auto helper_35 = helper_11 * helper_34;
			const auto helper_36 = helper_33 * helper_6 - helper_35;
			const auto helper_37 = helper_31 - helper_32 + helper_36;
			const auto helper_38 = helper_18 * helper_30;
			const auto helper_39 = helper_28 * helper_9;
			const auto // Not supported in C:
					   //
				helper_40 = helper_2.getHessian()(1, 2);
			const auto helper_41 = helper_40 * helper_6;
			const auto helper_42 = helper_3 * helper_34;
			const auto helper_43 = helper_41 - helper_42;
			const auto helper_44 = helper_38 - helper_39 + helper_43;
			const auto helper_45 = helper_27 * (helper_11 * helper_44 - helper_15 * helper_28 + helper_20 * helper_29 - helper_3 * helper_37);
			const auto helper_46 = helper_12 * helper_5 + helper_13 * helper_4 - helper_13 * helper_7 - helper_17 * helper_22 - helper_18 * helper_21 + helper_19 * helper_22;
			const auto helper_47 = lambda * log(helper_46);
			const auto // Not supported in C:
					   //
				helper_48 = helper_0.getHessian()(0, 1);
			const auto helper_49 = helper_11 * helper_20 - helper_15 * helper_3;
			const auto helper_50 = helper_26 * helper_3 - helper_49 * helper_8;
			const auto helper_51 = 1.0 / helper_6;
			const auto helper_52 = 1.0 / helper_20;
			const auto helper_53 = helper_27 * helper_52;
			const auto helper_54 = helper_51 * helper_53;
			const auto helper_55 = helper_48 * helper_50 * helper_54;
			const auto // Not supported in C:
					   //
				helper_56 = helper_0.getHessian()(1, 1);
			const auto helper_57 = helper_3 * helper_56;
			const auto // Not supported in C:
					   //
				helper_58 = helper_2.getHessian()(1, 1);
			const auto helper_59 = helper_58 * helper_6;
			const auto // Not supported in C:
					   //
				helper_60 = helper_2.getHessian()(0, 1);
			const auto helper_61 = helper_60 * helper_9;
			const auto helper_62 = helper_18 * helper_48;
			const auto helper_63 = -helper_57 + helper_59 - helper_61 + helper_62;
			const auto helper_64 = helper_27 / pow(helper_20, 2);
			const auto helper_65 = helper_50 * helper_63 * helper_64;
			const auto // Not supported in C:
					   //
				helper_66 = helper_0.getHessian()(0, 0);
			const auto helper_67 = helper_8 * helper_9;
			const auto helper_68 = helper_1 * helper_20;
			const auto helper_69 = -helper_67 + helper_68;
			const auto helper_70 = helper_19 * helper_26 + helper_49 * helper_69;
			const auto helper_71 = helper_53 * helper_66 * helper_70 / pow(helper_6, 2);
			const auto // Not supported in C:
					   //
				helper_72 = helper_2.getHessian()(0, 0);
			const auto helper_73 = helper_72 * helper_9;
			const auto helper_74 = helper_18 * helper_66;
			const auto helper_75 = helper_3 * helper_48;
			const auto helper_76 = helper_6 * helper_60;
			const auto helper_77 = -helper_73 + helper_74 - helper_75 + helper_76;
			const auto helper_78 = helper_51 * helper_70;
			const auto helper_79 = helper_64 * helper_77 * helper_78;
			const auto helper_80 = pow(helper_26, -2);
			const auto // Not supported in C:
					   //
				helper_81 = helper_0.getHessian()(2, 2);
			const auto helper_82 = helper_3 * helper_81;
			const auto // Not supported in C:
					   //
				helper_83 = helper_2.getHessian()(2, 2);
			const auto helper_84 = helper_6 * helper_83;
			const auto helper_85 = helper_30 * helper_5;
			const auto helper_86 = helper_1 * helper_28;
			const auto helper_87 = helper_11 * helper_81;
			const auto // Not supported in C:
					   //
				helper_88 = helper_10.getHessian()(2, 2);
			const auto helper_89 = helper_1 * helper_29;
			const auto helper_90 = helper_20 * (helper_22 * helper_30 + helper_6 * helper_88 - helper_87 - helper_89) + helper_24 * helper_44;
			const auto helper_91 = helper_80 * (-helper_15 * (-helper_82 + helper_84 + helper_85 - helper_86) - helper_37 * helper_8 + helper_90);
			const auto helper_92 = helper_49 * helper_91;
			const auto helper_93 = helper_12 * helper_83 + helper_13 * helper_82 + helper_13 * helper_86 - helper_14 * helper_83 - helper_17 * helper_88 - helper_18 * helper_87 - helper_18 * helper_89 + helper_19 * helper_88 - helper_21 * helper_40 + helper_22 * helper_38 - helper_22 * helper_39 - helper_22 * helper_42 + helper_23 * helper_40 - helper_31 * helper_5 + helper_32 * helper_5 + helper_33 * helper_4 - helper_33 * helper_7 + helper_35 * helper_5;
			const auto helper_94 = lambda / helper_46;
			const auto helper_95 = helper_27 * helper_94;
			const auto helper_96 = helper_11 * helper_56;
			const auto // Not supported in C:
					   //
				helper_97 = helper_10.getHessian()(1, 1);
			const auto helper_98 = helper_13 * helper_48;
			const auto // Not supported in C:
					   //
				helper_99 = helper_10.getHessian()(0, 1);
			const auto helper_100 = helper_9 * helper_99;
			const auto helper_101 = -helper_100 + helper_6 * helper_97 - helper_96 + helper_98;
			const auto helper_102 = helper_101 * helper_8;
			const auto helper_103 = helper_48 * helper_5;
			const auto helper_104 = helper_1 * helper_60;
			const auto helper_105 = helper_103 - helper_104 + helper_43;
			const auto helper_106 = helper_105 * helper_15;
			const auto helper_107 = helper_1 * helper_99;
			const auto helper_108 = helper_24 * helper_63;
			const auto helper_109 = helper_108 + helper_20 * (-helper_107 + helper_22 * helper_48 + helper_36);
			const auto helper_110 = -helper_102 - helper_106 + helper_109;
			const auto helper_111 = helper_52 * helper_80;
			const auto helper_112 = helper_111 * helper_50;
			const auto helper_113 = helper_100 * helper_5 + helper_104 * helper_13 - helper_107 * helper_18 + helper_12 * helper_40 + helper_13 * helper_42 - helper_14 * helper_40 - helper_17 * helper_33 - helper_18 * helper_35 + helper_19 * helper_33 - helper_21 * helper_58 - helper_22 * helper_57 - helper_22 * helper_61 + helper_22 * helper_62 + helper_23 * helper_58 + helper_4 * helper_97 + helper_5 * helper_96 - helper_5 * helper_98 - helper_7 * helper_97;
			const auto helper_114 = helper_113 * helper_94;
			const auto helper_115 = helper_13 * helper_66;
			const auto // Not supported in C:
					   //
				helper_116 = helper_10.getHessian()(0, 0);
			const auto helper_117 = helper_116 * helper_9;
			const auto helper_118 = helper_11 * helper_48;
			const auto helper_119 = helper_6 * helper_99;
			const auto helper_120 = helper_115 - helper_117 - helper_118 + helper_119;
			const auto helper_121 = helper_5 * helper_66;
			const auto helper_122 = helper_1 * helper_72;
			const auto helper_123 = helper_3 * helper_30;
			const auto helper_124 = helper_28 * helper_6;
			const auto helper_125 = helper_121 - helper_122 - helper_123 + helper_124;
			const auto helper_126 = helper_1 * helper_116;
			const auto helper_127 = helper_20 * (-helper_11 * helper_30 - helper_126 + helper_22 * helper_66 + helper_29 * helper_6) + helper_24 * helper_77;
			const auto helper_128 = -helper_120 * helper_8 - helper_125 * helper_15 + helper_127;
			const auto helper_129 = helper_111 * helper_128 * helper_78;
			const auto helper_130 = -helper_11 * helper_38 - helper_115 * helper_5 + helper_117 * helper_5 + helper_118 * helper_5 + helper_12 * helper_28 + helper_122 * helper_13 - helper_126 * helper_18 - helper_14 * helper_28 - helper_17 * helper_29 + helper_19 * helper_29 - helper_21 * helper_60 - helper_22 * helper_73 + helper_22 * helper_74 - helper_22 * helper_75 + helper_23 * helper_60 + helper_3 * helper_31 + helper_4 * helper_99 - helper_7 * helper_99;
			const auto helper_131 = helper_130 * helper_94;
			const auto helper_132 = helper_48 * helper_8;
			const auto helper_133 = helper_132 * helper_49;
			const auto helper_134 = helper_26 * helper_75;
			const auto helper_135 = helper_26 * helper_76;
			const auto helper_136 = helper_105 * helper_6;
			const auto helper_137 = helper_136 * helper_49;
			const auto helper_138 = helper_6 * helper_8;
			const auto helper_139 = helper_138 * (-helper_101 * helper_3 + helper_11 * helper_63 - helper_15 * helper_60 + helper_20 * helper_99);
			const auto helper_140 = helper_110 * helper_6;
			const auto helper_141 = helper_140 * helper_3;
			const auto helper_142 = helper_20 * helper_30;
			const auto helper_143 = helper_125 * helper_9;
			const auto helper_144 = helper_1 * helper_77;
			const auto helper_145 = -helper_132 + helper_142 - helper_143 + helper_144;
			const auto helper_146 = helper_54 * (helper_128 * helper_19 + helper_135 + helper_145 * helper_49 + helper_26 * helper_74 + helper_69 * (helper_11 * helper_77 + helper_116 * helper_20 - helper_120 * helper_3 - helper_15 * helper_72));
			const auto helper_147 = helper_4 - helper_7;
			const auto helper_148 = helper_101 * helper_147 + helper_109 + helper_15 * (-helper_103 + helper_104 - helper_41 + helper_42);
			const auto helper_149 = helper_15 * helper_27 * helper_30;
			const auto helper_150 = helper_27 * helper_6;
			const auto helper_151 = helper_150 * helper_37;
			const auto helper_152 = helper_67 - helper_68;
			const auto helper_153 = helper_15 * helper_152 + helper_26 * helper_9;
			const auto helper_154 = helper_153 * helper_64;
			const auto helper_155 = helper_15 * helper_6 * helper_80 * (helper_147 * helper_37 + helper_15 * (helper_82 - helper_84 - helper_85 + helper_86) + helper_90);
			const auto helper_156 = helper_148 * helper_6;
			const auto helper_157 = helper_156 * helper_24 * helper_80;
			const auto helper_158 = helper_150 * helper_93 * helper_94;
			const auto helper_159 = helper_150 * helper_24;
			const auto helper_160 = helper_120 * helper_147 + helper_127 + helper_15 * (-helper_121 + helper_122 + helper_123 - helper_124);
			const auto helper_161 = helper_160 * helper_80;
			const auto helper_162 = helper_153 * helper_161 * helper_52;
			const auto helper_163 = helper_26 * helper_48;
			const auto helper_164 = helper_102 * helper_6 + helper_106 * helper_6 + helper_16 * helper_48 + helper_163;
			const auto helper_165 = helper_47 * helper_53;
			const auto helper_166 = helper_132 + helper_143;
			const auto helper_167 = -helper_142 - helper_144 + helper_166;
			const auto helper_168 = -helper_160 * helper_9;
			const auto helper_169 = helper_73 - helper_74 + helper_75 - helper_76;
			const auto helper_170 = helper_132 * helper_27;
			const auto helper_171 = helper_142 * helper_27;
			const auto helper_172 = helper_136 * helper_27;
			const auto helper_173 = helper_150 * helper_44;
			const auto helper_174 = helper_140 * helper_8 * helper_80;
			const auto helper_175 = helper_20 * helper_6 * helper_91;
			res(0) = helper_110 * helper_112 * helper_47 - helper_114 * helper_50 * helper_53 - helper_129 * helper_47 + helper_131 * helper_54 * helper_70 + helper_146 * helper_47 - helper_45 * helper_47 - helper_47 * helper_54 * (-helper_133 + helper_134 + helper_135 - helper_137 - helper_139 + helper_141) + helper_47 * helper_55 + helper_47 * helper_65 - helper_47 * helper_71 - helper_47 * helper_79 + helper_47 * helper_92 - helper_49 * helper_93 * helper_95 + mu * (helper_45 + helper_81 - helper_92) + mu * (helper_129 - helper_146 + helper_66 + helper_71 + helper_79) - mu * (helper_112 * helper_148 + helper_54 * (helper_133 - helper_134 - helper_135 + helper_137 + helper_139 - helper_141) + helper_55 - helper_56 + helper_65);
			res(1) = -helper_108 * helper_150 * helper_47 * helper_52 + helper_114 * helper_159 - helper_131 * helper_153 * helper_53 - helper_149 * helper_47 - helper_15 * helper_158 - helper_151 * helper_47 + helper_154 * helper_47 * helper_77 + helper_155 * helper_47 - helper_157 * helper_47 + helper_162 * helper_47 + helper_165 * (helper_140 + helper_164) + helper_165 * (helper_120 * helper_69 - helper_15 * helper_167 - helper_163 + helper_168) + mu * (helper_149 + helper_151 - helper_155 + helper_83) - mu * (-helper_157 + helper_159 * helper_52 * (helper_57 - helper_59 + helper_61 - helper_62) + helper_53 * (helper_156 + helper_164) - helper_58) - mu * (-helper_154 * helper_169 + helper_162 + helper_53 * (helper_152 * (-helper_115 + helper_117 + helper_118 - helper_119) + helper_168 + helper_48 * (helper_16 - helper_25) + (helper_12 - helper_14) * (helper_1 * helper_169 + helper_166 + helper_30 * (helper_17 - helper_19))) - helper_72);
			res(2) = -helper_113 * helper_138 * helper_95 + helper_128 * helper_47 * helper_69 * helper_80 - helper_130 * helper_69 * helper_95 - helper_145 * helper_27 * helper_47 + helper_158 * helper_20 - helper_170 * helper_47 + helper_171 * helper_47 - helper_172 * helper_47 + helper_173 * helper_47 + helper_174 * helper_47 - helper_175 * helper_47 - mu * (-helper_116 + helper_161 * helper_69 + helper_167 * helper_27) + mu * (helper_170 + helper_172 - helper_174 + helper_97) + mu * (-helper_171 - helper_173 + helper_175 + helper_88);
		}

	} // namespace autogen
} // namespace polyfem
