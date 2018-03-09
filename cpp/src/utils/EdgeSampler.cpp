#include "EdgeSampler.hpp"

namespace poly_fem {
	//TODO sample hex edges correcly!!!!
	void EdgeSampler::sample_2d(const int resolution, Eigen::MatrixXd &samples)
	{
		samples.resize(4*resolution, 2);

		const Eigen::MatrixXd t = Eigen::VectorXd::LinSpaced(resolution, 0, 1);
		samples.setConstant(-1);

		int n = 0;
		samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);
		samples.block(n, 1, resolution, 1) = t;
		n += resolution;

		samples.block(n, 0, resolution, 1) = t;
		samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);
		n += resolution;

		samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);
		samples.block(n, 1, resolution, 1) = t;
		n += resolution;

		samples.block(n, 0, resolution, 1) = t;
		samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);
		n += resolution;

		assert(long(n) == samples.rows());
		assert(samples.minCoeff()  >= 0);
		assert(samples.maxCoeff()  <= 1);
	}


	void EdgeSampler::sample_3d(const int resolution, Eigen::MatrixXd &samples)
	{
		samples.resize(12*resolution, 3);
		const Eigen::MatrixXd t = Eigen::VectorXd::LinSpaced(resolution, 0, 1);
			//X
		int ii = 0;
		samples.block(ii*resolution, 0, resolution, 1) = t;
		samples.block(ii*resolution, 1, resolution, 1).setZero();
		samples.block(ii*resolution, 2, resolution, 1).setZero();

		++ii;
		samples.block(ii*resolution, 0, resolution, 1) = t;
		samples.block(ii*resolution, 1, resolution, 1).setOnes();
		samples.block(ii*resolution, 2, resolution, 1).setZero();

		++ii;
		samples.block(ii*resolution, 0, resolution, 1) = t;
		samples.block(ii*resolution, 1, resolution, 1).setZero();
		samples.block(ii*resolution, 2, resolution, 1).setOnes();

		++ii;
		samples.block(ii*resolution, 0, resolution, 1) = t;
		samples.block(ii*resolution, 1, resolution, 1).setOnes();
		samples.block(ii*resolution, 2, resolution, 1).setOnes();

			//Y
		++ii;
		samples.block(ii*resolution, 0, resolution, 1).setZero();
		samples.block(ii*resolution, 1, resolution, 1) = t;
		samples.block(ii*resolution, 2, resolution, 1).setZero();

		++ii;
		samples.block(ii*resolution, 0, resolution, 1).setOnes();
		samples.block(ii*resolution, 1, resolution, 1) = t;
		samples.block(ii*resolution, 2, resolution, 1).setZero();

		++ii;
		samples.block(ii*resolution, 0, resolution, 1).setZero();
		samples.block(ii*resolution, 1, resolution, 1) = t;
		samples.block(ii*resolution, 2, resolution, 1).setOnes();

		++ii;
		samples.block(ii*resolution, 0, resolution, 1).setOnes();
		samples.block(ii*resolution, 1, resolution, 1) = t;
		samples.block(ii*resolution, 2, resolution, 1).setOnes();

			//Z
		++ii;
		samples.block(ii*resolution, 0, resolution, 1).setZero();
		samples.block(ii*resolution, 1, resolution, 1).setZero();
		samples.block(ii*resolution, 2, resolution, 1) = t;

		++ii;
		samples.block(ii*resolution, 0, resolution, 1).setOnes();
		samples.block(ii*resolution, 1, resolution, 1).setZero();
		samples.block(ii*resolution, 2, resolution, 1) = t;

		++ii;
		samples.block(ii*resolution, 0, resolution, 1).setZero();
		samples.block(ii*resolution, 1, resolution, 1).setOnes();
		samples.block(ii*resolution, 2, resolution, 1) = t;

		++ii;
		samples.block(ii*resolution, 0, resolution, 1).setOnes();
		samples.block(ii*resolution, 1, resolution, 1).setOnes();
		samples.block(ii*resolution, 2, resolution, 1) = t;
	}
}
