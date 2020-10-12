#include <polyfem/ElasticityUtils.hpp>
#include <polyfem/MatrixUtils.hpp>
#include <polyfem/Logger.hpp>

namespace polyfem
{
	Eigen::VectorXd gradient_from_energy(const int size, const int n_bases, const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 6, 1>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun6,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 8, 1>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun8,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 12, 1>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun12,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 18, 1>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun18,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 24, 1>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun24,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 30, 1>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun30,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 60, 1>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun60,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 81, 1>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun81,
										 const std::function<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funN,
										 const std::function<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, BIG_N, 1>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funBigN,
										 const std::function<DScalar1<double, Eigen::VectorXd>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funn)
	{
		Eigen::VectorXd grad;

		switch (size * n_bases)
		{
		case 6:
		{
			auto auto_diff_energy = fun6(vals, displacement, da);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 8:
		{
			auto auto_diff_energy = fun8(vals, displacement, da);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 12:
		{
			auto auto_diff_energy = fun12(vals, displacement, da);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 18:
		{
			auto auto_diff_energy = fun18(vals, displacement, da);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 24:
		{
			auto auto_diff_energy = fun24(vals, displacement, da);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 30:
		{
			auto auto_diff_energy = fun30(vals, displacement, da);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 60:
		{
			auto auto_diff_energy = fun60(vals, displacement, da);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 81:
		{
			auto auto_diff_energy = fun81(vals, displacement, da);
			grad = auto_diff_energy.getGradient();
			break;
		}
		}

		if (grad.size() <= 0)
		{
			if (n_bases * size <= SMALL_N)
			{
				auto auto_diff_energy = funN(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
			}
			else if (n_bases * size <= BIG_N)
			{
				auto auto_diff_energy = funBigN(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
			}
			else
			{
				static bool show_message = true;

				if (show_message)
				{
					logger().debug("[Warning] grad {}^{} not using static sizes", n_bases, size);
					show_message = false;
				}

				auto auto_diff_energy = funn(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
			}
		}

		return grad;
	}

	Eigen::MatrixXd hessian_from_energy(const int size, const int n_bases, const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da,
										const std::function<DScalar2<double, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun6,
										const std::function<DScalar2<double, Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun8,
										const std::function<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun12,
										const std::function<DScalar2<double, Eigen::Matrix<double, 18, 1>, Eigen::Matrix<double, 18, 18>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun18,
										const std::function<DScalar2<double, Eigen::Matrix<double, 24, 1>, Eigen::Matrix<double, 24, 24>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun24,
										const std::function<DScalar2<double, Eigen::Matrix<double, 30, 1>, Eigen::Matrix<double, 30, 30>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun30,
										const std::function<DScalar2<double, Eigen::Matrix<double, 60, 1>, Eigen::Matrix<double, 60, 60>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun60,
										const std::function<DScalar2<double, Eigen::Matrix<double, 81, 1>, Eigen::Matrix<double, 81, 81>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun81,
										const std::function<DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, SMALL_N, SMALL_N>>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funN,
										const std::function<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funn)
	{
		Eigen::MatrixXd hessian;

		switch (size * n_bases)
		{
		case 6:
		{
			auto auto_diff_energy = fun6(vals, displacement, da);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 8:
		{
			auto auto_diff_energy = fun8(vals, displacement, da);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 12:
		{
			auto auto_diff_energy = fun12(vals, displacement, da);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 18:
		{
			auto auto_diff_energy = fun18(vals, displacement, da);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 24:
		{
			auto auto_diff_energy = fun24(vals, displacement, da);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 30:
		{
			auto auto_diff_energy = fun30(vals, displacement, da);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 60:
		{
			auto auto_diff_energy = fun60(vals, displacement, da);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 81:
		{
			auto auto_diff_energy = fun81(vals, displacement, da);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		}

		if (hessian.size() <= 0)
		{
			// #ifndef POLYFEM_ON_HPC
			// Somehow causes a segfault on the HPC (objects too big for the stack?)
			if (n_bases * size <= SMALL_N)
			{
				auto auto_diff_energy = funN(vals, displacement, da);
				hessian = auto_diff_energy.getHessian();
			}
			else
			// #endif
			{
				static bool show_message = true;

				if (show_message)
				{
					logger().debug("[Warning] hessian {}*{} not using static sizes", n_bases, size);
					show_message = false;
				}

				auto auto_diff_energy = funn(vals, displacement, da);
				hessian = auto_diff_energy.getHessian();
			}
		}

		// time.stop();
		// std::cout << "-- hessian: " << time.getElapsedTime() << std::endl;

		return hessian;
	}

	double convert_to_lambda(const bool is_volume, const double E, const double nu)
	{
		if (is_volume)
			return (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));

		return (nu * E) / (1.0 - nu * nu);
	}

	double convert_to_mu(const double E, const double nu)
	{
		return E / (2.0 * (1.0 + nu));
	}

	void compute_diplacement_grad(const int size, const ElementBases &bs, const ElementAssemblyValues &vals, const Eigen::MatrixXd &local_pts, const int p, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &displacement_grad)
	{
		assert(displacement.cols() == 1);

		displacement_grad.setZero();

		for (std::size_t j = 0; j < bs.bases.size(); ++j)
		{
			const Basis &b = bs.bases[j];
			const auto &loc_val = vals.basis_values[j];

			assert(bs.bases.size() == vals.basis_values.size());
			assert(loc_val.grad.rows() == local_pts.rows());
			assert(loc_val.grad.cols() == size);

			for (int d = 0; d < size; ++d)
			{
				for (std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					displacement_grad.row(d) += b.global()[ii].val * loc_val.grad.row(p) * displacement(b.global()[ii].index * size + d);
				}
			}
		}

		displacement_grad = (displacement_grad * vals.jac_it[p]).eval();
	}

	double von_mises_stress_for_stress_tensor(const Eigen::MatrixXd &stress)
	{
		double von_mises_stress;

		if (stress.rows() == 3)
		{
			von_mises_stress = 0.5 * (stress(0, 0) - stress(1, 1)) * (stress(0, 0) - stress(1, 1)) + 3.0 * stress(0, 1) * stress(1, 0);
			von_mises_stress += 0.5 * (stress(2, 2) - stress(1, 1)) * (stress(2, 2) - stress(1, 1)) + 3.0 * stress(2, 1) * stress(2, 1);
			von_mises_stress += 0.5 * (stress(2, 2) - stress(0, 0)) * (stress(2, 2) - stress(0, 0)) + 3.0 * stress(2, 0) * stress(2, 0);
		}
		else
		{
			// von_mises_stress = ( stress(0, 0) - stress(1, 1) ) * ( stress(0, 0) - stress(1, 1) ) + 3.0  *  stress(0, 1) * stress(1, 0);
			von_mises_stress = stress(0, 0) * stress(0, 0) - stress(0, 0) * stress(1, 1) + stress(1, 1) * stress(1, 1) + 3.0 * stress(0, 1) * stress(1, 0);
		}

		von_mises_stress = sqrt(fabs(von_mises_stress));

		return von_mises_stress;
	}

	void ElasticityTensor::resize(const int size)
	{
		if (size == 2)
			stifness_tensor_.resize(3, 3);
		else
			stifness_tensor_.resize(6, 6);

		stifness_tensor_.setZero();

		size_ = size;
	}

	double ElasticityTensor::operator()(int i, int j) const
	{
		if (j < i)
		{
			std::swap(i, j);
		}

		assert(j >= i);
		return stifness_tensor_(i, j);
	}

	double &ElasticityTensor::operator()(int i, int j)
	{
		if (j < i)
		{
			std::swap(i, j);
		}

		assert(j >= i);
		return stifness_tensor_(i, j);
	}

	void ElasticityTensor::set_from_entries(const std::vector<double> &entries)
	{
		if (size_ == 2)
		{
			if (entries.size() == 4)
			{
				set_orthotropic(
					entries[0],
					entries[1],
					entries[2],
					entries[3]);

				return;
			}

			assert(entries.size() >= 6);

			(*this)(0, 0) = entries[0];
			(*this)(0, 1) = entries[1];
			(*this)(0, 2) = entries[2];

			(*this)(1, 1) = entries[3];
			(*this)(1, 2) = entries[4];

			(*this)(2, 2) = entries[5];
		}
		else
		{
			if (entries.size() == 9)
			{
				set_orthotropic(
					entries[0],
					entries[1],
					entries[2],
					entries[3],
					entries[4],
					entries[5],
					entries[6],
					entries[7],
					entries[8]);

				return;
			}
			assert(entries.size() >= 21);

			(*this)(0, 0) = entries[0];
			(*this)(0, 1) = entries[1];
			(*this)(0, 2) = entries[2];
			(*this)(0, 3) = entries[3];
			(*this)(0, 4) = entries[4];
			(*this)(0, 5) = entries[5];

			(*this)(1, 1) = entries[6];
			(*this)(1, 2) = entries[7];
			(*this)(1, 3) = entries[8];
			(*this)(1, 4) = entries[9];
			(*this)(1, 5) = entries[10];

			(*this)(2, 2) = entries[11];
			(*this)(2, 3) = entries[12];
			(*this)(2, 4) = entries[13];
			(*this)(2, 5) = entries[14];

			(*this)(3, 3) = entries[15];
			(*this)(3, 4) = entries[16];
			(*this)(3, 5) = entries[17];

			(*this)(4, 4) = entries[18];
			(*this)(4, 5) = entries[19];

			(*this)(5, 5) = entries[20];
		}
	}

	void ElasticityTensor::set_from_lambda_mu(const double lambda, const double mu)
	{
		if (size_ == 2)
		{
			(*this)(0, 0) = 2 * mu + lambda;
			(*this)(0, 1) = lambda;
			(*this)(0, 2) = 0;

			(*this)(1, 1) = 2 * mu + lambda;
			(*this)(1, 2) = 0;

			(*this)(2, 2) = mu;
		}
		else
		{
			(*this)(0, 0) = 2 * mu + lambda;
			(*this)(0, 1) = lambda;
			(*this)(0, 2) = lambda;
			(*this)(0, 3) = 0;
			(*this)(0, 4) = 0;
			(*this)(0, 5) = 0;

			(*this)(1, 1) = 2 * mu + lambda;
			(*this)(1, 2) = lambda;
			(*this)(1, 3) = 0;
			(*this)(1, 4) = 0;
			(*this)(1, 5) = 0;

			(*this)(2, 2) = 2 * mu + lambda;
			(*this)(2, 3) = 0;
			(*this)(2, 4) = 0;
			(*this)(2, 5) = 0;

			(*this)(3, 3) = mu;
			(*this)(3, 4) = 0;
			(*this)(3, 5) = 0;

			(*this)(4, 4) = mu;
			(*this)(4, 5) = 0;

			(*this)(5, 5) = mu;
		}
	}

	void ElasticityTensor::set_from_young_poisson(const double young, const double nu)
	{
		if (size_ == 2)
		{
			stifness_tensor_ << 1.0, nu, 0.0,
				nu, 1.0, 0.0,
				0.0, 0.0, (1.0 - nu) / 2.0;
			stifness_tensor_ *= young / (1.0 - nu * nu);
		}
		else
		{
			assert(size_ == 3);
			const double v = nu;
			stifness_tensor_ << 1. - v, v, v, 0, 0, 0,
				v, 1. - v, v, 0, 0, 0,
				v, v, 1. - v, 0, 0, 0,
				0, 0, 0, (1. - 2. * v) / 2., 0, 0,
				0, 0, 0, 0, (1. - 2. * v) / 2., 0,
				0, 0, 0, 0, 0, (1. - 2. * v) / 2.;
			stifness_tensor_ *= young / ((1. + v) * (1. - 2. * v));
		}
	}

	void ElasticityTensor::set_orthotropic(
		double Ex, double Ey, double Ez,
		double nuYX, double nuZX, double nuZY,
		double muYZ, double muZX, double muXY)
	{
		//copied from Julian
		assert(size_ == 3);
		// Note: this isn't the flattened compliance tensor! Rather, it is the
		// matrix inverse of the flattened elasticity tensor. See the tensor
		// flattening writeup.
		stifness_tensor_ << 1.0 / Ex, -nuYX / Ey, -nuZX / Ez, 0.0, 0.0, 0.0,
			0.0, 1.0 / Ey, -nuZY / Ez, 0.0, 0.0, 0.0,
			0.0, 0.0, 1.0 / Ez, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 1.0 / muYZ, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 1.0 / muZX, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / muXY;
	}

	void ElasticityTensor::set_orthotropic(double Ex, double Ey, double nuYX, double muXY)
	{
		//copied from Julian
		assert(size_ == 2);
		// Note: this isn't the flattened compliance tensor! Rather, it is the
		// matrix inverse of the flattened elasticity tensor.
		stifness_tensor_ << 1.0 / Ex, -nuYX / Ey, 0.0,
			0.0, 1.0 / Ey, 0.0,
			0.0, 0.0, 1.0 / muXY;
	}

	template <int DIM>
	double ElasticityTensor::compute_stress(const std::array<double, DIM> &strain, const int j) const
	{
		double res = 0;

		for (int k = 0; k < DIM; ++k)
			res += (*this)(j, k) * strain[k];

		return res;
	}

	LameParameters::~LameParameters()
	{
		te_free(lambda_expr_);
		te_free(mu_expr_);
		delete vals_;
	}

	LameParameters::LameParameters()
	{
		lambda_expr_ = nullptr;
		mu_expr_ = nullptr;
		vals_ = new Internal();
		initialized_ = false;
	}

	void LameParameters::lambda_mu(double x, double y, double z, int el_id, double &lambda, double &mu) const
	{
		if (lambda_expr_)
		{
			assert(mu_expr_);
			vals_->x = x;
			vals_->y = y;
			vals_->z = z;

			double tmpl = te_eval(lambda_expr_);
			double tmpm = te_eval(mu_expr_);
			if (!is_lambda_mu_)
			{
				lambda = convert_to_lambda(size_ == 3, tmpl, tmpm);
				mu = convert_to_mu(tmpl, tmpm);
			}
			else
			{
				lambda = tmpl;
				mu = tmpm;
			}
		}
		else if (lambda_mat_.size() > 0)
		{
			assert(mu_mat_.size() > 0);

			lambda = lambda_mat_(el_id);
			mu = mu_mat_(el_id);
		}
		else
		{
			lambda = lambda_;
			mu = mu_;
		}
	}

	double iflargerthanzerothenelse(double check, double ttrue, double ffalse)
	{
		return check >= 0 ? ttrue : ffalse;
	}

	void LameParameters::init_multimaterial(Eigen::MatrixXd &Es, Eigen::MatrixXd &nus)
	{
		lambda_mat_.resize(Es.size(), 1);
		mu_mat_.resize(nus.size(), 1);
		assert(lambda_mat_.size() == mu_mat_.size());

		for (int i = 0; i < lambda_mat_.size(); ++i)
		{
			lambda_mat_(i) = convert_to_lambda(size_ == 3, Es(i), nus(i));
			mu_mat_(i) = convert_to_mu(Es(i), nus(i));
		}

		lambda_ = -1;
		mu_ = -1;
		initialized_ = true;
	}

	void LameParameters::init(const json &params)
	{
		size_ = params["size"];
		te_free(lambda_expr_);
		te_free(mu_expr_);
		lambda_expr_ = nullptr;
		mu_expr_ = nullptr;

		if (initialized_)
			return;

		if (params.count("young"))
		{
			set_e_nu(params["young"], params["nu"]);
		}
		else if (params.count("E"))
		{
			set_e_nu(params["E"], params["nu"]);
		}
		else
		{
			if (params["lambda"].is_number())
			{
				assert(params["mu"].is_number());

				lambda_ = params["lambda"];
				mu_ = params["mu"];

				lambda_mat_.resize(0, 0);
				mu_mat_.resize(0, 0);
			}
			else if (params["lambda"].is_array())
			{
				lambda_mat_.resize(params["lambda"].size(), 1);
				mu_mat_.resize(params["mu"].size(), 1);

				assert(lambda_mat_.size() == mu_mat_.size());

				for (int i = 0; i < lambda_mat_.size(); ++i)
				{
					lambda_mat_(i) = params["lambda"][i];
					mu_mat_(i) = params["mu"][i];
				}

				lambda_ = -1;
				mu_ = -1;
			}
			else if (params["lambda"].is_string())
			{
				read_matrix(params["lambda"], lambda_mat_);
				read_matrix(params["mu"], mu_mat_);

				assert(lambda_mat_.size() == mu_mat_.size());

				lambda_ = -1;
				mu_ = -1;
			}
			else
			{
				te_variable vars[4];
				vars[0] = {"x", &vals_->x};
				vars[1] = {"y", &vals_->y};
				vars[2] = {"z", &vals_->z};
				vars[3].name = "if";
				vars[3].address = (void *)&iflargerthanzerothenelse;
				vars[3].type = TE_FUNCTION3;

				assert(params["lambda"].is_string());
				assert(params["mu"].is_string());

				const std::string lambdas = params["lambda"];
				const std::string mus = params["mu"];

				int err;
				lambda_expr_ = te_compile(lambdas.c_str(), vars, 4, &err);
				mu_expr_ = te_compile(mus.c_str(), vars, 4, &err);

				is_lambda_mu_ = true;

				if (!lambda_expr_)
				{
					logger().error("Unable to parse {}, error, {}", lambdas, err);

					assert(false);
				}

				if (!mu_expr_)
				{
					logger().error("Unable to parse {}, error, {}", mus, err);

					assert(false);
				}
			}
		}
	}

	void LameParameters::set_e_nu(const json &E, const json &nu)
	{
		if (E.is_number())
		{
			assert(nu.is_number());

			lambda_ = convert_to_lambda(size_ == 3, E, nu);
			mu_ = convert_to_mu(E, nu);

			lambda_mat_.resize(0, 0);
			mu_mat_.resize(0, 0);
		}
		else if (E.is_array())
		{
			lambda_mat_.resize(E.size(), 1);
			mu_mat_.resize(nu.size(), 1);
			assert(lambda_mat_.size() == mu_mat_.size());

			for (int i = 0; i < lambda_mat_.size(); ++i)
			{
				lambda_mat_(i) = convert_to_lambda(size_ == 3, E[i], nu[i]);
				mu_mat_(i) = convert_to_mu(E[i], nu[i]);
			}

			lambda_ = -1;
			mu_ = -1;
		}
		else if (E.is_string())
		{
			Eigen::MatrixXd e_mat, nu_mat;
			read_matrix(E, e_mat);
			read_matrix(nu, nu_mat);

			lambda_mat_.resize(e_mat.size(), 1);
			mu_mat_.resize(nu_mat.size(), 1);
			assert(lambda_mat_.size() == mu_mat_.size());

			for (int i = 0; i < lambda_mat_.size(); ++i)
			{
				lambda_mat_(i) = convert_to_lambda(size_ == 3, e_mat(i), nu_mat(i));
				mu_mat_(i) = convert_to_mu(e_mat(i), nu_mat(i));
			}

			lambda_ = -1;
			mu_ = -1;
		}
		else
		{
			te_variable vars[4];
			vars[0] = {"x", &vals_->x};
			vars[1] = {"y", &vals_->y};
			vars[2] = {"z", &vals_->z};
			vars[3].name = "if";
			vars[3].address = (void *)&iflargerthanzerothenelse;
			vars[3].type = TE_FUNCTION3;

			assert(E.is_string());
			assert(nu.is_string());

			const std::string Es = E;
			const std::string nus = nu;

			int err;
			lambda_expr_ = te_compile(Es.c_str(), vars, 4, &err);
			mu_expr_ = te_compile(nus.c_str(), vars, 4, &err);

			is_lambda_mu_ = false;

			if (!lambda_expr_)
			{
				logger().error("Unable to parse {}, error, {}", Es, err);

				assert(false);
			}

			if (!mu_expr_)
			{
				logger().error("Unable to parse {}, error, {}", nus, err);

				assert(false);
			}
		}
	}

	Density::~Density()
	{
		te_free(rho_expr_);
		delete vals_;
	}

	Density::Density()
	{
		rho_expr_ = nullptr;
		vals_ = new Internal();
		initialized_ = false;
	}

	double Density::operator()(double x, double y, double z, int el_id) const
	{
		if (rho_expr_)
		{
			vals_->x = x;
			vals_->y = y;
			vals_->z = z;

			return te_eval(rho_expr_);
		}
		else if (rho_mat_.size() > 0)
		{
			return rho_mat_(el_id);
		}
		else
		{
			return rho_;
		}
	}

	void Density::init_multimaterial(Eigen::MatrixXd &rho)
	{
		rho_mat_ = rho;
		rho_ = -1;
		initialized_ = true;
	}

	void Density::init(const json &params)
	{
		te_free(rho_expr_);
		rho_expr_ = nullptr;

		if (initialized_)
			return;

		if (params.count("rho"))
		{
			set_rho(params["rho"]);
		}
		else if (params.count("density"))
		{
			set_rho(params["density"]);
		}
	}

	void Density::set_rho(const json &rho)
	{
		if (rho.is_number())
		{
			rho_ = rho;
			rho_mat_.resize(0, 0);
		}
		else if (rho.is_array())
		{
			rho_mat_.resize(rho.size(), 1);

			for (int i = 0; i < rho_mat_.size(); ++i)
			{
				rho_mat_(i) = rho[i];
			}

			rho_ = -1;
		}
		else if (rho.is_string())
		{
			te_variable vars[4];
			vars[0] = {"x", &vals_->x};
			vars[1] = {"y", &vals_->y};
			vars[2] = {"z", &vals_->z};
			vars[3].name = "if";
			vars[3].address = (void *)&iflargerthanzerothenelse;
			vars[3].type = TE_FUNCTION3;

			const std::string rhos = rho;

			int err;
			rho_expr_ = te_compile(rhos.c_str(), vars, 4, &err);
			if (!rho_expr_)
			{
				read_matrix(rho, rho_mat_);
				rho_ = -1;
			}
		}
	}

	//template instantiation
	template double ElasticityTensor::compute_stress<3>(const std::array<double, 3> &strain, const int j) const;
	template double ElasticityTensor::compute_stress<6>(const std::array<double, 6> &strain, const int j) const;

} // namespace polyfem
