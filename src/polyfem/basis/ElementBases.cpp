#include <polyfem/basis/ElementBases.hpp>

namespace polyfem
{
	using namespace assembler;
	namespace basis
	{

		bool ElementBases::is_complete() const
		{
			for (auto &b : bases)
			{
				if (!b.is_complete())
					return false;
			}

			return true;
		}
		void ElementBases::eval_geom_mapping(const Eigen::MatrixXd &samples, Eigen::MatrixXd &mapped) const
		{
			if (!has_parameterization)
			{
				// mapped = (scaling_ * samples).rowwise() + translation_;
				mapped = samples;
				return;
			}

			mapped = Eigen::MatrixXd::Zero(samples.rows(), samples.cols());
			std::vector<AssemblyValues> tmp_val;
			evaluate_bases(samples, tmp_val);

			const int n_local_bases = int(bases.size());
			for (int j = 0; j < n_local_bases; ++j)
			{
				const Basis &b = bases[j];
				const auto &tmp = tmp_val[j].val;

				for (std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					for (long k = 0; k < tmp.size(); ++k)
					{
						mapped.row(k) += tmp(k) * b.global()[ii].node * b.global()[ii].val;
					}
				}
			}
		}

		void ElementBases::evaluate_bases_default(const Eigen::MatrixXd &uv, std::vector<AssemblyValues> &basis_values) const
		{
			basis_values.resize(bases.size());

			// val.resize(uv.rows(), bases.size());
			// Eigen::MatrixXd tmp;

			for (size_t i = 0; i < bases.size(); ++i)
			{
				// bases[i].eval_basis(uv, tmp); // = basis_values[i].val
				// val.col(i) = tmp;

				bases[i].eval_basis(uv, basis_values[i].val);
				assert(basis_values[i].val.size() == uv.rows());
			}
		}

		void ElementBases::evaluate_grads_default(const Eigen::MatrixXd &uv, std::vector<AssemblyValues> &basis_values) const
		{
			basis_values.resize(bases.size());

			// grad.resize(uv.rows(), bases.size());
			// Eigen::MatrixXd grad_tmp;

			for (size_t i = 0; i < bases.size(); ++i)
			{
				// bases[i].eval_grad(uv, grad_tmp);
				// grad.col(i) = grad_tmp.col(dim);

				bases[i].eval_grad(uv, basis_values[i].grad);
				assert(basis_values[i].grad.rows() == uv.rows());
			}
		}

		void ElementBases::eval_geom_mapping_grads(const Eigen::MatrixXd &samples, std::vector<Eigen::MatrixXd> &grads) const
		{
			grads.resize(samples.rows());

			if (!has_parameterization)
			{
				// * scaling
				std::fill(grads.begin(), grads.end(), Eigen::MatrixXd::Identity(samples.cols(), samples.cols()));
				return;
			}

			Eigen::MatrixXd local_grad;

			const int n_local_bases = int(bases.size());
			const bool is_volume = samples.cols() == 3;
			Eigen::MatrixXd dxmv = Eigen::MatrixXd::Zero(samples.rows(), samples.cols());
			Eigen::MatrixXd dymv = Eigen::MatrixXd::Zero(samples.rows(), samples.cols());
			Eigen::MatrixXd dzmv = Eigen::MatrixXd::Zero(samples.rows(), samples.cols());

			std::vector<AssemblyValues> tmp_val;
			evaluate_grads(samples, tmp_val);

			for (int j = 0; j < n_local_bases; ++j)
			{
				const Basis &b = bases[j];
				const auto &grad = tmp_val[j].grad;

				for (std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					for (long k = 0; k < samples.rows(); ++k)
					{
						dxmv.row(k) += grad(k, 0) * b.global()[ii].node * b.global()[ii].val;
						dymv.row(k) += grad(k, 1) * b.global()[ii].node * b.global()[ii].val;
						if (is_volume)
							dzmv.row(k) += grad(k, 2) * b.global()[ii].node * b.global()[ii].val;
					}
				}
			}

			Eigen::MatrixXd tmp(samples.cols(), samples.cols());

			for (long k = 0; k < samples.rows(); ++k)
			{
				tmp.row(0) = dxmv.row(k);
				tmp.row(1) = dymv.row(k);
				if (is_volume)
					tmp.row(2) = dzmv.row(k);

				grads[k] = tmp;
			}
		}

		Eigen::MatrixXd ElementBases::nodes() const
		{
			if (bases.size() == 0)
				return Eigen::MatrixXd();
			const int dim = bases[0].global()[0].node.size();
			Eigen::MatrixXd _nodes(bases.size(), dim);
			for (int i = 0; i < bases.size(); ++i)
			{
				assert(bases[i].global().size() == 1);
				_nodes.row(i) = bases[i].global()[0].node;
			}
			return _nodes;
		}
	} // namespace basis
} // namespace polyfem
