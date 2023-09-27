
#include "InversionBarrierForm.hpp"

#include <polyfem/utils/GeometryUtils.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <ipc/barrier/barrier.hpp>
#include <ipc/utils/local_to_global.hpp>
#include <ipc/utils/eigen_ext.hpp>

namespace polyfem::solver
{
	InversionBarrierForm::InversionBarrierForm(
		const Eigen::MatrixXd &rest_positions, const Eigen::MatrixXi &elements, const int dim, const double vhat)
		: rest_positions_(rest_positions), elements_(elements), dim_(dim), vhat_(vhat)
	{
	}

	double InversionBarrierForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd V = rest_positions_ + utils::unflatten(x, dim_);

		auto storage = utils::create_thread_storage<double>(0.0);

		const double scale = 1.0 / (vhat_ * vhat_);

		utils::maybe_parallel_for(elements_.rows(), [&](int start, int end, int thread_id) {
			double &local_potential = utils::get_local_thread_storage(storage, thread_id);
			for (int i = start; i < end; i++)
			{
				local_potential +=
					scale * element_volume(rest_positions_(elements_.row(i), Eigen::all))
					* ipc::barrier(element_volume(V(elements_.row(i), Eigen::all)), vhat_);
			}
		});

		double potential = 0;
		for (const auto &local_potential : storage)
			potential += local_potential;
		return potential;
	}

	void InversionBarrierForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::MatrixXd V = rest_positions_ + utils::unflatten(x, dim_);

		auto storage = utils::create_thread_storage<Eigen::VectorXd>(Eigen::VectorXd::Zero(x.size()));

		const double scale = 1.0 / (vhat_ * vhat_);

		utils::maybe_parallel_for(elements_.rows(), [&](int start, int end, int thread_id) {
			Eigen::VectorXd &grad = utils::get_local_thread_storage(storage, thread_id);
			for (int i = start; i < end; i++)
			{
				const Eigen::MatrixXd element_vertices = V(elements_.row(i), Eigen::all);

				Eigen::VectorXd local_grad =
					(scale * element_volume(rest_positions_(elements_.row(i), Eigen::all))
					 * ipc::barrier_gradient(element_volume(element_vertices), vhat_))
					* element_volume_gradient(element_vertices);

				ipc::local_gradient_to_global_gradient(local_grad, elements_.row(i), dim_, grad);
			}
		});

		gradv = Eigen::VectorXd::Zero(x.size());
		for (const auto &local_grad : storage)
			gradv += local_grad;
	}

	void InversionBarrierForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		const Eigen::MatrixXd V = rest_positions_ + utils::unflatten(x, dim_);

		auto storage = utils::create_thread_storage(std::vector<Eigen::Triplet<double>>());

		const double scale = 1.0 / (vhat_ * vhat_);

		utils::maybe_parallel_for(elements_.rows(), [&](int start, int end, int thread_id) {
			std::vector<Eigen::Triplet<double>> &hess_triplets =
				utils::get_local_thread_storage(storage, thread_id);

			for (int i = start; i < end; i++)
			{
				const Eigen::MatrixXd element_vertices = V(elements_.row(i), Eigen::all);

				const double volume = element_volume(element_vertices);
				const Eigen::VectorXd volume_grad = element_volume_gradient(element_vertices);

				const double rest_volume = element_volume(rest_positions_(elements_.row(i), Eigen::all));

				Eigen::MatrixXd local_hess =
					(scale * rest_volume * ipc::barrier_hessian(volume, vhat_)) * volume_grad * volume_grad.transpose()
					+ (scale * rest_volume * ipc::barrier_gradient(volume, vhat_)) * element_volume_hessian(element_vertices);

				if (project_to_psd_)
					local_hess = ipc::project_to_psd(local_hess);

				ipc::local_hessian_to_global_triplets(local_hess, elements_.row(i), dim_, hess_triplets);
			}
		});

		hessian.resize(x.size(), x.size());
		for (const auto &local_hess_triplets : storage)
		{
			Eigen::SparseMatrix<double> local_hess(x.size(), x.size());
			local_hess.setFromTriplets(
				local_hess_triplets.begin(), local_hess_triplets.end());
			hessian += local_hess;
		}
	}

	double InversionBarrierForm::element_volume(const Eigen::MatrixXd &element_vertices)
	{
		if (element_vertices.rows() == 3)
			return utils::triangle_area(element_vertices);

		assert(element_vertices.rows() == 4 && element_vertices.cols() == 3);
		return utils::tetrahedron_volume(element_vertices);
	}

	Eigen::VectorXd InversionBarrierForm::element_volume_gradient(const Eigen::MatrixXd &element_vertices)
	{
		Eigen::VectorXd grad(element_vertices.size());
		if (element_vertices.rows() == 3)
		{
			assert(element_vertices.cols() == 2);
			utils::triangle_area_2D_gradient(
				element_vertices(0, 0), element_vertices(0, 1),
				element_vertices(1, 0), element_vertices(1, 1),
				element_vertices(2, 0), element_vertices(2, 1),
				grad.data());
		}
		else
		{
			assert(element_vertices.rows() == 4 && element_vertices.cols() == 3);
			utils::tetrahedron_volume_gradient(
				element_vertices(0, 0), element_vertices(0, 1), element_vertices(0, 2),
				element_vertices(1, 0), element_vertices(1, 1), element_vertices(1, 2),
				element_vertices(2, 0), element_vertices(2, 1), element_vertices(2, 2),
				element_vertices(3, 0), element_vertices(3, 1), element_vertices(3, 2),
				grad.data());
		}
		return grad;
	}

	Eigen::MatrixXd InversionBarrierForm::element_volume_hessian(const Eigen::MatrixXd &element_vertices)
	{
		Eigen::MatrixXd hess(element_vertices.size(), element_vertices.size());
		if (element_vertices.rows() == 3)
		{
			assert(element_vertices.cols() == 2);
			utils::triangle_area_2D_hessian(
				element_vertices(0, 0), element_vertices(0, 1),
				element_vertices(1, 0), element_vertices(1, 1),
				element_vertices(2, 0), element_vertices(2, 1),
				hess.data());
		}
		else
		{
			assert(element_vertices.rows() == 4 && element_vertices.cols() == 3);
			utils::tetrahedron_volume_hessian(
				element_vertices(0, 0), element_vertices(0, 1), element_vertices(0, 2),
				element_vertices(1, 0), element_vertices(1, 1), element_vertices(1, 2),
				element_vertices(2, 0), element_vertices(2, 1), element_vertices(2, 2),
				element_vertices(3, 0), element_vertices(3, 1), element_vertices(3, 2),
				hess.data());
		}
		return hess;
	}

	bool InversionBarrierForm::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		const Eigen::MatrixXd V = rest_positions_ + utils::unflatten(x1, dim_);

		for (size_t i = 0; i < elements_.rows(); ++i)
		{
			// TODO: use exact predicate for this
			if (element_volume(V(elements_.row(i), Eigen::all)) <= 0)
			{
				return false;
			}
		}

		return true;
	}
} // namespace polyfem::solver
