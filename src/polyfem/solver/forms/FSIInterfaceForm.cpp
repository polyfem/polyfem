#include "FSIInterfaceForm.hpp"

#include <cassert>

namespace polyfem::solver
{
	namespace
	{
		void append_block(
			const StiffnessMatrix &block,
			const int row_offset,
			const int col_offset,
			const double scale,
			std::vector<Eigen::Triplet<double>> &entries)
		{
			for (int k = 0; k < block.outerSize(); ++k)
				for (StiffnessMatrix::InnerIterator it(block, k); it; ++it)
					if (it.value() != 0)
						entries.emplace_back(
							row_offset + it.row(), col_offset + it.col(), scale * it.value());
		}
	} // namespace

	FSIInterfaceForm::FSIInterfaceForm(
		const int total_size,
		const int velocity_offset,
		const int mesh_displacement_offset,
		const int solid_displacement_offset,
		const int fluid_multiplier_offset,
		const int mesh_multiplier_offset,
		StiffnessMatrix fluid_velocity_trace,
		StiffnessMatrix fluid_solid_trace,
		StiffnessMatrix mesh_trace,
		StiffnessMatrix mesh_solid_trace,
		const time_integrator::ImplicitTimeIntegrator &fluid_integrator,
		const time_integrator::ImplicitTimeIntegrator &solid_integrator)
		: total_size_(total_size),
		  velocity_offset_(velocity_offset),
		  mesh_displacement_offset_(mesh_displacement_offset),
		  solid_displacement_offset_(solid_displacement_offset),
		  fluid_multiplier_offset_(fluid_multiplier_offset),
		  mesh_multiplier_offset_(mesh_multiplier_offset),
		  fluid_velocity_trace_(std::move(fluid_velocity_trace)),
		  fluid_solid_trace_(std::move(fluid_solid_trace)),
		  mesh_trace_(std::move(mesh_trace)),
		  mesh_solid_trace_(std::move(mesh_solid_trace)),
		  fluid_multiplier_mass_(make_multiplier_mass(fluid_velocity_trace_)),
		  mesh_multiplier_mass_(make_multiplier_mass(mesh_trace_)),
		  fluid_integrator_(fluid_integrator),
		  solid_integrator_(solid_integrator)
	{
		assert(total_size_ > 0);
		assert(fluid_velocity_trace_.rows() == fluid_solid_trace_.rows());
		assert(mesh_trace_.rows() == mesh_solid_trace_.rows());
		assert(fluid_solid_trace_.cols() == mesh_solid_trace_.cols());
		assert(fluid_multiplier_offset_ + fluid_multiplier_size() == mesh_multiplier_offset_);
		assert(mesh_multiplier_offset_ + mesh_multiplier_size() <= total_size_);
	}

	StiffnessMatrix FSIInterfaceForm::make_multiplier_mass(const StiffnessMatrix &trace)
	{
		Eigen::VectorXd row_mass = Eigen::VectorXd::Zero(trace.rows());
		for (int k = 0; k < trace.outerSize(); ++k)
			for (StiffnessMatrix::InnerIterator it(trace, k); it; ++it)
				row_mass(it.row()) += std::abs(it.value());
		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(trace.rows());
		for (int row = 0; row < trace.rows(); ++row)
			entries.emplace_back(row, row, std::max(row_mass(row), 1e-12));
		StiffnessMatrix result(trace.rows(), trace.rows());
		result.setFromTriplets(entries.begin(), entries.end());
		return result;
	}

	double FSIInterfaceForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd residual;
		first_derivative_unweighted(x, residual);
		return residual.squaredNorm();
	}

	Eigen::VectorXd FSIInterfaceForm::physical_constraint(
		const Eigen::VectorXd &velocity, const Eigen::VectorXd &solid_velocity) const
	{
		assert(velocity.size() == fluid_velocity_trace_.cols());
		assert(solid_velocity.size() == fluid_solid_trace_.cols());
		return fluid_velocity_trace_ * velocity - fluid_solid_trace_ * solid_velocity;
	}

	Eigen::VectorXd FSIInterfaceForm::mesh_constraint(
		const Eigen::VectorXd &mesh_displacement,
		const Eigen::VectorXd &solid_displacement) const
	{
		assert(mesh_displacement.size() == mesh_trace_.cols());
		assert(solid_displacement.size() == mesh_solid_trace_.cols());
		return mesh_trace_ * mesh_displacement - mesh_solid_trace_ * solid_displacement;
	}

	void FSIInterfaceForm::first_derivative_unweighted(
		const Eigen::VectorXd &x, Eigen::VectorXd &residual) const
	{
		assert(x.size() == total_size_);
		const int velocity_size = fluid_velocity_trace_.cols();
		const int mesh_size = mesh_trace_.cols();
		const int solid_size = fluid_solid_trace_.cols();
		const Eigen::VectorXd velocity = x.segment(velocity_offset_, velocity_size);
		const Eigen::VectorXd mesh_displacement = x.segment(mesh_displacement_offset_, mesh_size);
		const Eigen::VectorXd solid_displacement = x.segment(solid_displacement_offset_, solid_size);
		const Eigen::VectorXd fluid_multiplier = x.segment(fluid_multiplier_offset_, fluid_multiplier_size());
		const Eigen::VectorXd mesh_multiplier = x.segment(mesh_multiplier_offset_, mesh_multiplier_size());
		const Eigen::VectorXd solid_velocity = solid_integrator_.compute_velocity(solid_displacement);

		residual = Eigen::VectorXd::Zero(total_size_);
		residual.segment(velocity_offset_, velocity_size) +=
			fluid_integrator_.acceleration_scaling() * fluid_velocity_trace_.transpose() * fluid_multiplier;
		residual.segment(solid_displacement_offset_, solid_size) -=
			solid_integrator_.acceleration_scaling() * fluid_solid_trace_.transpose() * fluid_multiplier;
		residual.segment(fluid_multiplier_offset_, fluid_multiplier_size()) =
			physical_constraint(velocity, solid_velocity);

		residual.segment(mesh_displacement_offset_, mesh_size) += mesh_trace_.transpose() * mesh_multiplier;
		residual.segment(mesh_multiplier_offset_, mesh_multiplier_size()) =
			mesh_constraint(mesh_displacement, solid_displacement);
	}

	void FSIInterfaceForm::second_derivative_unweighted(
		const Eigen::VectorXd &, StiffnessMatrix &jacobian) const
	{
		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(
			2 * fluid_velocity_trace_.nonZeros() + 2 * fluid_solid_trace_.nonZeros()
			+ 2 * mesh_trace_.nonZeros() + mesh_solid_trace_.nonZeros());
		append_block(fluid_velocity_trace_.transpose(), velocity_offset_, fluid_multiplier_offset_,
					 fluid_integrator_.acceleration_scaling(), entries);
		append_block(fluid_solid_trace_.transpose(), solid_displacement_offset_, fluid_multiplier_offset_,
					 -solid_integrator_.acceleration_scaling(), entries);
		append_block(fluid_velocity_trace_, fluid_multiplier_offset_, velocity_offset_, 1, entries);
		append_block(fluid_solid_trace_, fluid_multiplier_offset_, solid_displacement_offset_,
					 -solid_integrator_.dv_dx(), entries);
		append_block(mesh_trace_.transpose(), mesh_displacement_offset_, mesh_multiplier_offset_, 1, entries);
		append_block(mesh_trace_, mesh_multiplier_offset_, mesh_displacement_offset_, 1, entries);
		append_block(mesh_solid_trace_, mesh_multiplier_offset_, solid_displacement_offset_, -1, entries);
		jacobian.resize(total_size_, total_size_);
		jacobian.setFromTriplets(entries.begin(), entries.end());
		jacobian.makeCompressed();
	}
} // namespace polyfem::solver
