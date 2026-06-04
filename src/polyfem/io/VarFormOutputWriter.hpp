#pragma once

#include <polyfem/io/OutData.hpp>

#include <Eigen/Dense>

#include <string>

namespace polyfem::varform
{
	class VarForm;
}

namespace polyfem::io
{
	class VarFormOutputWriter
	{
	public:
		explicit VarFormOutputWriter(const varform::VarForm &var_form);

		void export_data(const Eigen::MatrixXd &sol);
		void save_timestep(const double time, const int t, const double t0, const double dt, const Eigen::MatrixXd &sol);
		void save_subsolve(const int i, const int t, const Eigen::MatrixXd &sol);

	private:
		std::string resolve_output_path(const std::string &path) const;
		void ensure_sampler();
		OutGeometryData::ExportOptions export_options(const OutputSpace &space) const;
		OutputFieldFunction output_field_function(const Eigen::MatrixXd &sol, const OutGeometryData::ExportOptions &opts) const;

		const varform::VarForm &var_form_;
		OutGeometryData out_geom_;
		bool sampler_initialized_ = false;
	};
} // namespace polyfem::io
