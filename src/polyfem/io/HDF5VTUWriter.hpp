#pragma once

#include <polyfem/utils/Logger.hpp>

#include <highfive/H5Easy.hpp>
#include <highfive/H5File.hpp>

#include <Eigen/Dense>

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <array>

namespace polyfem
{
	namespace io
	{
		template <typename T>
		class HDF5VTKDataNode
		{

		public:
			HDF5VTKDataNode()
			{
			}

			HDF5VTKDataNode(const std::string &name, const Eigen::MatrixXd &data = Eigen::MatrixXd())
				: name_(name), data_(data)
			{
			}

			// const inline Eigen::MatrixXd &data() { return data_; }

			void initialize(const std::string &name, const Eigen::MatrixXd &data)
			{
				name_ = name;
				data_ = data;
			}

			void write(HighFive::File &file) const
			{
				assert(data_.cols() == 1 || data_.cols() == 3);
				if (data_.cols() == 3)
					H5Easy::dump(file, "/VTKHDF/PointData/" + name_, data_);
				else
				{
					Eigen::Matrix<T, Eigen::Dynamic, 1> tmp = data_.col(0);
					H5Easy::dump(file, "/VTKHDF/PointData/" + name_, tmp);
				}
			}

			inline bool empty() const { return data_.size() <= 0; }

		private:
			std::string name_;
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data_;
			int n_components_;
		};

		class HDF5VTUWriter
		{
		public:
			HDF5VTUWriter(bool binary = true);

			bool write_mesh(const std::string &path, const Eigen::MatrixXd &points, const Eigen::MatrixXi &cells);
			bool write_mesh(const std::string &path, const Eigen::MatrixXd &points, const std::vector<std::vector<int>> &cells, const bool is_simplicial, const bool has_poly);

			void add_field(const std::string &name, const Eigen::MatrixXd &data);
			void add_scalar_field(const std::string &name, const Eigen::MatrixXd &data);
			void add_vector_field(const std::string &name, const Eigen::MatrixXd &data);

			void clear();

		private:
			bool is_volume_;

			std::vector<HDF5VTKDataNode<double>> point_data_;
			std::vector<HDF5VTKDataNode<double>> cell_data_;
			std::string current_scalar_point_data_;
			std::string current_vector_point_data_;

			void write_point_data(HighFive::File &file);
			void write_header(const int n_vertices, const int n_elements, HighFive::Group &grp, HighFive::File &file);
			void write_points(const Eigen::MatrixXd &points, HighFive::File &file);
			void write_cells(const Eigen::MatrixXi &cells, HighFive::Group &grp, HighFive::File &file);
			void write_cells(const std::vector<std::vector<int>> &cells, const bool is_simplex, const bool is_poly, HighFive::Group &grp, HighFive::File &file);
		};
	} // namespace io
} // namespace polyfem
