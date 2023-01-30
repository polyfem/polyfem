#pragma once

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/base64Layer.hpp>

#include <Eigen/Dense>

#include <fstream>
#include <string>
#include <iostream>
#include <vector>

namespace polyfem
{
	namespace io
	{
		template <typename T>
		class VTKDataNode
		{

		public:
			VTKDataNode(bool binary)
				: binary_(binary)
			{
			}

			VTKDataNode(const std::string &name, const double binary, const std::string &numeric_type, const Eigen::MatrixXd &data = Eigen::MatrixXd(), const int n_components = 1)
				: name_(name), binary_(binary), numeric_type_(numeric_type), data_(binary_ ? data.transpose() : data), n_components_(n_components)
			{
			}

			// const inline Eigen::MatrixXd &data() { return data_; }

			void initialize(const std::string &name, const std::string &numeric_type, const Eigen::MatrixXd &data, const int n_components = 1)
			{
				name_ = name;
				numeric_type_ = numeric_type;
				data_ = binary_ ? data.transpose() : data;
				n_components_ = n_components;
			}

			void write(std::ostream &os) const
			{

				if (binary_)
				{
					utils::base64Layer base64(os);

					os << "<DataArray type=\"" << numeric_type_ << "\" Name=\"" << name_ << "\" NumberOfComponents=\"" << n_components_ << "\" format=\"binary\">\n";
					const uint64_t size = data_.size() * sizeof(T);
					base64.write(size);

					base64.write(data_.data(), data_.size());
					base64.close();
					os << "\n";
				}
				else
				{
					os << "<DataArray type=\"" << numeric_type_ << "\" Name=\"" << name_ << "\" NumberOfComponents=\"" << n_components_ << "\" format=\"ascii\">\n";
					os << data_;
				}
				os << "</DataArray>\n";
			}

			inline bool empty() const { return data_.size() <= 0; }

		private:
			std::string name_;
			bool binary_;
			/// Float32/
			std::string numeric_type_;
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data_;
			int n_components_;
		};

		class VTUWriter
		{
		public:
			VTUWriter(bool binary = true);

			bool write_mesh(const std::string &path, const Eigen::MatrixXd &points, const Eigen::MatrixXi &cells);
			bool write_mesh(const std::string &path, const Eigen::MatrixXd &points, const std::vector<std::vector<int>> &cells, const bool is_simplicial, const bool has_poly);

			void add_field(const std::string &name, const Eigen::MatrixXd &data);
			void add_scalar_field(const std::string &name, const Eigen::MatrixXd &data);
			void add_vector_field(const std::string &name, const Eigen::MatrixXd &data);

			void clear();

		private:
			bool is_volume_;
			bool binary_;

			std::vector<VTKDataNode<double>> point_data_;
			std::vector<VTKDataNode<double>> cell_data_;
			std::string current_scalar_point_data_;
			std::string current_vector_point_data_;

			void write_point_data(std::ostream &os);
			void write_header(const int n_vertices, const int n_elements, std::ostream &os);
			void write_footer(std::ostream &os);
			void write_points(const Eigen::MatrixXd &points, std::ostream &os);
			void write_cells(const Eigen::MatrixXi &cells, std::ostream &os);
			void write_cells(const std::vector<std::vector<int>> &cells, const bool is_simplex, const bool is_poly, std::ostream &os);
		};
	} // namespace io
} // namespace polyfem
