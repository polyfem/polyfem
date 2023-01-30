#include "VTUWriter.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem
{
	namespace io
	{
		namespace
		{
			// https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html#l00069
			static const int VTK_VERTEX = 1;
			static const int VTK_LINE = 3;
			static const int VTK_TETRA = 10;
			static const int VTK_TRIANGLE = 5;
			static const int VTK_QUAD = 9;
			static const int VTK_HEXAHEDRON = 12;
			static const int VTK_POLYGON = 7;
			static const int VTK_POLYHEDRON = 42;

			static const int VTK_LAGRANGE_TRIANGLE = 69;
			static const int VTK_LAGRANGE_QUADRILATERAL = 70;

			static const int VTK_LAGRANGE_TETRAHEDRON = 71;
			static const int VTK_LAGRANGE_HEXAHEDRON = 72;

			inline static int VTKTagVolume(const int n_vertices, bool is_simplex, bool is_poly)
			{
				switch (n_vertices)
				{
				case 1:
					return VTK_VERTEX;
				case 2:
					return VTK_LINE;
				case 3:
					return VTK_TRIANGLE;
				case 4:
					return VTK_TETRA;
				case 8:
					return VTK_HEXAHEDRON;
				default:
					if (is_poly)
						return VTK_POLYHEDRON;
					if (is_simplex)
						return VTK_LAGRANGE_TETRAHEDRON;
					else
						return VTK_LAGRANGE_HEXAHEDRON;
				}
			}

			inline static int VTKTagPlanar(const int n_vertices, bool is_simplex, bool is_poly)
			{
				switch (n_vertices)
				{
				case 1:
					return VTK_VERTEX;
				case 2:
					return VTK_LINE;
				case 3:
					return VTK_TRIANGLE;
				case 4:
					return VTK_QUAD;
				default:
					if (is_poly)
						return VTK_POLYGON;

					if (is_simplex)
						return VTK_LAGRANGE_TRIANGLE;
					else
						return VTK_LAGRANGE_QUADRILATERAL;
				}
			}
		} // namespace

		VTUWriter::VTUWriter(bool binary)
			: binary_(binary)
		{
		}
		void VTUWriter::write_point_data(std::ostream &os)
		{
			if (current_scalar_point_data_.empty() && current_vector_point_data_.empty())
				return;

			os << "<PointData ";
			if (!current_scalar_point_data_.empty())
				os << "Scalars=\"" << current_scalar_point_data_ << "\" ";
			if (!current_vector_point_data_.empty())
				os << "Vectors=\"" << current_vector_point_data_ << "\" ";
			os << ">\n";

			for (auto it = point_data_.begin(); it != point_data_.end(); ++it)
			{
				it->write(os);
			}

			os << "</PointData>\n";
		}

		void VTUWriter::write_header(const int n_vertices, const int n_elements, std::ostream &os)
		{
			os << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" header_type=\"UInt64\">\n";
			os << "<UnstructuredGrid>\n";
			os << "<Piece NumberOfPoints=\"" << n_vertices << "\" NumberOfCells=\"" << n_elements << "\">\n";
		}

		void VTUWriter::write_footer(std::ostream &os)
		{
			os << "</Piece>\n";
			os << "</UnstructuredGrid>\n";
			os << "</VTKFile>\n";
		}

		void VTUWriter::write_points(const Eigen::MatrixXd &points, std::ostream &os)
		{
			os << "<Points>\n";
			if (binary_)
			{

				Eigen::MatrixXd tmp = points.transpose();
				if (tmp.rows() != 3)
				{
					tmp.conservativeResize(3, tmp.cols());
					tmp.row(2).setZero();
				}

				utils::base64Layer base64(os);

				os << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"binary\">\n";
				const uint64_t size = tmp.size() * sizeof(double);
				base64.write(size);

				base64.write(tmp.data(), tmp.size());
				base64.close();
				os << "\n";
			}
			else
			{
				os << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";

				for (int d = 0; d < points.rows(); ++d)
				{

					for (int i = 0; i < points.cols(); ++i)
					{
						os << points(d, i);
						if (i < points.cols() - 1)
						{
							os << " ";
						}
					}

					if (!is_volume_)
						os << " 0";

					os << "\n";
				}
			}

			os << "</DataArray>\n";
			os << "</Points>\n";
		}

		void VTUWriter::write_cells(const Eigen::MatrixXi &cells, std::ostream &os)
		{
			const int n_cells = cells.rows();
			const int n_cell_vertices = cells.cols();
			os << "<Cells>\n";
			utils::base64Layer base64(os);

			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			if (binary_)
			{
				os << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"binary\" >\n";
				const uint64_t size = cells.size() * sizeof(int64_t);
				base64.write(size);

				Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> tmp = cells.transpose().template cast<int64_t>();
				base64.write(tmp.data(), tmp.size());
				base64.close();
				os << "\n";
			}
			else
			{
				os << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";

				for (int c = 0; c < n_cells; ++c)
				{
					for (int i = 0; i < n_cell_vertices; ++i)
					{
						const int64_t v_index = cells(c, i);

						os << v_index;
						if (i < n_cell_vertices - 1)
						{
							os << " ";
						}
					}
				}

				os << "\n";
			}

			os << "</DataArray>\n";
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			const int int_tag = is_volume_ ? VTKTagVolume(n_cell_vertices, true, false) : VTKTagPlanar(n_cell_vertices, true, false);
			if (binary_)
			{
				os << "<DataArray type=\"Int8\" Name=\"types\" format=\"binary\">\n";
				const uint64_t size = n_cells * sizeof(int8_t);
				base64.write(size);
			}
			else
				os << "<DataArray type=\"Int8\" Name=\"types\" format=\"ascii\">\n";

			for (int i = 0; i < n_cells; ++i)
			{
				const int8_t tag = int_tag;
				if (binary_)
					base64.write(tag);
				else
					os << tag << "\n";
			}
			if (binary_)
			{
				base64.close();
				os << "\n";
			}
			os << "</DataArray>\n";

			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			if (binary_)
			{
				os << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"binary\">\n";
				const uint64_t size = n_cells * sizeof(int64_t);
				base64.write(size);
			}
			else
				os << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";

			int64_t acc = n_cell_vertices;
			for (int i = 0; i < n_cells; ++i)
			{
				if (binary_)
					base64.write(acc);
				else
					os << acc << "\n";
				acc += n_cell_vertices;
			}
			if (binary_)
			{
				base64.close();
				os << "\n";
			}

			os << "</DataArray>\n";
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			os << "</Cells>\n";
		}

		void VTUWriter::write_cells(const std::vector<std::vector<int>> &cells, const bool is_simplex, const bool is_poly, std::ostream &os)
		{
			const int n_cells = cells.size();
			os << "<Cells>\n";
			utils::base64Layer base64(os);

			int n_cells_indices = 0;
			for (const auto &c : cells)
			{
				n_cells_indices += c.size();
			}

			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			if (binary_)
			{
				os << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"binary\" >\n";
				const uint64_t size = n_cells_indices * sizeof(int64_t);
				base64.write(size);

				Eigen::Matrix<int64_t, Eigen::Dynamic, 1> tmp(n_cells_indices);
				int index = 0;
				for (const auto &c : cells)
				{
					for (const int i : c)
						tmp(index++) = i;
				}
				base64.write(tmp.data(), tmp.size());
				base64.close();
				os << "\n";
			}
			else
			{
				os << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";

				for (const auto &c : cells)
				{
					for (int i = 0; i < c.size(); ++i)
					{
						const int64_t v_index = c[i];

						os << v_index;
						if (i < c.size() - 1)
						{
							os << " ";
						}
					}
					os << "\n";
				}
			}

			os << "</DataArray>\n";
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			if (binary_)
			{
				os << "<DataArray type=\"UInt8\" Name=\"types\" format=\"binary\">\n";
				const uint64_t size = n_cells * sizeof(uint8_t);
				base64.write(size);
			}
			else
				os << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";

			for (int i = 0; i < n_cells; ++i)
			{
				const int int_tag = is_volume_ ? VTKTagVolume(cells[i].size(), is_simplex, is_poly) : VTKTagPlanar(cells[i].size(), is_simplex, is_poly);
				const uint8_t tag = int_tag;

				if (binary_)
					base64.write(tag);
				else
					os << int_tag << "\n";
			}
			if (binary_)
			{
				base64.close();
				os << "\n";
			}
			os << "</DataArray>\n";

			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			if (binary_)
			{
				os << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"binary\">\n";
				const uint64_t size = n_cells * sizeof(int64_t);
				base64.write(size);
			}
			else
				os << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";

			int64_t acc = 0;
			for (int i = 0; i < n_cells; ++i)
			{
				acc += cells[i].size();
				if (binary_)
					base64.write(acc);
				else
					os << acc << "\n";
			}
			if (binary_)
			{
				base64.close();
				os << "\n";
			}

			os << "</DataArray>\n";
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			os << "</Cells>\n";
		}

		void VTUWriter::clear()
		{
			point_data_.clear();
			cell_data_.clear();
		}

		void VTUWriter::add_field(const std::string &name, const Eigen::MatrixXd &data)
		{
			using std::abs;

			Eigen::MatrixXd tmp;
			tmp.resizeLike(data);

			for (long i = 0; i < data.size(); ++i)
				tmp(i) = abs(data(i)) < 1e-16 ? 0 : data(i);

			if (tmp.cols() == 1)
				add_scalar_field(name, tmp);
			else
				add_vector_field(name, tmp);
		}

		void VTUWriter::add_scalar_field(const std::string &name, const Eigen::MatrixXd &data)
		{
			point_data_.push_back(VTKDataNode<double>(binary_));
			point_data_.back().initialize(name, "Float64", data);
			current_scalar_point_data_ = name;
		}

		void VTUWriter::add_vector_field(const std::string &name, const Eigen::MatrixXd &data)
		{
			point_data_.push_back(VTKDataNode<double>(binary_));

			Eigen::MatrixXd tmp = data;

			if (data.cols() == 2)
			{
				tmp.conservativeResize(tmp.rows(), 3);
				tmp.col(2).setZero();
			}

			point_data_.back().initialize(name, "Float64", tmp, tmp.cols());
			current_vector_point_data_ = name;
		}

		bool VTUWriter::write_mesh(const std::string &path, const Eigen::MatrixXd &points, const Eigen::MatrixXi &cells)
		{
			std::ofstream os;
			os.open(path.c_str());
			if (!os.good())
			{
				os.close();
				return false;
			}

			is_volume_ = points.cols() == 3;

			write_header(points.rows(), cells.rows(), os);
			write_points(points, os);
			write_point_data(os);
			write_cells(cells, os);

			write_footer(os);
			os.close();
			clear();
			return true;
		}

		bool VTUWriter::write_mesh(const std::string &path, const Eigen::MatrixXd &points, const std::vector<std::vector<int>> &cells, const bool is_simplicial, const bool has_poly)
		{
			std::ofstream os;
			os.open(path.c_str());
			if (!os.good())
			{
				os.close();
				return false;
			}

			is_volume_ = points.cols() == 3;

			write_header(points.rows(), cells.size(), os);
			write_points(points, os);
			write_point_data(os);
			write_cells(cells, is_simplicial, has_poly, os);

			write_footer(os);
			os.close();
			clear();
			return true;
		}
	} // namespace io
} // namespace polyfem
