#include <polyfem/VTUWriter.hpp>
#include <polyfem/Logger.hpp>
namespace polyfem
{
    namespace
    {
        // https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html#l00069
        static const int VTK_LINE = 3;
        static const int VTK_TETRA = 10;
        static const int VTK_TRIANGLE = 5;
        static const int VTK_QUAD = 9;
        static const int VTK_HEXAHEDRON = 12;
        static const int VTK_POLYGON = 7;

        inline static int VTKTagVolume(const int n_vertices)
        {
            switch (n_vertices)
            {
            case 3:
                return VTK_TRIANGLE;
            case 4:
                return VTK_TETRA;
            case 8:
                return VTK_HEXAHEDRON;
            default:
                //element type not supported. To add it (http://www.vtk.org/VTK/img/file-formats.pdf)
                logger().error("{} not supported", n_vertices);
                assert(false);
                return -1;
            }
        }

        inline static int VTKTagPlanar(const int n_vertices)
        {
            switch (n_vertices)
            {
            case 2:
                return VTK_LINE;
            case 3:
                return VTK_TRIANGLE;
            case 4:
                return VTK_QUAD;
            default:
                //element type not supported. To add it (http://www.vtk.org/VTK/img/file-formats.pdf)
                logger().error("{} not supported", n_vertices);
                assert(false);
                return -1;
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

            base64Layer base64(os);

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
        base64Layer base64(os);

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
        int min_tag, max_tag;
        if (!is_volume_)
        {
            min_tag = VTKTagPlanar(n_cell_vertices);
            max_tag = VTKTagPlanar(n_cell_vertices);
        }
        else
        {
            min_tag = VTKTagVolume(n_cell_vertices);
            max_tag = VTKTagVolume(n_cell_vertices);
        }

        if (binary_)
        {
            os << "<DataArray type=\"Int8\" Name=\"types\" format=\"binary\" RangeMin=\"" << min_tag << "\" RangeMax=\"" << max_tag << "\">\n";
            const uint64_t size = n_cells * sizeof(int8_t);
            base64.write(size);
        }
        else
            os << "<DataArray type=\"Int8\" Name=\"types\" format=\"ascii\" RangeMin=\"" << min_tag << "\" RangeMax=\"" << max_tag << "\">\n";

        for (int i = 0; i < n_cells; ++i)
        {
            const int8_t tag = min_tag;
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
            os << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"binary\" RangeMin=\"" << n_cell_vertices << "\" RangeMax=\"" << n_cells * n_cell_vertices << "\">\n";
            const uint64_t size = n_cells * sizeof(int64_t);
            base64.write(size);
        }
        else
            os << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\" RangeMin=\"" << n_cell_vertices << "\" RangeMax=\"" << n_cells * n_cell_vertices << "\">\n";

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

        //FIXME?
        // if (data.cols() == 2)
        // {
        //  express::BlockEigen::MatrixXd<typename Eigen::MatrixXd::EntryType> data3((data.rows() * 3) / 2, data.columns(), 3, data.columns());
        //  data3.allSet(0);
        //  for (int i = 0; i < data3.nBlockRows(); ++i) {
        //      data3.setBlockAt(i, 0, data.rowRange(i * n_components, (i + 1) * n_components));
        //  }
        //  point_data_.back().initialize(name, "Float32", data3, 3);
        // } else

        point_data_.back().initialize(name, "Float64", data, data.cols());
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
} // namespace polyfem
