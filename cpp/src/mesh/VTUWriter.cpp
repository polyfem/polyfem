#include "VTUWriter.hpp"

namespace poly_fem
{
    namespace
    {
        static const int VTK_TETRA = 10;
        static const int VTK_TRIANGLE = 5;
        static const int VTK_QUAD = 9;
        static const int VTK_HEXAHEDRON = 12;
        static const int VTK_POLYGON = 7;

        inline static int VTKTagVolume(const int n_vertices)
        {
            switch (n_vertices) {
                case 4:
                return VTK_TETRA;
                case 8:
                return VTK_HEXAHEDRON;
                default:
                //element type not supported. To add it (http://www.vtk.org/VTK/img/file-formats.pdf)
                std::cerr << "[Error] " << n_vertices << " not supported" << std::endl;
                assert(false);
                return -1;
            }
        }

        inline static int VTKTagPlanar(const int n_vertices)
        {
            switch (n_vertices) {
                case 3:
                return VTK_TRIANGLE;
                case 4:
                return VTK_QUAD;
                default:
                //element type not supported. To add it (http://www.vtk.org/VTK/img/file-formats.pdf)
                std::cerr << "[Error] " << n_vertices << " not supported" << std::endl;
                assert(false);
                return -1;
            }
        }
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

        for (auto it = point_data_.begin(); it != point_data_.end(); ++it) {
            it->write(os);
        }

        os << "</PointData>\n";
    }

    void VTUWriter::write_header(const int n_vertices, const int n_elements, std::ostream &os)
    {
        os << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\">\n";
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

            if(!is_volume_)
                os << " 0";

            os << "\n";
        }

        os << "</DataArray>\n";
        os << "</Points>\n";
    }

    void VTUWriter::write_cells(const Eigen::MatrixXi &tets, std::ostream &os)
    {
        const int n_cells = tets.rows();
        os << "<Cells>\n";
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        os << "<DataArray type=\"UInt64\" Name=\"connectivity\" format=\"ascii\">\n";

        const int n_vertices = tets.cols();

        for(int c = 0; c < n_cells; ++c)
        {
            for (int i = 0; i < n_vertices; ++i)
            {
                const int v_index = tets(c,i);
                os << v_index;
                if (i < n_vertices - 1) {
                    os << " ";
                }
            }
            os << "\n";
        }

        os << "</DataArray>\n";
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        int min_tag, max_tag;
        if (!is_volume_) {
            min_tag = VTKTagPlanar(n_vertices);
            max_tag = VTKTagPlanar(n_vertices);
        } else
        {
            min_tag = VTKTagVolume(n_vertices);
            max_tag = VTKTagVolume(n_vertices);
        }

        os << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\" RangeMin=\"" << min_tag << "\" RangeMax=\"" << max_tag << "\">\n";
        for (int i = 0; i < n_cells; ++i)
        {
            if (is_volume_)
                os << VTKTagVolume(n_vertices) << "\n";
            else
                os << VTKTagPlanar(n_vertices) << "\n";
        }
        os << "</DataArray>\n";

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        os << "<DataArray type=\"UInt64\" Name=\"offsets\" format=\"ascii\" RangeMin=\"" << n_vertices << "\" RangeMax=\"" << n_cells *n_vertices << "\">\n";

        int acc = n_vertices;
        for (int i = 0; i < n_cells; ++i) {
            os << acc << "\n";
            acc += n_vertices;
        }

        os << "</DataArray>\n";
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        os << "</Cells>\n";
    }

    void VTUWriter::clear()
    {
        point_data_.clear();
        cell_data_.clear();;
    }

    void VTUWriter::add_field(const std::string &name, const Eigen::MatrixXd &data)
    {
        if(data.cols() == 1)
            add_scalar_field(name, data);
        else
            add_vector_field(name, data);
    }

    void VTUWriter::add_scalar_field(const std::string &name, const Eigen::MatrixXd &data)
    {
        point_data_.push_back(VTKDataNode<double>());
        point_data_.back().initialize(name, "Float32", data);
        current_scalar_point_data_ = name;
    }

    void VTUWriter::add_vector_field(const std::string &name, const Eigen::MatrixXd &data)
    {
        point_data_.push_back(VTKDataNode<double>());

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

        point_data_.back().initialize(name, "Float32", data, data.cols());
        current_vector_point_data_ = name;
    }

    bool VTUWriter::write_tet_mesh(const std::string &path, const Eigen::MatrixXd &points, const Eigen::MatrixXi &tets)
    {
        std::ofstream os;
        os.open(path.c_str());
        if (!os.good()) {
            os.close();
            return false;
        }

        is_volume_ = points.cols() == 3;

        write_header(points.rows(), tets.rows(), os);
        write_points(points, os);
        write_point_data(os);
        write_cells(tets, os);

        write_footer(os);
        os.close();
        clear();
        return true;
    }
}
