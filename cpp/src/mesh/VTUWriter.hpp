#ifndef VTU_WRITER_HPP
#define VTU_WRITER_HPP

#include <Eigen/Dense>

#include <fstream>
#include <string>
#include <iostream>
#include <vector>

namespace poly_fem {
    namespace
    {
        template<typename T>
        class VTKDataNode
        {

        public:
            VTKDataNode()
            { }

            VTKDataNode(const std::string &name, const std::string &numeric_type, const Eigen::MatrixXd &data = Eigen::MatrixXd(), const int n_components = 1)
            : name_(name), numeric_type_(numeric_type), data_(data), n_components_(n_components)
            { }

            inline Eigen::MatrixXd &data() { return data_; }

            void initialize(const std::string &name, const std::string &numeric_type, const Eigen::MatrixXd &data, const int n_components = 1)
            {
                name_ = name;
                numeric_type_ = numeric_type;
                data_ = data;
                n_components_ = n_components;
            }

            void write(std::ostream &os) const
            {
                os << "<DataArray type=\"" << numeric_type_ << "\" Name=\"" << name_ << "\" NumberOfComponents=\"" <<
                n_components_ << "\" format=\"ascii\">\n";
                if (data_.cols() != 1)
                {
                    std::cout << "Warning: writing matrix in vtu file (check ordering of values)" << std::endl;
                }
                os << data_;
                os << "</DataArray>\n";
            }

            inline bool empty() const { return data_.size() <= 0; }

        private:
            std::string name_;
            ///Float32/
            std::string numeric_type_;
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data_;
            int n_components_;
        };
    }


    class VTUWriter
    {
    public:
        bool write_tet_mesh(const std::string &path, const Eigen::MatrixXd &points, const Eigen::MatrixXi &tets);

        void add_field(const std::string &name, const Eigen::MatrixXd &data);
        void add_scalar_field(const std::string &name, const Eigen::MatrixXd &data);
        void add_vector_field(const std::string &name, const Eigen::MatrixXd &data);

        void clear();
    private:
        bool is_volume_;

        std::vector<VTKDataNode<double>> point_data_;
        std::vector<VTKDataNode<double>> cell_data_;
        std::string current_scalar_point_data_;
        std::string current_vector_point_data_;


        void write_point_data(std::ostream &os);
        void write_header(const int n_vertices, const int n_elements, std::ostream &os);
        void write_footer(std::ostream &os);
        void write_points(const Eigen::MatrixXd &points, std::ostream &os);
        void write_cells(const Eigen::MatrixXi &tets, std::ostream &os);

    };
}

#endif //VTU_WRITER_HPP
