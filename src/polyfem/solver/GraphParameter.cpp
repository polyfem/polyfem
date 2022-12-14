#include "GraphParameter.hpp"
#include <polyfem/io/MshReader.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <sstream>

namespace polyfem
{
    namespace {
        template <typename T>
        std::string to_string_with_precision(const T a_value, const int n = 6)
        {
            std::ostringstream out;
            out.precision(n);
            out << std::fixed << a_value;
            return out.str();
        }

        RowVectorNd get_barycenter(const mesh::Mesh &mesh, int e)
        {
            RowVectorNd barycenter;
            if (!mesh.is_volume())
            {
                const auto &mesh2d = dynamic_cast<const mesh::Mesh2D &>(mesh);
                barycenter = mesh2d.face_barycenter(e);
            }
            else
            {
                const auto &mesh3d = dynamic_cast<const mesh::Mesh3D &>(mesh);
                barycenter = mesh3d.cell_barycenter(e);
            }
            return barycenter;
        }
    }

    GraphParameter::GraphParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args) : Parameter(states_ptr, args)
    {
        parameter_name_ = "shape";
        
        max_change_ = args["max_change"];
        isosurface_inflator_prefix_ = args["isosurface_inflator_prefix"];
        out_velocity_path_ = "micro-tmp-velocity.msh";
        out_msh_path_ = "micro-tmp.msh";

        initial_guess_ = args["initial"];
        full_dim_ = get_state().n_geom_bases * get_state().mesh->dimension();
        optimization_dim_ = initial_guess_.size();

        bounds_.resize(optimization_dim_, 2);
        int i = 0;
        if (args["lower_bound"].size() > 0)
            for (double l : args["lower_bound"])
                bounds_(i++, 0) = l;
        else
            bounds_.col(0).setZero();
        i = 0;
        if (args["upper_bound"].size() > 0)
            for (double u : args["upper_bound"])
                bounds_(i++, 1) = u;
        else
            bounds_.col(1).setConstant(std::numeric_limits<double>::max());

        // periodic pattern
        unit_size_ = args["unit_size"];
        compute_pattern_period();
        
        pre_solve(initial_guess_);
    }

    void GraphParameter::compute_pattern_period()
    {
        const mesh::Mesh &mesh = *(get_state().mesh);
        int elem_period_ = 0;
        full_to_periodic_.clear();

        if (unit_size_ == 0)
        {
            full_to_periodic_.reserve(get_state().n_geom_bases);
            for (int i = 0; i < get_state().n_geom_bases; i++)
                full_to_periodic_.push_back(i);
            
            return;
        }
        
        RowVectorNd min, max;
        mesh.bounding_box(min, max);

        Eigen::VectorXi nums;
        nums.setZero(mesh.dimension());
        for (int d = 0; d < mesh.dimension(); d++)
        {
            int tmp = std::lround((max(d) - min(d)) / unit_size_);
            if (abs(tmp * unit_size_ + min(d) - max(d)) > 1e-8)
                log_and_throw_error("Mesh size is not periodic!");
            nums(d) = tmp;
        }

        for (int e = 0; e < mesh.n_elements(); e++)
        {
            if ((get_barycenter(mesh, e) - min).maxCoeff() >= unit_size_)
            {
                elem_period_ = e;
                break;
            }
        }
        if (elem_period_ == 0)
            elem_period_ = mesh.n_elements();

        // node correspondence
        {
            full_to_periodic_.assign(get_state().n_geom_bases, -1);

            utils::maybe_parallel_for(mesh.n_elements(), [&](int start, int end, int thread_id) {
                for (int e = start; e < end; e++)
                {
                    RowVectorNd offset = get_barycenter(mesh, e) - get_barycenter(mesh, e % elem_period_);
                    
                    assert(!mesh.is_volume());
                    for (int lv = 0; lv < mesh.n_face_vertices(e); lv++) // only 2D
                    {
                        int vid1 = mesh.face_vertex(e, lv);
                        auto p1 = mesh.point(vid1);
                        bool flag = false;
                        
                        if (e < elem_period_)
                        {
                            flag = true;
                            full_to_periodic_[vid1] = vid1;
                        }
                        else
                        {
                            double min_diff = 1e5;
                            int min_id = -1;
                            for (int lv2 = 0; lv2 < mesh.n_face_vertices(e % elem_period_); lv2++)
                            {
                                int vid2 = mesh.face_vertex(e % elem_period_, lv2);
                                auto p2 = mesh.point(vid2);

                                if ((p1 - offset - p2).norm() < min_diff)
                                {
                                    min_diff = (p1 - offset - p2).norm();
                                    min_id = vid2;
                                }
                            }
                            if (min_diff > 1e-5)
                                log_and_throw_error("Failed to find periodic node in periodic pattern, error = {}!", min_diff);
                            full_to_periodic_[vid1] = min_id;
                        }
                    }
                }
            });
        }
        
        logger().info("Number of elements in one period: {}, number of periods: {}", elem_period_, nums.prod());
    }

    Eigen::MatrixXd GraphParameter::map(const Eigen::VectorXd &x) const
    {
        assert(false);
        return x;
    }

    Eigen::VectorXd GraphParameter::map_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &full_grad) const
    {
        return shape_velocity_.transpose() * full_grad;
    }
    
    Eigen::VectorXd GraphParameter::get_lower_bound(const Eigen::VectorXd &x) const 
    {
        Eigen::VectorXd min = bounds_.col(0);
        for (int i = 0; i < min.size(); i++)
        {
            if (x(i) - min(i) > max_change_)
                min(i) = x(i) - max_change_;
        }
        return min;
    }

    Eigen::VectorXd GraphParameter::get_upper_bound(const Eigen::VectorXd &x) const 
    {
        Eigen::VectorXd max = bounds_.col(1);
        for (int i = 0; i < max.size(); i++)
        {
            if (max(i) - x(i) > max_change_)
                max(i) = x(i) + max_change_;
        }
        return max;
    }

    bool GraphParameter::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
    {
        if ((x1 - x0).cwiseAbs().maxCoeff() > max_change_)
            return false;
        if ((x1 - bounds_.col(0)).minCoeff() < 0)
            return false;
        if ((bounds_.col(1) - x1).minCoeff() < 0)
            return false;

        return generate_graph_mesh(x1);
    }

    bool GraphParameter::generate_graph_mesh(const Eigen::VectorXd &x)
    {
        std::string shape_params = "--params \"";
        for (int i = 0; i < x.size(); i++)
            shape_params += to_string_with_precision(x(i), 16) + " ";
        shape_params += "\" ";

        std::string command = isosurface_inflator_prefix_ + " " + shape_params + " -S " + out_velocity_path_ + " " + out_msh_path_;

        int return_val;
        try 
        {
            return_val = system(command.c_str());
        }
        catch (const std::exception &err)
        {
            logger().error("remesh command \"{}\" returns {}", command, return_val);

            return false;
        }

        logger().info("remesh command \"{}\" returns {}", command, return_val);

        if (unit_size_ > 0)
        {
            command = "python ../tile.py " + out_msh_path_;
            try 
            {
                return_val = system(command.c_str());
            }
            catch (const std::exception &err)
            {
                logger().error("tile command \"{}\" returns {}", command, return_val);

                return false;
            }

            logger().info("tile command \"{}\" returns {}", command, return_val);
        }

        return true;
    }

    bool GraphParameter::pre_solve(const Eigen::VectorXd &newX)
    {
        if (!generate_graph_mesh(newX))
            return false;

        const int cur_log = states_ptr_[0]->current_log_level;
        states_ptr_[0]->set_log_level(spdlog::level::level_enum::err); // log level is global, only need to change in one state
        // reload mesh and recompute basis
        for (auto state : states_ptr_)
        {
            assert(state->in_args["geometry"].size() == 1);
            state->in_args["geometry"][0]["mesh"] = out_msh_path_;

			state->mesh.reset();
			state->mesh = nullptr;
			state->assembler.update_lame_params(Eigen::MatrixXd(), Eigen::MatrixXd());

			json in_args = state->in_args;
			std::cout << in_args << std::endl;
			state->init(in_args, false);

			state->load_mesh();
			state->stats.compute_mesh_stats(*state->mesh);
			state->build_basis();

            full_dim_ = state->n_geom_bases * state->mesh->dimension();
        }

        states_ptr_[0]->set_log_level(static_cast<spdlog::level::level_enum>(cur_log)); // log level is global, only need to change in one state

        compute_pattern_period();

        // load shape velocity
        const int dim = get_state().mesh->dimension();
        Eigen::MatrixXd vertices;
        Eigen::MatrixXi cells;
        std::vector<std::vector<int>> elements;
        std::vector<std::vector<double>> weights;
        std::vector<int> body_ids;
        std::vector<std::string> node_data_name;
        std::vector<std::vector<double>> node_data;
        io::MshReader::load(out_velocity_path_, vertices, cells, elements, weights, body_ids, node_data_name, node_data);

        for (int j = 0; j < vertices.rows(); j++)
        {
            auto v1 = vertices.row(j);
            auto v2 = get_state().mesh->point(j);
            if ((v1 - v2).norm() > 1e-6)
                log_and_throw_error("Inconsistent reduced mesh and full mesh, diff = {}!", (v1 - v2).norm());
        }
        
        assert(node_data_name.size() == node_data.size());
        shape_velocity_.setZero(get_state().n_geom_bases * dim, node_data.size() - 1);
        const auto &primitive_to_node = get_state().iso_parametric() ? get_state().primitive_to_bases_node : get_state().primitive_to_geom_bases_node;
        for (int j = 0; j < get_state().n_geom_bases; j++) // j is full vertex id
        {
            if (primitive_to_node[j] >= 0 && primitive_to_node[j] < get_state().n_geom_bases)
                for (int i = 1; i < node_data_name.size(); i++)
                    for (int d = 0; d < dim; d++)
                        shape_velocity_(primitive_to_node[j] * dim + d, i - 1) = node_data[0][full_to_periodic_[j] * 3 + d] * node_data[i][full_to_periodic_[j]];
        }

        // for (int i = 1; i < node_data_name.size(); i++) // debug shape velocity
        // {
        //     static int idx = 0;
        //     std::vector<io::SolutionFrame> solution_frames;
        //     get_state().out_geom.export_data(
        //         get_state(),
        //         shape_velocity_.col(i - 1), Eigen::MatrixXd(),
        //         !get_state().args["time"].is_null(),
        //         0, 0,
        //         io::OutGeometryData::ExportOptions(get_state().args, get_state().mesh->is_linear(), get_state().problem->is_scalar(), get_state().solve_export_to_file),
        //         "shape_vel_" + std::to_string(idx) + ".vtu",
        //         "", // nodes_path,
        //         "", // solution_path,
        //         "", // stress_path,
        //         "", // mises_path,
        //         get_state().is_contact_enabled(), solution_frames);
        //     idx++;
        // }
        
        return true;
    }
}