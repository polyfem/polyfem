#include "GraphParameter.hpp"
#include <polyfem/io/MshReader.hpp>

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
    }

    GraphParameter::GraphParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args) : Parameter(states_ptr, args)
    {
        parameter_name_ = "shape";
        
        graph_exe_path_ = args["graph_exe_path"];
        graph_path_ = args["graph_path"];
        symmetry_type_ = args["symmetry_type"];
        out_velocity_path_ = args["out_velocity_path"];

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
        
        pre_solve(initial_guess_);
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
        return bounds_.col(0);
    }

    Eigen::VectorXd GraphParameter::get_upper_bound(const Eigen::VectorXd &x) const 
    {
        return bounds_.col(1);
    }

    bool GraphParameter::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
    {
        if ((x1 - bounds_.col(0)).minCoeff() < 0)
            return false;
        if ((bounds_.col(1) - x1).minCoeff() < 0)
            return false;

        return true;
    }

    bool GraphParameter::pre_solve(const Eigen::VectorXd &newX)
    {
        // send parameters and graph to microstructure inflator
        std::string out_mesh_path = get_state().resolve_output_path("micro-tmp.msh");
        
        std::string shape_params = "--params \"";
        for (int i = 0; i < newX.size(); i++)
            shape_params += to_string_with_precision(newX(i), 16) + " ";
        shape_params += "\" ";

        std::string command = graph_exe_path_ + " " + symmetry_type_ + " " + graph_path_ + " " + shape_params + " -S " + out_velocity_path_ + " " + out_mesh_path;

        int return_val = system(command.c_str());
        if (return_val == 0)
            logger().info("remesh command \"{}\" returns {}", command, return_val);
        else
            log_and_throw_error("remesh command \"{}\" returns {}", command, return_val);

        // reload mesh and recompute basis
        for (auto state : states_ptr_)
        {
            assert(state->in_args["geometry"].size() == 1);
            state->in_args["geometry"][0]["mesh"] = out_mesh_path;

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
        
        assert(node_data_name.size() == node_data.size());
        shape_velocity_.setZero(get_state().n_geom_bases * dim, node_data.size() - 1);
        const auto &primitive_to_node = get_state().iso_parametric() ? get_state().primitive_to_bases_node : get_state().primitive_to_geom_bases_node;
        for (int i = 1; i < node_data_name.size(); i++)
        {
            for (int j = 0; j < get_state().n_geom_bases; j++)
                for (int d = 0; d < dim; d++)
                    if (primitive_to_node[j] >= 0 && primitive_to_node[j] < get_state().n_geom_bases)
                        shape_velocity_(primitive_to_node[j] * dim + d, i - 1) = node_data[0][j * 3 + d] * node_data[i][j];

            // debug shape velocity
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
        }
        

        return true;
    }
}