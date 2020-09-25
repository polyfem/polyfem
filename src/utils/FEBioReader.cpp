#include <polyfem/FEBioReader.hpp>

#include <polyfem/GenericProblem.hpp>
#include <polyfem/AssemblerUtils.hpp>

#include <polyfem/StringUtils.hpp>
#include <polyfem/Logger.hpp>

#include <igl/Timer.h>

#include <tinyxml2.h>

namespace polyfem
{
    namespace
    {
        template <typename XMLNode>
        std::string load_materials(const XMLNode *febio, std::map<int, std::tuple<double, double, double>> &materials)
        {
            double E;
            double nu;
            double rho;
            std::vector<std::string> material_names;

            const tinyxml2::XMLElement *material_parent = febio->FirstChildElement("Material");
            for (const tinyxml2::XMLElement *material_node = material_parent->FirstChildElement("material"); material_node != NULL; material_node = material_node->NextSiblingElement("material"))
            {
                const std::string material = std::string(material_node->Attribute("type"));
                const int mid = material_node->IntAttribute("id");

                E = material_node->FirstChildElement("E")->DoubleText();
                nu = material_node->FirstChildElement("v")->DoubleText();
                if (material_node->FirstChildElement("density"))
                    rho = material_node->FirstChildElement("density")->DoubleText();
                else
                    rho = 1;

                //TODO check if all the same
                if (material == "neo-Hookean")
                    material_names.push_back("NeoHookean");
                else if (material == "isotropic elastic")
                    material_names.push_back("LinearElasticity");
                else
                    logger().error("Unsupported material {}", material);

                materials[mid] = std::tuple<double, double, double>(E, nu, rho);
            }

            std::sort(material_names.begin(), material_names.end());
            material_names.erase(std::unique(material_names.begin(), material_names.end()), material_names.end());
            assert(material_names.size() == 1);
            return material_names.front();
        }

        template <typename XMLNode>
        void load_nodes(const XMLNode *geometry, Eigen::MatrixXd &V)
        {
            const auto *nodes = geometry->FirstChildElement("Nodes");
            std::vector<Eigen::Vector3d> vertices;
            for (const tinyxml2::XMLElement *child = nodes->FirstChildElement("node"); child != NULL; child = child->NextSiblingElement("node"))
            {
                const std::string pos_str = std::string(child->GetText());
                const auto vs = StringUtils::split(pos_str, ",");
                assert(vs.size() == 3);

                vertices.emplace_back(atof(vs[0].c_str()), atof(vs[1].c_str()), atof(vs[2].c_str()));
            }

            V.resize(vertices.size(), 3);
            for (int i = 0; i < vertices.size(); ++i)
                V.row(i) = vertices[i].transpose();
        }

        template <typename XMLNode>
        int load_elements(const XMLNode *geometry, const std::map<int, std::tuple<double, double, double>> &materials, Eigen::MatrixXi &T, std::vector<std::vector<int>> &nodes, Eigen::MatrixXd &Es, Eigen::MatrixXd &nus, Eigen::MatrixXd &rhos)
        {
            std::vector<Eigen::Vector4i> tets;
            nodes.clear();
            std::vector<int> mids;
            int order = 1;

            for (const tinyxml2::XMLElement *elements = geometry->FirstChildElement("Elements"); elements != NULL; elements = elements->NextSiblingElement("Elements"))
            {
                const std::string el_type = std::string(elements->Attribute("type"));
                const int mid = elements->IntAttribute("mat");

                if (el_type != "tet4" && el_type != "tet10" && el_type != "tet20")
                {
                    logger().error("Unsupported elemet type {}", el_type);
                    continue;
                }

                if (el_type == "tet4")
                    order = std::max(1, order);
                else if (el_type == "tet10")
                    order = std::max(2, order);
                else if (el_type == "tet20")
                    order = std::max(3, order);

                for (const tinyxml2::XMLElement *child = elements->FirstChildElement("elem"); child != NULL; child = child->NextSiblingElement("elem"))
                {
                    const std::string ids = std::string(child->GetText());
                    const auto tt = StringUtils::split(ids, ",");
                    assert(tt.size() >= 4);

                    tets.emplace_back(atoi(tt[0].c_str()) - 1, atoi(tt[1].c_str()) - 1, atoi(tt[2].c_str()) - 1, atoi(tt[3].c_str()) - 1);
                    nodes.emplace_back();
                    mids.emplace_back(mid);
                    for (int n = 0; n < tt.size(); ++n)
                        nodes.back().push_back(atoi(tt[n].c_str()) - 1);

                    if (el_type == "tet10")
                    {
                        assert(nodes.back().size() == 10);
                        std::swap(nodes.back()[8], nodes.back()[9]);
                    }
                    else if (el_type == "tet20")
                    {
                        assert(nodes.back().size() == 20);
                        std::swap(nodes.back()[8], nodes.back()[9]);
                        std::swap(nodes.back()[10], nodes.back()[11]);
                        std::swap(nodes.back()[12], nodes.back()[15]);
                        std::swap(nodes.back()[13], nodes.back()[14]);
                        std::swap(nodes.back()[16], nodes.back()[19]);
                        std::swap(nodes.back()[17], nodes.back()[19]);
                    }
                }
            }

            T.resize(tets.size(), 4);
            Es.resize(tets.size(), 1);
            nus.resize(tets.size(), 1);
            rhos.resize(tets.size(), 1);
            for (int i = 0; i < tets.size(); ++i)
            {
                T.row(i) = tets[i].transpose();
                const auto it = materials.find(mids[i]);
                assert(it != materials.end());
                Es(i) = std::get<0>(it->second);
                nus(i) = std::get<1>(it->second);
                rhos(i) = std::get<2>(it->second);
            }

            return order;
        }

        template <typename XMLNode>
        void load_node_sets(const XMLNode *geometry, const int n_nodes, std::vector<std::vector<int>> &nodeSet, std::map<std::string, int> &names)
        {
            nodeSet.resize(n_nodes);
            int id = 1;
            names.clear();

            for (const tinyxml2::XMLElement *child = geometry->FirstChildElement("NodeSet"); child != NULL; child = child->NextSiblingElement("NodeSet"))
            {
                const std::string name = std::string(child->Attribute("name"));
                names[name] = id;

                for (const tinyxml2::XMLElement *nodeid = child->FirstChildElement("node"); nodeid != NULL; nodeid = nodeid->NextSiblingElement("node"))
                {
                    const int nid = nodeid->IntAttribute("id");
                    nodeSet[nid - 1].push_back(id);
                }

                id++;
            }

            for (const tinyxml2::XMLElement *child = geometry->FirstChildElement("Surface"); child != NULL; child = child->NextSiblingElement("Surface"))
            {
                const std::string name = std::string(child->Attribute("name"));
                names[name] = id;

                //TODO  only tri3
                for (const tinyxml2::XMLElement *nodeid = child->FirstChildElement("tri3"); nodeid != NULL; nodeid = nodeid->NextSiblingElement("tri3"))
                {
                    const std::string ids = std::string(nodeid->GetText());
                    const auto tt = StringUtils::split(ids, ",");
                    assert(tt.size() == 3);
                    nodeSet[atoi(tt[0].c_str()) - 1].push_back(id);
                    nodeSet[atoi(tt[1].c_str()) - 1].push_back(id);
                    nodeSet[atoi(tt[2].c_str()) - 1].push_back(id);
                }

                id++;
            }

            for (auto &n : nodeSet)
            {
                std::sort(n.begin(), n.end());
                n.erase(std::unique(n.begin(), n.end()), n.end());
            }
        }

        template <typename XMLNode>
        void load_boundary_conditions(const XMLNode *boundaries, const std::map<std::string, int> &names, GenericTensorProblem &gproblem)
        {
            for (const tinyxml2::XMLElement *child = boundaries->FirstChildElement("fix"); child != NULL; child = child->NextSiblingElement("fix"))
            {
                const std::string name = std::string(child->Attribute("node_set"));
                const std::string bc = std::string(child->Attribute("bc"));
                const auto bcs = StringUtils::split(bc, ",");

                bool isx = false;
                bool isy = false;
                bool isz = false;

                for (const auto &s : bcs)
                {
                    if (s == "x")
                        isx = true;
                    else if (s == "y")
                        isy = true;
                    else if (s == "z")
                        isz = true;
                }
                gproblem.add_dirichlet_boundary(names.at(name), Eigen::RowVector3d::Zero(), isx, isy, isz);
            }

            for (const tinyxml2::XMLElement *child = boundaries->FirstChildElement("prescribe"); child != NULL; child = child->NextSiblingElement("prescribe"))
            {
                const std::string name = std::string(child->Attribute("node_set"));
                const std::string bc = std::string(child->Attribute("bc"));

                bool isx = bc == "x";
                bool isy = bc == "y";
                bool isz = bc == "z";

                const double value = atof(child->FirstChildElement("scale")->GetText());
                Eigen::RowVector3d val = Eigen::RowVector3d::Zero();

                if (isx)
                    val(0) = value;
                else if (isy)
                    val(1) = value;
                else if (isz)
                    val(2) = value;

                gproblem.add_dirichlet_boundary(names.at(name), val, isx, isy, isz);
            }
        }

        template <typename XMLNode>
        void load_loads(const XMLNode *loads, const std::map<std::string, int> &names, GenericTensorProblem &gproblem)
        {
            if (loads == nullptr)
                return;

            for (const tinyxml2::XMLElement *child = loads->FirstChildElement("surface_load"); child != NULL; child = child->NextSiblingElement("surface_load"))
            {
                const std::string name = std::string(child->Attribute("surface"));
                const std::string type = std::string(child->Attribute("type"));
                if (type == "traction")
                {
                    const std::string traction = std::string(child->FirstChildElement("traction")->GetText());
                    const auto bcs = StringUtils::split(traction, ",");
                    assert(bcs.size() == 3);

                    Eigen::RowVector3d force(atof(bcs[0].c_str()), atof(bcs[1].c_str()), atof(bcs[2].c_str()));
                    gproblem.add_neumann_boundary(names.at(name), force);
                }
                else if (type == "pressure")
                {
                    const std::string pressures = std::string(child->FirstChildElement("pressure")->GetText());
                    const double pressure = atof(pressures.c_str());
                    //TODO added minus here
                    gproblem.add_pressure_boundary(names.at(name), -pressure);
                }
                else
                {
                    logger().error("Unsupported surface load {}", type);
                }
            }
        }

        template <typename XMLNode>
        void load_body_loads(const XMLNode *loads, const std::map<std::string, int> &names, GenericTensorProblem &gproblem)
        {
            if (loads == nullptr)
                return;

            int counter = 0;
            for (const tinyxml2::XMLElement *child = loads->FirstChildElement("body_load"); child != NULL; child = child->NextSiblingElement("body_load"))
            {
                ++counter;

                const std::string name = std::string(child->Attribute("elem_set"));
                const std::string type = std::string(child->Attribute("type"));
                if (type == "const")
                {
                    const std::string xs = std::string(child->FirstChildElement("x")->GetText());
                    const std::string ys = std::string(child->FirstChildElement("y")->GetText());
                    const std::string zs = std::string(child->FirstChildElement("z")->GetText());

                    const double x = atof(xs.c_str());
                    const double y = atof(ys.c_str());
                    const double z = atof(zs.c_str());

                    gproblem.set_rhs(x, y, z);
                }
                else
                {
                    logger().error("Unsupported surface load {}", type);
                }
            }

            if (counter > 1)
                logger().error("Loading only last body load");
        }
    } // namespace

    void FEBioReader::load(const std::string &path, State &state, const std::string &export_solution)
    {
        igl::Timer timer;
        timer.start();
        logger().info("Loading feb file...");

        state.args["normalize_mesh"] = false;
        if (!export_solution.empty())
            state.args["export"]["solution_mat"] = export_solution;

        tinyxml2::XMLDocument doc;
        doc.LoadFile(path.c_str());

        const auto *febio = doc.FirstChildElement("febio_spec");
        const std::string ver = std::string(febio->Attribute("version"));
        assert(ver == "2.5");

        std::map<int, std::tuple<double, double, double>> materials;
        state.args["tensor_formulation"] = load_materials(febio, materials);

        const auto *geometry = febio->FirstChildElement("Geometry");

        Eigen::MatrixXd V;
        load_nodes(geometry, V);

        Eigen::MatrixXi T;
        std::vector<std::vector<int>> nodes;
        Eigen::MatrixXd Es, nus, rhos;
        const int element_order = load_elements(geometry, materials, T, nodes, Es, nus, rhos);
        const int current_order = state.args["discr_order"];
        state.args["discr_order"] = std::max(current_order, element_order);

        state.load_mesh(V, T);
        state.mesh->attach_higher_order_nodes(V, nodes);

        if (materials.size() == 1)
        {
            state.args["params"]["E"] = std::get<0>(materials.begin()->second);
            state.args["params"]["nu"] = std::get<1>(materials.begin()->second);
            state.args["params"]["rho"] = std::get<2>(materials.begin()->second);
        }
        else
        {
            AssemblerUtils::instance().init_multimaterial(Es, nus);
            state.density.init_multimaterial(rhos);
        }

        std::vector<std::vector<int>> nodeSet;
        std::map<std::string, int> names;
        load_node_sets(geometry, V.rows(), nodeSet, names);
        state.mesh->compute_boundary_ids([&nodeSet](const std::vector<int> &vs, bool is_boundary) {
            std::vector<int> tmp;
            for (const int v : vs)
                tmp.insert(tmp.end(), nodeSet[v].begin(), nodeSet[v].end());

            std::sort(tmp.begin(), tmp.end());

            int prev = -1;
            int count = 1;
            for (const int id : tmp)
            {
                if (id == prev)
                    count++;
                else
                {
                    count = 1;
                    prev = id;
                }
                if (count == vs.size())
                    return prev;
            }

            return 0;
        });

        state.problem = ProblemFactory::factory().get_problem("GenericTensor");
        GenericTensorProblem &gproblem = *dynamic_cast<GenericTensorProblem *>(state.problem.get());

        const auto *boundaries = febio->FirstChildElement("Boundary");
        load_boundary_conditions(boundaries, names, gproblem);

        const auto *loads = febio->FirstChildElement("Loads");
        load_loads(loads, names, gproblem);
        load_body_loads(loads, names, gproblem);

        timer.stop();
        logger().info(" took {}s", timer.getElapsedTime());
    }
} // namespace polyfem
