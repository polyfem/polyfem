#include "FEBioReader.hpp"

#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/utils/RBFInterpolation.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <igl/Timer.h>

#include <tinyxml2.h>

namespace polyfem
{
	using namespace assembler;

	namespace utils
	{
		namespace
		{
			struct BCData
			{
				Eigen::RowVector3d val;
				bool isx, isy, isz;
			};

			std::shared_ptr<Interpolation> get_interpolation(const bool time_dependent)
			{
				if (time_dependent)
					return std::make_shared<LinearInterpolation>();
				else
					return std::make_shared<NoInterpolation>();
			}

			//verify
			template <typename XMLNode>
			bool load_control(const XMLNode *control, json &args)
			{
				const auto *tsn = control->FirstChildElement("time_steps");
				const auto *ssn = control->FirstChildElement("step_size");
				if (tsn && ssn)
				{
					const int time_steps = tsn->IntText();
					const double step_size = ssn->DoubleText();
					args["tend"] = step_size * time_steps;
					args["time_steps"] = time_steps;
				}

				const auto *an = control->FirstChildElement("analysis");

				if (an)
				{
					const std::string type = std::string(an->Attribute("type"));
					return type == "dynamic";
				}

				return false;
			}

			template <typename XMLNode>
			std::string load_materials(const XMLNode *febio, std::map<int, std::tuple<double, double, double, std::string>> &materials)
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

					std::string mat = "";
					if (material == "neo-Hookean")
						mat = "NeoHookean";
					else if (material == "isotropic elastic")
						mat = "LinearElasticity";
					else
					{
						logger().error("Unsupported material {}, reverting to isotropic elastic", material);
						mat = "LinearElasticity";
					}

					material_names.push_back(mat);
					materials[mid] = std::tuple<double, double, double, std::string>(E, nu, rho, mat);
				}

				std::sort(material_names.begin(), material_names.end());
				material_names.erase(std::unique(material_names.begin(), material_names.end()), material_names.end());
				// assert(material_names.size() == 1);
				if (material_names.size() != 1)
				{
					return "MultiModels";
				}
				return material_names.front();
			}

			template <typename XMLNode>
			void load_nodes(const XMLNode *geometry, Eigen::MatrixXd &V)
			{
				std::vector<Eigen::Vector3d> vertices;
				for (const tinyxml2::XMLElement *nodes = geometry->FirstChildElement("Nodes"); nodes != NULL; nodes = nodes->NextSiblingElement("Nodes"))
				{
					for (const tinyxml2::XMLElement *child = nodes->FirstChildElement("node"); child != NULL; child = child->NextSiblingElement("node"))
					{
						const std::string pos_str = std::string(child->GetText());
						const auto vs = StringUtils::split(pos_str, ",");
						assert(vs.size() == 3);

						vertices.emplace_back(atof(vs[0].c_str()), atof(vs[1].c_str()), atof(vs[2].c_str()));
					}
				}

				V.resize(vertices.size(), 3);
				for (int i = 0; i < vertices.size(); ++i)
					V.row(i) = vertices[i].transpose();
			}

			//Need to chage!
			template <typename XMLNode>
			int load_elements(const XMLNode *geometry, const int numV, const std::map<int, std::tuple<double, double, double, std::string>> &materials, Eigen::MatrixXi &T, std::vector<std::vector<int>> &nodes, Eigen::MatrixXd &Es, Eigen::MatrixXd &nus, Eigen::MatrixXd &rhos, std::vector<std::string> &mats, std::vector<int> &mids)
			{
				std::vector<Eigen::VectorXi> els;
				nodes.clear();
				mids.clear();
				int order = 1;
				bool is_hex = false;
				std::string type = "";

				for (const tinyxml2::XMLElement *elements = geometry->FirstChildElement("Elements"); elements != NULL; elements = elements->NextSiblingElement("Elements"))
				{
					const std::string el_type = std::string(elements->Attribute("type"));
					const int mid = elements->IntAttribute("mat");

					if (el_type != "tet4" && el_type != "tet10" && el_type != "tet20" && el_type != "hex8")
					{
						logger().error("Unsupported elemet type {}", el_type);
						continue;
					}

					if (type.empty())
					{
						if (el_type == "tet4" || el_type == "tet10" || el_type == "tet20")
							type = "tet";
						else
							type = "hex";
					}
					else if (el_type.rfind(type, 0) != 0)
					{
						logger().error("Unsupported elemet type {} since the mesh contains also {}", el_type, type);
						continue;
					}

					if (el_type == "tet4")
						order = std::max(1, order);
					else if (el_type == "tet10")
						order = std::max(2, order);
					else if (el_type == "tet20")
						order = std::max(3, order);
					else if (el_type == "hex8")
					{
						order = std::max(1, order);
						is_hex = true;
					}

					for (const tinyxml2::XMLElement *child = elements->FirstChildElement("elem"); child != NULL; child = child->NextSiblingElement("elem"))
					{
						const std::string ids = std::string(child->GetText());
						const auto tt = StringUtils::split(ids, ",");
						assert(tt.size() >= 4);
						const int node_size = is_hex ? 8 : 4;

						els.emplace_back(node_size);

						for (int n = 0; n < node_size; ++n)
						{
							els.back()[n] = atoi(tt[n].c_str()) - 1;
							assert(els.back()[n] < numV);
						}
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

				T.resize(els.size(), is_hex ? 8 : 4);
				Es.resize(els.size(), 1);
				nus.resize(els.size(), 1);
				rhos.resize(els.size(), 1);
				mats.resize(els.size());
				for (int i = 0; i < els.size(); ++i)
				{
					T.row(i) = els[i].transpose();
					const auto it = materials.find(mids[i]);
					assert(it != materials.end());
					if (it == materials.end())
					{
						logger().error("Unable to find material {}", mids[i]);
						throw std::runtime_error("Invalid material");
					}
					Es(i) = std::get<0>(it->second);
					nus(i) = std::get<1>(it->second);
					rhos(i) = std::get<2>(it->second);
					mats[i] = std::get<3>(it->second);
				}

				return order;
			}

			template <typename XMLNode>
			void load_node_sets(const XMLNode *geometry, const int n_nodes, std::vector<std::vector<int>> &nodeSet, std::map<std::string, int> &names)
			{
				nodeSet.resize(n_nodes);
				int id = 1;
				names.clear();

				std::vector<std::vector<int>> prev_nodes;
				std::vector<std::string> tmp_names;

				for (const tinyxml2::XMLElement *child = geometry->FirstChildElement("NodeSet"); child != NULL; child = child->NextSiblingElement("NodeSet"))
				{
					const std::string name = std::string(child->Attribute("name"));
					if (names.find(name) != names.end())
					{
						logger().warn("Nodeset {} already exists", name);
						continue;
					}

					std::vector<int> tmp;
					for (const tinyxml2::XMLElement *nodeid = child->FirstChildElement("node"); nodeid != NULL; nodeid = nodeid->NextSiblingElement("node"))
					{
						const int nid = nodeid->IntAttribute("id");
						tmp.push_back(nid);
					}

					bool found = false;
					for (int i = 0; i < prev_nodes.size(); ++i)
					{
						if (prev_nodes[i] == tmp)
						{
							found = true;
							names[name] = names[tmp_names[i]];
							logger().trace("Id {}, '{}' and '{}' are now the same", names.at(name), name, tmp_names[i]);
							break;
						}
					}

					if (!found)
					{
						prev_nodes.emplace_back(tmp);
						tmp_names.emplace_back(name);
						names[name] = id;
						id++;
					}
				}

				for (const tinyxml2::XMLElement *child = geometry->FirstChildElement("NodeSet"); child != NULL; child = child->NextSiblingElement("NodeSet"))
				{
					const std::string name = std::string(child->Attribute("name"));
					const int lid = names.at(name);

					for (const tinyxml2::XMLElement *nodeid = child->FirstChildElement("node"); nodeid != NULL; nodeid = nodeid->NextSiblingElement("node"))
					{
						const int nid = nodeid->IntAttribute("id");
						nodeSet[nid - 1].push_back(lid);
					}
				}

				//Duplicated surface id
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

					for (const tinyxml2::XMLElement *nodeid = child->FirstChildElement("quad4"); nodeid != NULL; nodeid = nodeid->NextSiblingElement("quad4"))
					{
						const std::string ids = std::string(nodeid->GetText());
						const auto tt = StringUtils::split(ids, ",");
						assert(tt.size() == 4);
						const int index3 = atoi(tt[3].c_str()) - 1;
						nodeSet[atoi(tt[0].c_str()) - 1].push_back(id);
						nodeSet[atoi(tt[1].c_str()) - 1].push_back(id);
						nodeSet[atoi(tt[2].c_str()) - 1].push_back(id);
						nodeSet[index3].push_back(id);
					}

					id++;
				}

				for (auto &n : nodeSet)
				{
					std::sort(n.begin(), n.end());
					n.erase(std::unique(n.begin(), n.end()), n.end());
				}
			}

			//Need to chage!
			template <typename XMLNode>
			void load_boundary_conditions(const XMLNode *boundaries, const std::map<std::string, int> &names, const double dt, const std::string &root_file, GenericTensorProblem &gproblem)
			{
				std::map<int, BCData> allbc;
				for (const tinyxml2::XMLElement *child = boundaries->FirstChildElement("fix"); child != NULL; child = child->NextSiblingElement("fix"))
				{
					const std::string name = std::string(child->Attribute("node_set"));
					if (names.find(name) == names.end())
					{
						logger().error("Sideset {} not present, skipping", name);
						continue;
					}
					const int id = names.at(name);
					const std::string bc = std::string(child->Attribute("bc"));
					const auto bcs = StringUtils::split(bc, ",");

					BCData bcdata;
					bcdata.val = Eigen::RowVector3d::Zero();
					bcdata.isx = false;
					bcdata.isy = false;
					bcdata.isz = false;

					for (const auto &s : bcs)
					{
						if (s == "x")
							bcdata.isx = true;
						else if (s == "y")
							bcdata.isy = true;
						else if (s == "z")
							bcdata.isz = true;
					}

					auto it = allbc.find(id);
					if (it == allbc.end())
					{
						allbc[id] = bcdata;
					}
					else
					{
						if (bcdata.isx)
						{
							assert(!it->second.isx);
							it->second.isx = true;
							it->second.val(0) = 0;
						}
						if (bcdata.isy)
						{
							assert(!it->second.isz);
							it->second.isy = true;
							it->second.val(1) = 0;
						}
						if (bcdata.isz)
						{
							assert(!it->second.isz);
							it->second.isz = true;
							it->second.val(2) = 0;
						}
					}
				}

				for (const tinyxml2::XMLElement *child = boundaries->FirstChildElement("prescribe"); child != NULL; child = child->NextSiblingElement("prescribe"))
				{
					const std::string name = std::string(child->Attribute("node_set"));
					if (names.find(name) == names.end())
					{
						logger().error("Sideset {} not present, skipping", name);
						continue;
					}
					const int id = names.at(name);
					const std::string bc = std::string(child->Attribute("bc"));

					BCData bcdata;
					bcdata.isx = bc == "x";
					bcdata.isy = bc == "y";
					bcdata.isz = bc == "z";

					const double value = atof(child->FirstChildElement("scale")->GetText()) * (gproblem.is_time_dependent() ? dt : 1);
					bcdata.val = Eigen::RowVector3d::Zero();

					if (bcdata.isx)
						bcdata.val(0) = value;
					else if (bcdata.isy)
						bcdata.val(1) = value;
					else if (bcdata.isz)
						bcdata.val(2) = value;

					auto it = allbc.find(id);
					if (it == allbc.end())
					{
						allbc[id] = bcdata;
					}
					else
					{
						if (bcdata.isx)
						{
							assert(!it->second.isx);
							it->second.isx = true;
							it->second.val(0) = bcdata.val(0);
						}
						if (bcdata.isy)
						{
							assert(!it->second.isy);
							it->second.isy = true;
							it->second.val(1) = bcdata.val(1);
						}
						if (bcdata.isz)
						{
							assert(!it->second.isz);
							it->second.isz = true;
							it->second.val(2) = bcdata.val(2);
						}
					}
				}

				for (auto it = allbc.begin(); it != allbc.end(); ++it)
				{
					logger().trace("adding Dirichlet id={} value=({}) fixed=({}, {}, {})", it->first, it->second.val, it->second.isx, it->second.isy, it->second.isz);
					gproblem.add_dirichlet_boundary(it->first, it->second.val, it->second.isx, it->second.isy, it->second.isz, get_interpolation(gproblem.is_time_dependent()));
				}

				for (const tinyxml2::XMLElement *child = boundaries->FirstChildElement("vector_bc"); child != NULL; child = child->NextSiblingElement("vector_bc"))
				{
					const std::string name = std::string(child->Attribute("node_set"));
					if (names.find(name) == names.end())
					{
						logger().error("Sideset {} not present, skipping", name);
						continue;
					}
					const int id = names.at(name);
					const std::string centers = resolve_path(std::string(child->Attribute("centers")), root_file);
					const std::string values = resolve_path(std::string(child->Attribute("values")), root_file);
					const std::string rbf = "thin_plate"; //TODO
					const double eps = 1e-3;              //TODO
					//TODO add is x,y,z

					Eigen::MatrixXd centers_mat, values_mat;
					read_matrix(centers, centers_mat);
					read_matrix(values, values_mat);

					RBFInterpolation interp(values_mat, centers_mat, rbf, eps);
					logger().trace("adding vector Dirichlet id={} centers={} values={} rbf={} eps={}", id, centers, values, rbf, eps);

					gproblem.add_dirichlet_boundary(
						id, [interp](double x, double y, double z, double t) {
							Eigen::Matrix<double, 3, 1> v;
							v[0] = x;
							v[1] = y;
							v[2] = z;
							return interp.interpolate(v);
						},
						true, true, true, get_interpolation(gproblem.is_time_dependent()));
				}

				const bool is_time_dept = gproblem.is_time_dependent();
				for (const tinyxml2::XMLElement *child = boundaries->FirstChildElement("scaling"); child != NULL; child = child->NextSiblingElement("scaling"))
				{
					const std::string centres = std::string(child->Attribute("center"));
					const std::string factors = std::string(child->Attribute("factor"));
					const std::string name = std::string(child->Attribute("node_set"));
					if (names.find(name) == names.end())
					{
						logger().error("Sideset {} not present, skipping", name);
						continue;
					}
					const int id = names.at(name);

					const auto centrec = StringUtils::split(centres, ",");
					if (centrec.size() != 3)
					{
						logger().error("Skipping scaling, center is not 3d");
						continue;
					}
					const Eigen::Vector3d center(
						atof(centrec[0].c_str()),
						atof(centrec[1].c_str()),
						atof(centrec[2].c_str()));

					const double scaling = atof(factors.c_str());
					logger().trace("adding scaling Dirichlet id={} center=({}) scaling={}", id, center.transpose(), scaling);
					gproblem.add_dirichlet_boundary(
						id, [center, scaling, is_time_dept](double x, double y, double z, double t) {
							Eigen::Matrix<double, 3, 1> v;
							Eigen::Matrix<double, 3, 1> target;
							v[0] = x;
							v[1] = y;
							v[2] = z;
							target = v;

							const double s = is_time_dept ? (scaling * t) : scaling;
							target -= center;
							target *= s;
							target += center;
							return (target - v).eval();
						},
						true, true, true);
				}
			}

			//Need to chage!
			template <typename XMLNode>
			void load_loads(const XMLNode *loads, const std::map<std::string, int> &names, const double dt, GenericTensorProblem &gproblem)
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

						Eigen::RowVector3d scalev;
						scalev.setOnes();

						for (const tinyxml2::XMLElement *scale = child->FirstChildElement("scale"); scale != NULL; scale = scale->NextSiblingElement("scale"))
						{
							const std::string scales = std::string(scale->GetText());
							// const int scale_loc = child->IntAttribute("lc");
							scalev.setConstant(atof(scales.c_str()));
						}

						const auto bcs = StringUtils::split(traction, ",");
						assert(bcs.size() == 3);

						Eigen::RowVector3d force(atof(bcs[0].c_str()), atof(bcs[1].c_str()), atof(bcs[2].c_str()));
						force.array() *= scalev.array();

						if (gproblem.is_time_dependent())
							force *= dt;
						logger().trace("adding Neumann id={} force=({})", names.at(name), force);
						gproblem.add_neumann_boundary(names.at(name), force, get_interpolation(gproblem.is_time_dependent()));
					}
					else if (type == "pressure")
					{
						const std::string pressures = std::string(child->FirstChildElement("pressure")->GetText());
						const double pressure = atof(pressures.c_str()) * (gproblem.is_time_dependent() ? dt : 1);
						//TODO added minus here
						logger().trace("adding Pressure id={} pressure={}", names.at(name), -pressure);
						gproblem.add_pressure_boundary(names.at(name), -pressure, get_interpolation(gproblem.is_time_dependent()));
					}
					else
					{
						logger().error("Unsupported surface load {}", type);
					}
				}
			}

			//Need to chage!
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

		void FEBioReader::load(const std::string &path, const json &args_in, State &state, const std::string &export_solution)
		{
			igl::Timer timer;
			timer.start();
			logger().info("Loading feb file...");

			if (!args_in.contains("normalize_mesh"))
				state.args["normalize_mesh"] = false;
			if (!args_in.contains("quadrature_order"))
				state.args["quadrature_order"] = 0;

			if (args_in.contains("export"))
			{
				if (!export_solution.empty() && !args_in["export"].contains("solution_mat"))
					state.args["export"]["solution_mat"] = export_solution;

				if (!args_in["export"].contains("body_ids"))
					state.args["export"]["body_ids"] = true;
			}
			else
			{
				if (!export_solution.empty())
					state.args["export"]["solution_mat"] = export_solution;

				state.args["export"]["body_ids"] = true;
			}

			state.args["root_path"] = path;

			tinyxml2::XMLDocument doc;
			doc.LoadFile(path.c_str());
			if (doc.Error())
			{
				logger().error("Unable to read {}, error {}", path, doc.ErrorStr());
				throw std::runtime_error("Invalid XML");
			}

			const auto *febio = doc.FirstChildElement("febio_spec");
			const std::string ver = std::string(febio->Attribute("version"));
			assert(ver == "2.5");
			if (ver != "2.5")
			{
				logger().error("Unsuppoted FEBio version {}, use 2.5", ver);
				throw std::runtime_error("Unsuppoted FEBio version");
			}

			const auto *control = febio->FirstChildElement("Control");
			bool time_dependent = false;
			if (control)
			{
				time_dependent = load_control(control, state.args);
			}

			std::map<int, std::tuple<double, double, double, std::string>> materials;

			//TODO teseo FIx me
			const std::string formulation_in = load_materials(febio, materials);
			if (!args_in.contains("tensor_formulation"))
				state.args["tensor_formulation"] = formulation_in;

			const auto *geometry = febio->FirstChildElement("Geometry");

			bool has_collisions = false;

			for (const tinyxml2::XMLElement *spair = geometry->FirstChildElement("SurfacePair"); spair != NULL; spair = spair->NextSiblingElement("SurfacePair"))
			{
				has_collisions = true;
			}

			Eigen::MatrixXd V;
			load_nodes(geometry, V);

			const Eigen::MatrixXd box_min = V.colwise().minCoeff();
			const Eigen::MatrixXd box_max = V.colwise().maxCoeff();
			const double diag = (box_max - box_min).norm();

			if (has_collisions)
			{
				state.args["contact"]["enabled"] = true;

				if (!args_in.contains("dhat"))
				{
					state.args["contact"]["dhat"] = 1e-3 * diag;
					state.has_dhat = false;
				}

				if (!args_in.contains("line_search"))
					state.args["solver"]["nonlinear"]["line_search"]["method"] = "backtracking";

				if (args_in.contains("solver_params"))
				{
					if (!args_in["solver_params"].contains("nl_iterations"))
						state.args["solver_params"]["nl_iterations"] = 200;
				}
				else
				{
					state.args["solver_params"]["nl_iterations"] = 200;
				}
			}

			logger().trace("has_collision={}, dhat={}", has_collisions, double(state.args["contact"]["dhat"]));

			if (!args_in.contains("compute_error"))
				state.args["compute_error"] = false;

			Eigen::MatrixXi T;
			std::vector<std::vector<int>> nodes;
			std::vector<int> mids;

			Eigen::MatrixXd Es, nus, rhos;
			std::vector<std::string> mats;
			const int element_order = load_elements(geometry, V.rows(), materials, T, nodes, Es, nus, rhos, mats, mids);
			const int current_order = state.args["space"]["discr_order"];
			state.args["space"]["discr_order"] = std::max(current_order, element_order);

			if (state.args["space"]["discr_order"] == 1)
				state.args["vismesh_rel_area"] = 100000;

			state.load_mesh(V, T);
			if (T.cols() == 4)
				state.mesh->attach_higher_order_nodes(V, nodes);

			state.mesh->set_body_ids(mids);

			if (materials.size() == 1)
			{
				state.args["params"]["E"] = std::get<0>(materials.begin()->second);
				state.args["params"]["nu"] = std::get<1>(materials.begin()->second);
				state.args["params"]["rho"] = std::get<2>(materials.begin()->second);
			}
			else
			{
				json params = state.args["params"];
				state.assembler.set_size(3);
				// state.assembler.set_parameters(params);
				// state.assembler.init_multimaterial(true, Es, nus);
				state.assembler.init_multimodels(mats);
				// state.density.init_multimaterial(rhos);
			}

			std::vector<std::vector<int>> nodeSet;
			std::map<std::string, int> names;
			load_node_sets(geometry, V.rows(), nodeSet, names);
			state.mesh->compute_boundary_ids([&nodeSet](const std::vector<int> &vs, bool is_boundary) {
				if (!is_boundary)
					return 0;
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

			state.problem = std::make_shared<GenericTensorProblem>("GenericTensor");
			GenericTensorProblem &gproblem = *dynamic_cast<GenericTensorProblem *>(state.problem.get());
			gproblem.set_time_dependent(time_dependent);

			const double dt = 1; //double(state.args["tend"]) / int(state.args["time_steps"]);
			const auto *boundaries = febio->FirstChildElement("Boundary");
			load_boundary_conditions(boundaries, names, dt, path, gproblem);

			const auto *loads = febio->FirstChildElement("Loads");
			load_loads(loads, names, dt, gproblem);
			load_body_loads(loads, names, gproblem);

			// state.args["solver"]["nonlinear"]["line_search"]["method"] = "backtracking";
			// state.args["project_to_psd"] = true;

			timer.stop();
			logger().info(" took {}s", timer.getElapsedTime());
		}
	} // namespace utils
} // namespace polyfem
