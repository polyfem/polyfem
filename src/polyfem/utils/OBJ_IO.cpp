// Modified version of read_obj from libigl to include reading polyline elements
// as edges.
//
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "OBJ_IO.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

#include <igl/edges.h>
#include <igl/list_to_matrix.h>

#include <polyfem/utils/Logger.hpp>

namespace polyfem
{
	namespace utils
	{
		namespace
		{
			std::string remove_newline(std::string s)
			{
				s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
				return s;
			}
		} // namespace

		bool OBJReader::load(
			const std::string obj_file_name,
			std::vector<std::vector<double>> &V,
			std::vector<std::vector<double>> &TC,
			std::vector<std::vector<double>> &N,
			std::vector<std::vector<int>> &F,
			std::vector<std::vector<int>> &FTC,
			std::vector<std::vector<int>> &FN,
			std::vector<std::vector<int>> &L)
		{
			// Open file, and check for error
			FILE *obj_file = fopen(obj_file_name.c_str(), "r");
			if (obj_file == NULL)
			{
				logger().error("OBJReader::load: {:s} could not be opened!", obj_file_name);
				return false;
			}
			return load(obj_file, V, TC, N, F, FTC, FN, L);
		}

		bool OBJReader::load(
			FILE *obj_file,
			std::vector<std::vector<double>> &V,
			std::vector<std::vector<double>> &TC,
			std::vector<std::vector<double>> &N,
			std::vector<std::vector<int>> &F,
			std::vector<std::vector<int>> &FTC,
			std::vector<std::vector<int>> &FN,
			std::vector<std::vector<int>> &L)
		{
			// File open was successful so clear outputs
			V.clear();
			TC.clear();
			N.clear();
			F.clear();
			FTC.clear();
			FN.clear();
			L.clear();

			// variables and constants to assist parsing the .obj file
			// Constant strings to compare against
			std::string v("v");
			std::string vn("vn");
			std::string vt("vt");
			std::string f("f");
			std::string l("l");
			std::string tic_tac_toe("#");

			const int LINE_MAX_LEN = 2048;

			char line[LINE_MAX_LEN];
			int line_no = 1;
			while (fgets(line, LINE_MAX_LEN, obj_file) != NULL)
			{
				char type[LINE_MAX_LEN];
				// Read first word containing type
				if (sscanf(line, "%s", type) == 1)
				{
					// Get pointer to rest of line right after type
					char *rest_of_line = &line[strlen(type)];
					if (type == v)
					{
						std::istringstream ls(&line[1]);
						std::vector<double> vertex{std::istream_iterator<double>(ls),
												   std::istream_iterator<double>()};

						// if (vertex.size() < 3) {
						//     logger().error(
						//         "OBJReader::load: vertex on line {:d} should have at "
						//         "least 3 coordinates",
						//         line_no);
						//     fclose(obj_file);
						//     return false;
						// }

						V.push_back(vertex);
					}
					else if (type == vn)
					{
						double x[3];
						int count =
							sscanf(rest_of_line, "%lf %lf %lf\n", &x[0], &x[1], &x[2]);
						if (count != 3)
						{
							logger().error(
								"OBJReader::load: normal on line {:d} should have 3 "
								"coordinates",
								line_no);
							fclose(obj_file);
							return false;
						}
						std::vector<double> normal(count);
						for (int i = 0; i < count; i++)
						{
							normal[i] = x[i];
						}
						N.push_back(normal);
					}
					else if (type == vt)
					{
						double x[3];
						int count =
							sscanf(rest_of_line, "%lf %lf %lf\n", &x[0], &x[1], &x[2]);
						if (count != 2 && count != 3)
						{
							logger().error(
								"OBJReader::load: texture coords on line {:d} should have "
								"2 or 3 coordinates (has {:d})",
								line_no, count);
							fclose(obj_file);
							return false;
						}
						std::vector<double> tex(count);
						for (int i = 0; i < count; i++)
						{
							tex[i] = x[i];
						}
						TC.push_back(tex);
					}
					else if (type == f)
					{
						const auto &shift = [&V](const int i) -> int {
							return i < 0 ? i + V.size() : i - 1;
						};
						const auto &shift_t = [&TC](const int i) -> int {
							return i < 0 ? i + TC.size() : i - 1;
						};
						const auto &shift_n = [&N](const int i) -> int {
							return i < 0 ? i + N.size() : i - 1;
						};
						std::vector<int> f;
						std::vector<int> ftc;
						std::vector<int> fn;
						// Read each "word" after type
						char word[LINE_MAX_LEN];
						int offset;
						while (sscanf(rest_of_line, "%s%n", word, &offset) == 1)
						{
							// adjust offset
							rest_of_line += offset;
							// Process word
							long int i, it, in;
							if (sscanf(word, "%ld/%ld/%ld", &i, &it, &in) == 3)
							{
								f.push_back(shift(i));
								ftc.push_back(shift_t(it));
								fn.push_back(shift_n(in));
							}
							else if (sscanf(word, "%ld/%ld", &i, &it) == 2)
							{
								f.push_back(shift(i));
								ftc.push_back(shift_t(it));
							}
							else if (sscanf(word, "%ld//%ld", &i, &in) == 2)
							{
								f.push_back(shift(i));
								fn.push_back(shift_n(in));
							}
							else if (sscanf(word, "%ld", &i) == 1)
							{
								f.push_back(shift(i));
							}
							else
							{
								logger().error(
									"OBJReader::load: face on line {:d} has invalid "
									"element format",
									line_no);
								fclose(obj_file);
								return false;
							}
						}
						if ((f.size() > 0 && fn.size() == 0 && ftc.size() == 0)
							|| (f.size() > 0 && fn.size() == f.size()
								&& ftc.size() == 0)
							|| (f.size() > 0 && fn.size() == 0
								&& ftc.size() == f.size())
							|| (f.size() > 0 && fn.size() == f.size()
								&& ftc.size() == f.size()))
						{
							// No matter what add each type to lists so that lists
							// are the correct lengths
							F.push_back(f);
							FTC.push_back(ftc);
							FN.push_back(fn);
						}
						else
						{
							logger().error(
								"OBJReader::load: face on line {:d} has invalid format",
								line_no);
							fclose(obj_file);
							return false;
						}
					}
					else if (type == l)
					{
						std::istringstream ls(&line[1]);
						std::vector<int> polyline{std::istream_iterator<int>(ls),
												  std::istream_iterator<int>()};

						if (polyline.size() < 2)
						{
							logger().error(
								"OBJReader::load: line element on line {:d} should have "
								"at least 2 vertices",
								line_no);
							fclose(obj_file);
							return false;
						}

						for (int i = 0; i < polyline.size(); i++)
						{
							polyline[i] = polyline[i] < 0 ? polyline[i] + V.size()
														  : polyline[i] - 1;
						}

						L.push_back(polyline);
					}
					else if (
						strlen(type) >= 1
						&& (type[0] == '#' || type[0] == 'g' || type[0] == 's'
							|| strcmp("usemtl", type) == 0
							|| strcmp("mtllib", type) == 0))
					{
						// ignore comments or other stuff
					}
					else
					{
						// ignore any other lines
						std::string line_no_newline = remove_newline(line);
						logger().warn(
							"OBJReader::load: ignored non-comment line {:d}: {:s}", line_no,
							line_no_newline);
					}
				}
				else
				{
					// ignore empty line
				}
				line_no++;
			}
			fclose(obj_file);

			assert(F.size() == FN.size());
			assert(F.size() == FTC.size());

			return true;
		}

		bool OBJReader::load(
			const std::string obj_file_name,
			std::vector<std::vector<double>> &V,
			std::vector<std::vector<int>> &F,
			std::vector<std::vector<int>> &L)
		{
			std::vector<std::vector<double>> TC, N;
			std::vector<std::vector<int>> FTC, FN;
			return load(obj_file_name, V, TC, N, F, FTC, FN, L);
		}

		bool OBJReader::load(
			const std::string str,
			Eigen::MatrixXd &V,
			Eigen::MatrixXi &E,
			Eigen::MatrixXi &F)
		{
			std::vector<std::vector<double>> vV, vTC, vN;
			std::vector<std::vector<int>> vF, vFTC, vFN, vL;
			bool success = load(str, vV, vTC, vN, vF, vFTC, vFN, vL);
			if (!success)
			{
				// load(str,vV,vTC,vN,vF,vFTC,vFN) should have already printed
				// an error message
				return false;
			}
			bool V_rect = igl::list_to_matrix(vV, V);
			if (!V_rect)
			{
				// igl::list_to_matrix(vV,V) already printed error message
				return false;
			}
			bool F_rect = igl::list_to_matrix(vF, F);
			if (!F_rect)
			{
				// igl::list_to_matrix(vF,F) already printed error message
				return false;
			}
			std::vector<std::vector<int>> vE;
			for (const std::vector<int> &polyline : vL)
			{
				for (int i = 1; i < polyline.size(); i++)
				{
					vE.push_back({{polyline[i - 1], polyline[i]}});
				}
			}
			bool E_rect = igl::list_to_matrix(vE, E);
			if (!E_rect)
			{
				logger().error("OBJReader::load: edges not rectangular matrix!");
				return false;
			}
			// if (F.size())
			// {
			// 	Eigen::MatrixXi faceE;
			// 	igl::edges(F, faceE);
			// 	E.conservativeResize(E.rows() + faceE.rows(), 2);
			// 	E.bottomRows(faceE.rows()) = faceE;
			// }

			return true;
		}

		bool OBJWriter::save(const std::string &path, const Eigen::MatrixXd &v, const Eigen::MatrixXi &e, const Eigen::MatrixXi &f)
		{
			const Eigen::IOFormat OBJ_VERTEX_FORMAT(
				/*precision=*/Eigen::FullPrecision,
				/*flags=*/Eigen::DontAlignCols,
				/*coeffSeparator=*/" ",
				/*rowSeparator=*/"",
				/*rowPrefix=*/"v ",
				/*rowSuffix=*/v.cols() == 2 ? " 0\n" : "\n",
				/*matPrefix=*/"",
				/*fill=*/"");

			std::ofstream obj(path, std::ios::out);
			if (!obj.is_open())
				return false;

			for (int i = 0; i < v.rows(); ++i)
				obj << v.row(i).format(OBJ_VERTEX_FORMAT);

			for (int i = 0; i < e.rows(); ++i)
				obj << fmt::format("l {} {}\n", e(i, 0) + 1, e(i, 1) + 1);

			for (int i = 0; i < f.rows(); ++i)
				obj << fmt::format("f {} {} {}\n", f(i, 0) + 1, f(i, 1) + 1, f(i, 2) + 1);

			return true;
		}
	} // namespace utils
} // namespace polyfem
