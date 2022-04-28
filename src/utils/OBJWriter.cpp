// Modified version of read_obj from libigl to include reading polyline elements
// as edges.
//
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <polyfem/OBJWriter.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

namespace polyfem
{
	bool OBJWriter::save(const std::string &path, const Eigen::MatrixXd &v, const Eigen::MatrixXi &e, const Eigen::MatrixXi &f)
	{
		std::ofstream obj(path, std::ios::out);
		if (!obj.is_open())
			return false;

		obj.precision(15);

		for (int i = 0; i < v.rows(); ++i)
			obj << "v " << v(i, 0) << " " << v(i, 1) << " " << (v.cols() > 2 ? v(i, 2) : 0) << "\n";

		for (int i = 0; i < e.rows(); ++i)
			obj << "l " << e(i, 0) + 1 << " " << e(i, 1) + 1 << "\n";

		for (int i = 0; i < f.rows(); ++i)
			obj << "f " << f(i, 0) + 1 << " " << f(i, 1) + 1 << " " << f(i, 2) + 1 << "\n";

		return true;
	}

} // namespace polyfem
