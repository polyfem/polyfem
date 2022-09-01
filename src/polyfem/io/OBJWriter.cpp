// Modified version of read_obj from libigl to include reading polyline elements
// as edges.
//
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "OBJWriter.hpp"

#include <polyfem/utils/Logger.hpp>

#include <fstream>

namespace polyfem::io
{
	bool OBJWriter::write(const std::string &path, const Eigen::MatrixXd &v, const Eigen::MatrixXi &e, const Eigen::MatrixXi &f)
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

		obj << fmt::format(
			"# Vertices: {:d}\n# Edges: {:d}\n# Faces: {:d}\n",
			v.rows(), e.rows(), f.rows());

		for (int i = 0; i < v.rows(); ++i)
			obj << v.row(i).format(OBJ_VERTEX_FORMAT);

		for (int i = 0; i < e.rows(); ++i)
			obj << fmt::format("l {} {}\n", e(i, 0) + 1, e(i, 1) + 1);

		for (int i = 0; i < f.rows(); ++i)
			obj << fmt::format("f {} {} {}\n", f(i, 0) + 1, f(i, 1) + 1, f(i, 2) + 1);

		return true;
	}
} // namespace polyfem::io
