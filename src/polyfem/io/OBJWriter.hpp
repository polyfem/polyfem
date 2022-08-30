// Modified version of read_obj from libigl to include reading polyline elements
// as edges.
//
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <string>

#include <Eigen/Core>

namespace polyfem::io
{
	class OBJWriter
	{
	public:
		OBJWriter() = delete;

		static bool write(
			const std::string &path,
			const Eigen::MatrixXd &v,
			const Eigen::MatrixXi &e,
			const Eigen::MatrixXi &f);

		static bool write(
			const std::string &path,
			const Eigen::MatrixXd &v,
			const Eigen::MatrixXi &e_or_f)
		{
			if (e_or_f.cols() == 2)
				return write(path, v, e_or_f, Eigen::MatrixXi());
			else
				return write(path, v, Eigen::MatrixXi(), e_or_f);
		}
	};
} // namespace polyfem::io
