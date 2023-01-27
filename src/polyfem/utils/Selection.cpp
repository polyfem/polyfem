#include "Selection.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/StringUtils.hpp>

#include <polyfem/io/MatrixIO.hpp>

#include <memory>

namespace polyfem::utils
{
	using namespace polyfem::mesh;

	std::shared_ptr<Selection> Selection::build(
		const json &selection,
		const Selection::BBox &mesh_bbox,
		const std::string &root_path)
	{
		std::shared_ptr<Selection> res = nullptr;
		if (selection.contains("box"))
			res = std::make_shared<BoxSelection>(selection, mesh_bbox);
		else if (selection.contains("threshold"))
			res = std::make_shared<BoxSideSelection>(selection, mesh_bbox);
		else if (selection.contains("center"))
			res = std::make_shared<SphereSelection>(selection, mesh_bbox);
		else if (selection.contains("radius"))
			res = std::make_shared<CylinderSelection>(selection, mesh_bbox);
		else if (selection.contains("axis"))
			res = std::make_shared<AxisPlaneSelection>(selection, mesh_bbox);
		else if (selection.contains("normal"))
			res = std::make_shared<PlaneSelection>(selection, mesh_bbox);
		else if (selection["id"].is_string()) // assume ID is a file path
			res = std::make_shared<FileSelection>(
				resolve_path(selection["id"], root_path), selection.value("id_offset", 0));
		else if (selection["id"].is_number_integer()) // assume ID is uniform
			res = std::make_shared<UniformSelection>(selection["id"]);
		else
			log_and_throw_error("Selection not recognized: {}", selection.dump());

		return res;
	}

	std::vector<std::shared_ptr<Selection>> Selection::build_selections(
		const json &j_selections,
		const Selection::BBox &mesh_bbox,
		const std::string &root_path)
	{
		std::vector<std::shared_ptr<Selection>> selections;
		if (j_selections.is_number_integer())
		{
			selections.push_back(std::make_shared<UniformSelection>(j_selections.get<int>()));
		}
		else if (j_selections.is_string())
		{
			selections.push_back(std::make_shared<FileSelection>(resolve_path(j_selections, root_path)));
		}
		else if (j_selections.is_object())
		{
			selections.push_back(build(j_selections, mesh_bbox));
		}
		else if (j_selections.is_array())
		{
			for (const json &s : j_selections.get<std::vector<json>>())
			{
				selections.push_back(build(s, mesh_bbox));
			}
		}
		else if (!j_selections.is_null())
		{
			log_and_throw_error("Invalid selections: {}", j_selections);
		}
		return selections;
	}

	// ------------------------------------------------------------------------

	BoxSelection::BoxSelection(
		const json &selection,
		const Selection::BBox &mesh_bbox)
		: Selection(selection["id"].get<int>())
	{
		auto bboxj = selection["box"];

		const int dim = bboxj[0].size();
		assert(bboxj[1].size() == dim);

		bbox_[0] = bboxj[0];
		bbox_[1] = bboxj[1];

		if (selection.value("relative", false))
		{
			RowVectorNd mesh_width = mesh_bbox[1] - mesh_bbox[0];
			bbox_[0] = mesh_width.cwiseProduct(bbox_[0]) + mesh_bbox[0];
			bbox_[1] = mesh_width.cwiseProduct(bbox_[1]) + mesh_bbox[0];
		}
	}

	bool BoxSelection::inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const
	{
		assert(bbox_[0].size() == p.size());
		assert(bbox_[1].size() == p.size());
		bool inside = true;

		for (int d = 0; d < p.size(); ++d)
		{
			if (p[d] < bbox_[0][d] || p[d] > bbox_[1][d])
			{
				inside = false;
				break;
			}
		}

		return inside;
	}

	// ------------------------------------------------------------------------

	BoxSideSelection::BoxSideSelection(
		const json &selection,
		const Selection::BBox &mesh_bbox)
		: Selection(0), mesh_bbox_(mesh_bbox)
	{
		tolerance_ = selection.at("threshold");
		if (tolerance_ < 0)
			tolerance_ = mesh_bbox.size() == 3 ? 1e-2 : 1e-7;
		id_offset_ = selection.at("id_offset");
	}

	int BoxSideSelection::id(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const
	{
		assert(p.size() == 2 || p.size() == 3);
		assert(mesh_bbox_[0].size() == p.size());
		assert(mesh_bbox_[1].size() == p.size());
		const auto &[min_corner, max_corner] = mesh_bbox_;

		if (std::abs(p(0) - min_corner(0)) < tolerance_)
			return 1 + id_offset_; // left
		else if (std::abs(p(1) - min_corner(1)) < tolerance_)
			return 2 + id_offset_; // bottom
		else if (std::abs(p(0) - max_corner(0)) < tolerance_)
			return 3 + id_offset_; // right
		else if (std::abs(p(1) - max_corner(1)) < tolerance_)
			return 4 + id_offset_; // top
		else if (p.size() == 3 && std::abs(p(2) - min_corner(2)) < tolerance_)
			return 5 + id_offset_; // back
		else if (p.size() == 3 && std::abs(p(2) - max_corner(2)) < tolerance_)
			return 6 + id_offset_; // front
		else
			return 7 + id_offset_; // all other sides
	}

	// ------------------------------------------------------------------------

	SphereSelection::SphereSelection(
		const json &selection,
		const Selection::BBox &mesh_bbox)
		: Selection(selection["id"].get<int>())
	{
		center_ = selection["center"];
		radius2_ = selection["radius"];

		if (selection.value("relative", false))
		{
			RowVectorNd mesh_width = mesh_bbox[1] - mesh_bbox[0];
			center_ = mesh_width.cwiseProduct(center_) + mesh_bbox[0];
			radius2_ = mesh_width.norm() * radius2_;
		}

		radius2_ *= radius2_;

		id_ = selection["id"];
	}

	bool SphereSelection::inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const
	{
		assert(center_.size() == p.size());

		return (p - center_).squaredNorm() <= radius2_;
	}

	// ------------------------------------------------------------------------

	CylinderSelection::CylinderSelection(
		const json &selection,
		const Selection::BBox &mesh_bbox)
		: Selection(selection["id"].get<int>())
	{
		point_ = selection["p1"];
		RowVectorNd p2 = selection["p2"];
		radius2_ = selection["radius"];

		if (selection.value("relative", false))
		{
			RowVectorNd mesh_width = mesh_bbox[1] - mesh_bbox[0];
			point_ = mesh_width.cwiseProduct(point_) + mesh_bbox[0];
			p2 = mesh_width.cwiseProduct(p2) + mesh_bbox[0];
			radius2_ = mesh_width.norm() * radius2_;
		}

		radius2_ *= radius2_;
		height_ = (point_ - p2).norm();
		axis_ = (p2 - point_).normalized();

		id_ = selection["id"];
	}

	bool CylinderSelection::inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const
	{
		assert(point_.size() == p.size());

		const RowVectorNd v = p - point_;
		const double proj = axis_.dot(v);

		if (proj < 0)
			return false;
		if (proj > height_)
			return false;

		return (v - axis_ * proj).squaredNorm() <= radius2_;
	}

	// ------------------------------------------------------------------------

	AxisPlaneSelection::AxisPlaneSelection(
		const json &selection,
		const Selection::BBox &mesh_bbox)
		: Selection(selection["id"].get<int>())
	{
		position_ = selection["position"];

		if (selection["axis"].is_string())
		{
			std::string axis = selection["axis"];
			int sign = axis[0] == '-' ? -1 : 1;
			int dim = std::tolower(axis.back()) - 'x' + 1;
			assert(dim >= 1 && dim <= 3);
			axis_ = sign * dim;
		}
		else
		{
			assert(selection["axis"].is_number_integer());
			axis_ = selection["axis"];
			assert(std::abs(axis_) >= 1 && std::abs(axis_) <= 3);
		}

		if (selection.value("relative", false))
		{
			int dim = std::abs(axis_) - 1;
			position_ = (mesh_bbox[1][dim] - mesh_bbox[0][dim]) * position_ + mesh_bbox[0][dim];
		}
	}

	bool AxisPlaneSelection::inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const
	{
		const double v = p[std::abs(axis_) - 1];

		if (axis_ > 0)
			return v >= position_;
		else
			return v <= position_;
	}

	// ------------------------------------------------------------------------

	PlaneSelection::PlaneSelection(
		const json &selection,
		const Selection::BBox &mesh_bbox)
		: Selection(selection["id"].get<int>())
	{
		normal_ = selection["normal"];
		normal_.normalize();
		if (selection.contains("point"))
		{
			point_ = selection["point"];
			if (selection.value("relative", false))
				point_ = (mesh_bbox[1] - mesh_bbox[0]).cwiseProduct(point_) + mesh_bbox[0];
		}
		else
			point_ = normal_ * selection.value("offset", 0.0);
	}

	bool PlaneSelection::inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const
	{
		assert(p.size() == normal_.size());
		const RowVectorNd pp = p - point_;
		return pp.dot(normal_) >= 0;
	}

	// ------------------------------------------------------------------------

	SpecifiedSelection::SpecifiedSelection(
		const std::vector<int> &ids)
		: Selection(0),
		  ids_(ids)
	{
	}

	int SpecifiedSelection::id(const size_t element_id, const std::vector<int> &vs, const RowVectorNd &p) const
	{
		return ids_.at(element_id);
	}

	// ------------------------------------------------------------------------

	FileSelection::FileSelection(
		const std::string &file_path,
		const int id_offset)
	{
		Eigen::MatrixXi mat;
		const auto ok = io::read_matrix(file_path, mat);
		if (!ok)
		{
			logger().error("Unable to open selection file \"{}\"!", file_path);
			return;
		}

		if (mat.cols() == 1)
		{
			for (int k = 0; k < mat.size(); ++k)
				this->ids_.push_back(mat(k) + id_offset);
		}
		else
		{
			data_.resize(mat.rows());

			for (int i = 0; i < mat.rows(); ++i)
			{
				data_[i].first = mat(i, 0) + id_offset;

				for (int j = 1; j < mat.cols(); ++j)
				{
					data_[i].second.push_back(mat(i, j));
				}

				std::sort(data_[i].second.begin(), data_[i].second.end());
			}
		}
	}

	bool FileSelection::inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const
	{
		if (data_.empty())
			return SpecifiedSelection::inside(p_id, vs, p);

		std::vector<int> tmp;
		for (const auto &t : data_)
		{
			if (t.second == vs)
				return true;
		}
		return false;
	}

	int FileSelection::id(const size_t element_id, const std::vector<int> &vs, const RowVectorNd &p) const
	{
		if (data_.empty())
			return SpecifiedSelection::id(element_id, vs, p);

		std::vector<int> tmp;
		for (const auto &t : data_)
		{
			if (t.second == vs)
				return t.first;
		}
		return -1;
	}
} // namespace polyfem::utils
