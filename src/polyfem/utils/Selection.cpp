#include "Selection.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/StringUtils.hpp>

#include <memory>

namespace polyfem::utils
{
	using namespace polyfem::mesh;

	std::shared_ptr<Selection> Selection::build(
		const json &selection,
		const Selection::BBox &mesh_bbox,
		const std::string &root_path)
	{
		if (!selection.contains("id"))
		{
			logger().error("Selection does not contain an id field: {}", selection.dump());
			throw std::runtime_error("Selection id not found");
		}

		std::shared_ptr<Selection> res = nullptr;
		if (selection.contains("box"))
			res = std::make_shared<BoxSelection>(selection, mesh_bbox);
		else if (selection.contains("center"))
			res = std::make_shared<SphereSelection>(selection, mesh_bbox);
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
			log_and_throw_error(fmt::format("Selection not recognized: {}", selection.dump()));

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
			// TODO clean me
			if (!j_selections.contains("threshold"))
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
			log_and_throw_error(fmt::format("Invalid selections: {}", j_selections));
		}
		return selections;
	}

	///////////////////////////////////////////////////////////////////////

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

	///////////////////////////////////////////////////////////////////////

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

	///////////////////////////////////////////////////////////////////////

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

	///////////////////////////////////////////////////////////////////////

	PlaneSelection::PlaneSelection(
		const json &selection,
		const Selection::BBox &mesh_bbox)
		: Selection(selection["id"].get<int>())
	{
		normal_ = selection["normal"];
		normal_.normalized();
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

	///////////////////////////////////////////////////////////////////////

	SpecifiedSelection::SpecifiedSelection(
		const std::vector<int> &ids)
		: Selection(0),
		  ids_(ids)
	{
	}

	int SpecifiedSelection::id(const size_t element_id) const
	{
		return ids_.at(element_id);
	}

	///////////////////////////////////////////////////////////////////////

	FileSelection::FileSelection(
		const std::string &file_path,
		const int id_offset)
	{
		std::ifstream file(file_path);
		if (!file.is_open())
		{
			logger().error("Unable to open selection file \"{}\"!", file_path);
			return;
		}

		std::string line;
		while (std::getline(file, line))
		{
			if (line.empty())
				continue;
			int id;
			std::istringstream(line) >> id;
			this->ids_.push_back(id + id_offset);
		}
	}
} // namespace polyfem::utils
