#include "Selection.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <memory>

namespace polyfem
{
	using namespace mesh;
	namespace utils
	{
		Selection::Selection(
			const int id,
			const size_t start_element_id,
			const size_t end_element_id)
			: id_(id),
			  start_element_id_(start_element_id),
			  end_element_id_(end_element_id)
		{
		}

		bool Selection::inside(const size_t element_id, const RowVectorNd &p) const
		{
			return element_id >= this->start_element_id_ && element_id < this->end_element_id_;
		}

		std::shared_ptr<Selection> Selection::build(
			const json &selection,
			const Selection::BBox &mesh_bbox,
			const size_t start_element_id,
			const size_t end_element_id)
		{
			if (!selection.contains("id"))
			{
				logger().error("Selection does not contain an id field: {}", selection.dump());
				throw std::runtime_error("Selection id not found");
			}

			std::shared_ptr<Selection> res = nullptr;
			if (selection.contains("box"))
				res = std::make_shared<BoxSelection>(selection, mesh_bbox, start_element_id, end_element_id);
			else if (selection.contains("center"))
				res = std::make_shared<SphereSelection>(selection, mesh_bbox, start_element_id, end_element_id);
			else if (selection.contains("axis"))
				res = std::make_shared<AxisPlaneSelection>(selection, mesh_bbox, start_element_id, end_element_id);
			else if (selection.contains("normal"))
				res = std::make_shared<PlaneSelection>(selection, mesh_bbox, start_element_id, end_element_id);
			else if (selection["id"].is_string()) // assume ID is a file path
				res = std::make_shared<FileSelection>(selection["id"], start_element_id, end_element_id, selection.value("id_offset", 0));
			else if (selection["id"].is_number_integer()) // assume ID is uniform
				res = std::make_shared<UniformSelection>(selection["id"], start_element_id, end_element_id);
			else
			{
				logger().error("Selection not recognized: {}", selection.dump());
				throw std::runtime_error("Selection not recognized");
			}

			return res;
		}

		///////////////////////////////////////////////////////////////////////

		BoxSelection::BoxSelection(
			const json &selection,
			const Selection::BBox &mesh_bbox,
			const size_t start_element_id,
			const size_t end_element_id)
			: Selection(selection["id"].get<int>(), start_element_id, end_element_id)
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

		bool BoxSelection::inside(const size_t element_id, const RowVectorNd &p) const
		{
			if (!Selection::inside(element_id, p))
				return false;

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
			const Selection::BBox &mesh_bbox,
			const size_t start_element_id,
			const size_t end_element_id)
			: Selection(selection["id"].get<int>(), start_element_id, end_element_id)
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

		bool SphereSelection::inside(const size_t element_id, const RowVectorNd &p) const
		{
			if (!Selection::inside(element_id, p))
				return false;

			assert(center_.size() == p.size());

			return (p - center_).squaredNorm() <= radius2_;
		}

		///////////////////////////////////////////////////////////////////////

		AxisPlaneSelection::AxisPlaneSelection(
			const json &selection,
			const Selection::BBox &mesh_bbox,
			const size_t start_element_id,
			const size_t end_element_id)
			: Selection(selection["id"].get<int>(), start_element_id, end_element_id)
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

		bool AxisPlaneSelection::inside(const size_t element_id, const RowVectorNd &p) const
		{
			if (!Selection::inside(element_id, p))
				return false;

			const double v = p[std::abs(axis_) - 1];

			if (axis_ > 0)
				return v >= position_;
			else
				return v <= position_;
		}

		///////////////////////////////////////////////////////////////////////

		PlaneSelection::PlaneSelection(
			const json &selection,
			const Selection::BBox &mesh_bbox,
			const size_t start_element_id,
			const size_t end_element_id)
			: Selection(selection["id"].get<int>(), start_element_id, end_element_id)
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

		bool PlaneSelection::inside(const size_t element_id, const RowVectorNd &p) const
		{
			if (!Selection::inside(element_id, p))
				return false;

			assert(p.size() == normal_.size());
			const RowVectorNd pp = p - point_;
			return pp.dot(normal_) >= 0;
		}

		///////////////////////////////////////////////////////////////////////

		SpecifiedSelection::SpecifiedSelection(
			const std::vector<int> &ids,
			const size_t start_element_id,
			const size_t end_element_id)
			: Selection(0, start_element_id, end_element_id),
			  ids_(ids)
		{
			assert(ids.size() == end_element_id - start_element_id);
		}

		int SpecifiedSelection::id(const size_t element_id) const
		{
			assert(element_id >= this->start_element_id_);
			assert(element_id < this->end_element_id_);
			return ids_.at(element_id - this->start_element_id_);
		}

		///////////////////////////////////////////////////////////////////////

		FileSelection::FileSelection(
			const std::string &file_path,
			const size_t start_element_id,
			const size_t end_element_id,
			const int id_offset)
			: SpecifiedSelection(std::vector<int>(end_element_id - start_element_id), start_element_id, end_element_id)
		{
			std::ifstream file(file_path);
			if (!file.is_open())
			{
				logger().error("Unable to open selection file \"{}\"!", file_path);
				return;
			}

			this->ids_.resize(end_element_id - start_element_id, 0);

			std::string line;
			int i = 0;
			while (std::getline(file, line))
			{
				if (line.empty())
					continue;
				assert(i < this->ids_.size());
				int id;
				std::istringstream(line) >> id;
				this->ids_[i++] = id + id_offset;
			}

			if (i != this->ids_.size())
			{
				logger().warn(
					"Selection file \"{}\" is missing {} tag(s). Using 0 as default.",
					file_path, this->ids_.size() - i);
			}
		}
	} // namespace utils
} // namespace polyfem
