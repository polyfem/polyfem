#include "BoxSetter.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <memory>

namespace polyfem
{
	using namespace mesh;

	std::shared_ptr<Selection> Selection::build(const json &selection, const Selection::BBox &mesh_bbox)
	{
		std::shared_ptr<Selection> res = nullptr;
		if (selection.contains("box"))
			res = std::make_shared<Box>(selection, mesh_bbox);
		else if (selection.contains("center"))
			res = std::make_shared<Sphere>(selection, mesh_bbox);
		else if (selection.contains("axis"))
			res = std::make_shared<AxisPlane>(selection, mesh_bbox);
		else if (selection.contains("normal"))
			res = std::make_shared<Plane>(selection, mesh_bbox);
		else
		{
			logger().error("Selection not recognised: {}", selection.dump());
		}

		return res;
	}

	Box::Box(const json &selection, const Selection::BBox &mesh_bbox)
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

	bool Box::inside(const RowVectorNd &p) const
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

	Sphere::Sphere(const json &selection, const Selection::BBox &mesh_bbox)
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
	}

	bool Sphere::inside(const RowVectorNd &p) const
	{
		assert(center_.size() == p.size());

		return (p - center_).squaredNorm() <= radius2_;
	}

	AxisPlane::AxisPlane(const json &selection, const Selection::BBox &mesh_bbox)
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

	bool AxisPlane::inside(const RowVectorNd &p) const
	{
		const double v = p[std::abs(axis_) - 1];

		if (axis_ > 0)
			return v >= position_;
		else
			return v <= position_;
	}

	Plane::Plane(const json &selection, const Selection::BBox &mesh_bbox)
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

	bool Plane::inside(const RowVectorNd &p) const
	{
		assert(p.size() == normal_.size());
		const RowVectorNd pp = p - point_;
		return pp.dot(normal_) >= 0;
	}

	namespace
	{
		std::vector<std::pair<int, std::shared_ptr<Selection>>> get_selections(const json &args, const std::string &key, const Selection::BBox &mesh_bbox)
		{
			std::vector<std::pair<int, std::shared_ptr<Selection>>> selections;

			if (is_param_valid(args, key))
			{
				const auto boundary = args[key];
				assert(boundary.is_array());

				for (size_t i = 0; i < boundary.size(); ++i)
				{
					const auto selection = boundary[i];
					int id = selection["id"];

					selections.emplace_back(id, Selection::build(selection, mesh_bbox));
				}
			}

			return selections;
		}
	} // namespace

	void BoxSetter::set_sidesets(const json &args, Mesh &mesh)
	{
		Selection::BBox mesh_bbox;
		mesh.bounding_box(mesh_bbox[0], mesh_bbox[1]);

		std::vector<std::pair<int, std::shared_ptr<Selection>>> boundary = get_selections(args, "boundary_sidesets", mesh_bbox);
		std::vector<std::pair<int, std::shared_ptr<Selection>>> body = get_selections(args, "body_ids", mesh_bbox);

		if (!boundary.empty())
		{
			mesh.compute_boundary_ids([&boundary](const RowVectorNd &p) {
				for (const auto &[id, selection] : boundary)
				{
					if (selection->inside(p))
						return id;
				}

				return 0;
			});
		}

		if (!body.empty())
		{
			mesh.compute_body_ids([&body](const RowVectorNd &p) {
				for (const auto &[id, selection] : body)
				{
					if (selection->inside(p))
						return id;
				}

				return 0;
			});
		}
	}
} // namespace polyfem
