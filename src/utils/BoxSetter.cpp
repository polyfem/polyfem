#include <polyfem/BoxSetter.hpp>

#include <polyfem/Logger.hpp>
#include <polyfem/Types.hpp>

#include <memory>

namespace polyfem
{
    std::shared_ptr<Selection> Selection::build(const json &selection)
    {
        std::shared_ptr<Selection> res = nullptr;
        if (selection.find("box") != selection.end())
            res = std::make_shared<Box>(selection);
        else if (selection.find("center") != selection.end())
            res = std::make_shared<Sphere>(selection);
        else if (selection.find("axis") != selection.end())
            res = std::make_shared<AxisPlane>(selection);
        else if (selection.find("normal") != selection.end())
            res = std::make_shared<Plane>(selection);
        else
        {
            logger().error("Selection not recognised: {}", selection.dump());
        }

        return res;
    }

    Box::Box(const json &selection)
    {
        auto bboxj = selection["box"];
        const int dim = bboxj[0].size();
        assert(bboxj[1].size() == dim);
        bbox_[0].resize(dim);
        bbox_[1].resize(dim);

        for (size_t k = 0; k < dim; ++k)
        {
            bbox_[0][k] = bboxj[0][k];
            bbox_[1][k] = bboxj[1][k];
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

    Sphere::Sphere(const json &selection)
    {
        auto center = selection["center"];
        radius2_ = selection["radius"];
        radius2_ *= radius2_;

        const int dim = center.size();
        center_.resize(dim);

        for (size_t k = 0; k < dim; ++k)
        {
            center_[k] = center[k];
        }
    }

    bool Sphere::inside(const RowVectorNd &p) const
    {
        assert(center_.size() == p.size());

        return (p - center_).squaredNorm() <= radius2_;
    }

    AxisPlane::AxisPlane(const json &selection)
    {
        position_ = selection["position"];
        axis_ = selection["axis"];
    }

    bool AxisPlane::inside(const RowVectorNd &p) const
    {
        const double v = p[std::abs(axis_) - 1];

        if (axis_ > 0)
            return v >= position_;
        else
            return v <= position_;
    }

    Plane::Plane(const json &selection)
    {
        auto normal = selection["normal"];
        const int dim = normal.size();
        normal_.resize(dim);

        for (size_t k = 0; k < dim; ++k)
        {
            normal_[k] = normal[k];
        }

        const double offset = selection["offset"];
        point_ = normal_ * offset;
    }

    bool Plane::inside(const RowVectorNd &p) const
    {
        assert(p.size() == normal_.size());
        const RowVectorNd pp = p - point_;
        return pp.dot(normal_) >= 0;
    }

    namespace
    {
        std::vector<std::pair<int, std::shared_ptr<Selection>>> get_selections(const json &args, const std::string &key)
        {
            std::vector<std::pair<int, std::shared_ptr<Selection>>> selections;

            if (args.find(key) != args.end())
            {
                const auto boundary = args[key];
                assert(boundary.is_array());

                for (size_t i = 0; i < boundary.size(); ++i)
                {
                    const auto selection = boundary[i];
                    int id = selection["id"];

                    selections.emplace_back(id, Selection::build(selection));
                }
            }

            return selections;
        }
    } // namespace

    void BoxSetter::set_sidesets(const json &args, Mesh &mesh)
    {
        std::vector<std::pair<int, std::shared_ptr<Selection>>> boundary = get_selections(args, "boundary_sidesets");
        std::vector<std::pair<int, std::shared_ptr<Selection>>> body = get_selections(args, "body_ids");

        if (!boundary.empty())
        {
            mesh.compute_boundary_ids([&boundary](const RowVectorNd &p) {
                for (const auto &b : boundary)
                {
                    const auto &selection = b.second;
                    const bool inside = selection->inside(p);

                    if (inside)
                        return b.first;
                }

                return 0;
            });
        }

        if (!body.empty())
        {
            mesh.compute_body_ids([&body](const RowVectorNd &p) {
                for (const auto &b : body)
                {
                    const auto &selection = b.second;
                    const bool inside = selection->inside(p);

                    if (inside)
                        return b.first;
                }

                return 0;
            });
        }
    }
} // namespace polyfem