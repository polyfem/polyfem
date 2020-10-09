#include <polyfem/BoxSetter.hpp>

#include <polyfem/Types.hpp>

namespace polyfem
{
    typedef std::array<RowVectorNd, 2> BBox;
    namespace
    {
        std::vector<std::pair<int, BBox>> get_boxes(const json &args, const std::string &key)
        {
            std::vector<std::pair<int, BBox>> boxes;

            if (args.find(key) != args.end())
            {
                const auto boundary = args[key];
                assert(boundary.is_array());

                for (size_t i = 0; i < boundary.size(); ++i)
                {
                    const auto box = boundary[i];
                    int id = box["id"];
                    auto bboxj = box["box"];
                    BBox bbox;
                    const int dim = bboxj[0].size();
                    bbox[0].resize(dim);
                    bbox[1].resize(dim);

                    for (size_t k = 0; k < dim; ++k)
                    {
                        bbox[0][k] = bboxj[0][k];
                        bbox[1][k] = bboxj[1][k];
                    }

                    boxes.emplace_back(id, bbox);
                }
            }

            return boxes;
        }
    } // namespace

    void BoxSetter::set_sidesets(const json &args, Mesh &mesh)
    {

        std::vector<std::pair<int, BBox>> boundary = get_boxes(args, "boundary_sidesets");
        std::vector<std::pair<int, BBox>> body = get_boxes(args, "body_ids");

        if (!boundary.empty())
        {
            mesh.compute_boundary_ids([&boundary](const RowVectorNd &p) {
                for (const auto &b : boundary)
                {
                    const auto &bbox = b.second;
                    assert(bbox[0].size() == p.size());
                    bool inside = true;

                    for (int d = 0; d < p.size(); ++d)
                    {
                        if (p[d] < bbox[0][d] || p[d] > bbox[1][d])
                        {
                            inside = false;
                            break;
                        }
                    }

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
                    const auto &bbox = b.second;
                    assert(bbox[0].size() == p.size());
                    bool inside = true;

                    for (int d = 0; d < p.size(); ++d)
                    {
                        if (p[d] < bbox[0][d] || p[d] > bbox[1][d])
                        {
                            inside = false;
                            break;
                        }
                    }

                    if (inside)
                        return b.first;
                }

                return 0;
            });
        }
    }
} // namespace polyfem