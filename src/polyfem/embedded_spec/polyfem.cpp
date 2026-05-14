#include "polyfem.hpp"

#include <string>

namespace jse
{
namespace embed
{
namespace polyfem
{

const nlohmann::json &spec()
{
    static const nlohmann::json value = []() {
        std::string text;
        text.reserve(348642);
        text += R"JSE_JSON(
[
    {
        "doc": "Root of the configuration file.",
        "optional": [
            "units",
            "preset_problem",
            "common",
            "root_path",
            "space",
            "time",
            "contact",
            "solver",
            "boundary_conditions",
            "initial_conditions",
            "constraints",
            "output",
            "input",
            "tests"
        ],
        "pointer": "/",
        "required": [
            "geometry",
            "materials"
        ],
        "type": "object"
    },
    {
        "default": "",
        "doc": "Path to common settings will patch the current file.",
        "extensions": [
            ".json"
        ],
        "pointer": "/common",
        "type": "file"
    },
    {
        "default": "",
        "doc": "Path for all relative paths, set automatically to the folder containing this JSON.",
        "pointer": "/root_path",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Basic units used in the code.",
        "optional": [
            "length",
            "mass",
            "time",
            "characteristic_length"
        ],
        "pointer": "/units",
        "type": "object"
    },
    {
        "default": "m",
        "doc": "Length unit.",
        "pointer": "/units/length",
        "type": "string"
    },
    {
        "default": "kg",
        "doc": "Mass unit.",
        "pointer": "/units/mass",
        "type": "string"
    },
    {
        "default": "s",
        "doc": "Time unit.",
        "pointer": "/units/time",
        "type": "string"
    },
    {
        "default": 1,
        "doc": "Characteristic length, used for tolerances.",
        "pointer": "/units/characteristic_length",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Used to test to compare different norms of solutions.",
        "optional": [
            "err_h1",
            "err_h1_semi",
            "err_l2",
            "err_linf",
            "err_linf_grad",
            "err_lp",
            "margin",
            "time_steps"
        ],
        "pointer": "/tests",
        "type": "object"
    },
    {
        "default": 0,
        "doc": "Reference h1 solution's norm.",
        "pointer": "/tests/err_h1",
        "type": "float"
    },
    {
        "default": 0,
        "doc": "Reference h1 seminorm solution's norm.",
        "pointer": "/tests/err_h1_semi",
        "type": "float"
    },
    {
        "default": 0,
        "doc": "Reference $L^2$ solution's norm.",
        "pointer": "/tests/err_l2",
        "type": "float"
    },
    {
        "default": 0,
        "doc": "Reference $L^\\infty$ solution's norm.",
        "pointer": "/tests/err_linf",
        "type": "float"
    },
    {
        "default": 0,
        "doc": "Reference $L^\\infty$ solution's gradient norm.",
        "pointer": "/tests/err_linf_grad",
        "type": "float"
    },
    {
        "default": 0,
        "doc": "Reference $L^8$ solution's gradient norm.",
        "pointer": "/tests/err_lp",
        "type": "float"
    },
    {
        "default": 1e-05,
        "doc": "Reference tolerance used in tests.",
        "pointer": "/tests/margin",
        "type": "float"
    },
    {
        "default": 1,
        "doc": "Number of time steps to test.",
        "min": 1,
        "pointer": "/tests/time_steps",
        "type": "int"
    },
    {
        "doc": "Number of time steps to test.",
        "options": [
            "all",
            "static"
        ],
        "pointer": "/tests/time_steps",
        "type": "string"
    },
    {
        "doc": "List of geometry objects.",
        "min": 1,
        "pointer": "/geometry",
        "type": "list"
    },
    {
        "#type_name": "mesh",
        "doc": "Each geometry object stores a mesh, a set of transformations applied to it after loading, and a set of selections, which can be used to specify boundary conditions, materials, optimization parameters and other quantities that can be associated with a part of an object.",
        "optional": [
            "type",
            "extract",
            "unit",
            "transformation",
            "volume_selection",
            "surface_selection",
            "curve_selection",
            "point_selection",
            "n_refs",
            "advanced",
            "enabled",
            "is_obstacle"
        ],
        "pointer": "/geometry/*",
        "required": [
            "mesh"
        ],
        "type": "object"
    },
    {
        "doc": "Each geometry object stores a mesh, a set of transformations applied to it after loading, and a set of selections, which can be used to specify boundary conditions, materials, optimization parameters and other quantities that can be associated with a part of an object.",
        "optional": [
            "type",
            "extract",
            "unit",
            "transformation",
            "volume_selection",
            "surface_selection",
            "curve_selection",
            "point_selection",
            "n_refs",
            "advanced",
            "enabled",
            "is_obstacle"
        ],
        "pointer": "/geometry/*",
        "required": [
            "mesh",
            "array"
        ],
        "type": "object",
        "type_name": "mesh_array"
    },
    {
        "doc": "Plane geometry object defined by its origin and normal.",
        "optional": [
            "type",
            "enabled",
            "is_obstacle"
        ],
        "pointer": "/geometry/*",
        "required": [
            "point",
            "normal"
        ],
        "type": "object",
        "type_name": "plane"
    },
    {
        "doc": "Plane orthogonal to gravity defined by its height.",
        "optional": [
            "type",
            "enabled",
            "is_obstacle"
        ],
        "pointer": "/geometry/*",
        "required": [
            "height"
        ],
        "type": "object",
        "type_name": "ground"
    },
    {
        "doc": "Mesh sequence.",
        "optional": [
            "type",
            "extract",
            "unit",
            "transformation",
            "n_refs",
            "advanced",
            "enabled",
            "is_obstacle"
        ],
        "pointer": "/geometry/*",
        "required": [
            "mesh_sequence",
            "fps"
        ],
        "type": "object",
        "type_name": "mesh_sequence"
    },
    {
        "doc": "Path of the mesh file to load.",
        "extensions": [
            ".obj",
            ".msh",
            ".stl",
            ".ply",
            ".mesh"
        ],
        "pointer": "/geometry/*/mesh",
        "type": "file"
    },
    {
        "default": "mesh",
        "doc": "Type of geometry, currently only one supported. In future we will add stuff like planes, spheres, etc.",
        "options": [
            "mesh",
            "plane",
            "ground",
            "mesh_sequence",
            "mesh_array"
        ],
        "pointer": "/geometry/*/type",
        "type": "string"
    },
    {
        "default": "volume",
        "doc": "Used to extract stuff from the mesh. Eg extract surface extracts the surface from a tet mesh.",
        "options": [
            "volume",
            "edges",
            "points",
            "surface"
        ],
        "pointer": "/geometry/*/extract",
        "type": "string"
    },
    {
        "default": "",
        "doc": "Units of the geometric model.",
        "pointer": "/geometry/*/unit",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Geometric transformations applied to the geometry after loading it.",
        "optional": [
            "translation",
            "rotation",
            "rotation_mode",
            "scale",
            "dimensions"
        ],
        "pointer": "/geometry/*/transformation",
        "type": "object"
    },
    {
        "default": "xyz")JSE_JSON";
        text += R"JSE_JSON(,
        "doc": "Type of rotation, supported are any permutation of [xyz]+, axis_angle, quaternion, or rotation_vector.",
        "pointer": "/geometry/*/transformation/rotation_mode",
        "type": "string"
    },
    {
        "default": [],
        "doc": "Translate (two entries for 2D problems or three entries for 3D problems).",
        "pointer": "/geometry/*/transformation/translation",
        "type": "list"
    },
    {
        "default": [],
        "doc": "Rotate, in 2D, one number, the rotation angle, in 3D, three or four Euler angles, axis+angle, or a unit quaternion. Depends on rotation mode.",
        "pointer": "/geometry/*/transformation/rotation",
        "type": "list"
    },
    {
        "default": [],
        "doc": "Scale by specified factors along axes (two entries for 2D problems or three entries for 3D problems).",
        "pointer": "/geometry/*/transformation/scale",
        "type": "list"
    },
    {
        "default": 1,
        "doc": "Scale the object so that bounding box dimensions match specified dimensions, 2 entries for 2D problems, 3 entries for 3D problems.",
        "pointer": "/geometry/*/transformation/dimensions",
        "type": "float"
    },
    {
        "doc": "Scale the object so that bounding box dimensions match specified dimensions, 2 entries for 2D problems, 3 entries for 3D problems.",
        "pointer": "/geometry/*/transformation/dimensions",
        "type": "list"
    },
    {
        "default": 0,
        "pointer": "/geometry/*/transformation/dimensions/*",
        "type": "float"
    },
    {
        "default": 0,
        "pointer": "/geometry/*/transformation/translation/*",
        "type": "float"
    },
    {
        "default": 0,
        "pointer": "/geometry/*/transformation/rotation/*",
        "type": "float"
    },
    {
        "default": 0,
        "pointer": "/geometry/*/transformation/scale/*",
        "type": "float"
    },
    {
        "doc": "Assign specified ID to all elements of the geometry.",
        "pointer": "/geometry/*/volume_selection",
        "type": "int"
    },
    {
        "doc": "Load ids from a file; the file is required to have one ID per element of the geometry",
        "extensions": [
            ".txt"
        ],
        "pointer": "/geometry/*/volume_selection",
        "type": "file"
    },
    {
        "doc": "Threshold for box side selection.",
        "pointer": "/geometry/*/volume_selection/*/threshold",
        "type": "float"
    },
    {
        "#type_name": "file",
        "default": null,
        "doc": "Load ids from a file; the file is required to have one ID per element of the geometry",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/volume_selection/*",
        "required": [
            "file"
        ],
        "type": "object"
    },
    {
        "#type_name": "box",
        "default": null,
        "doc": "Assign the ID to all elements with barycenters inside an axis-aligned box given by the list of its 2 corners, one with min, the other with max coordinates along all axes.  If relative option is set to true, the coordinates of the box corners are specified in bilinear/trilinear coordinates  with respect to the bounding box of the geometry.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/volume_selection/*",
        "required": [
            "id",
            "box"
        ],
        "type": "object"
    },
    {
        "#type_name": "sphere",
        "default": null,
        "doc": "Assign the ID to all elements with barycenters inside a sphere with specified center and radius.  If relative option is set to true, the coordinates of the  center are specified in bilinear/trilinear coordinates with respect to the bounding box of the geometry, and the radius is specified relative to the bounding box diagonal length.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/volume_selection/*",
        "required": [
            "id",
            "radius",
            "center"
        ],
        "type": "object"
    },
    {
        "#type_name": "cylinder",
        "default": null,
        "doc": "Assign the ID to all elements with barycenters inside a cylinder with specified axis (p1, p2) and radius.  If relative option is set to true, the coordinates of the  center are specified in bilinear/trilinear coordinates with respect to the bounding box of the geometry, and the radius is specified relative to the bounding box diagonal length.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/volume_selection/*",
        "required": [
            "id",
            "radius",
            "p1",
            "p2"
        ],
        "type": "object"
    },
    {
        "#type_name": "plane",
        "default": null,
        "doc": "Assign the ID to all elements with barycenters in a halfspace. The halfspace boundary plane is defined by a point in the plane and the normal, which points to the halfspace. The option relative set to true indicates that the point position is specified in bilinear/trilinear coordinates with respect to the bounding box of the geometry.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/volume_selection/*",
        "required": [
            "id",
            "point",
            "normal"
        ],
        "type": "object"
    },
    {
        "#type_name": "axis",
        "default": null,
        "doc": "Same as halfspace, but the boundary plane is axis-aligned. The choice of axis is specified either by a string matching the regexp r\"[+-][xyzXYZ]\" or an int matching the regular expression [+-]?[123] where the sign is the side of the plane to select and letter or number indicates the axis to which the plane is perpendicular. The offset is the plane offset from the origin. If the relative option is set to true, the offset is with respect to the center of the bounding box.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/volume_selection/*",
        "required": [
            "id",
            "axis",
            "position"
        ],
        "type": "object"
    },
    {
        "default": true,
        "doc": "If true, only elements with at least a face on the geometry boundary are selected.",
        "pointer": "/geometry/*/volume_selection/*/boundary_only",
        "type": "bool"
    },
    {
        "doc": "Load ids from a file; the file is required to have one ID per element of the geometry",
        "extensions": [
            ".txt"
        ],
        "pointer": "/geometry/*/volume_selection/*/file",
        "type": "file"
    },
    {
        "pointer": "/geometry/*/volume_selection/*/id",
        "type": "int"
    },
    {
        "pointer": "/geometry/*/volume_selection/*/radius",
        "type": "float"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/volume_selection/*/center",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/volume_selection/*/center/*",
        "type": "float"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/volume_selection/*/p1",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/volume_selection/*/p1/*",
        "type": "float"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/volume_selection/*/p2",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/volume_selection/*/p2/*",
        "type": "float"
    },
    {
        "pointer": "/geometry/*/volume_selection/*/axis",
        "type": "int"
    },
    {
        "pointer": "/geometry/*/volume_selection/*/axis",
        "type": "string"
    },
    {
        "pointer": "/geometry/*/volume_se)JSE_JSON";
        text += R"JSE_JSON(lection/*/offset",
        "type": "float"
    },
    {
        "pointer": "/geometry/*/volume_selection/*/position",
        "type": "float"
    },
    {
        "default": false,
        "pointer": "/geometry/*/volume_selection/*/relative",
        "type": "bool"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/volume_selection/*/point",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/volume_selection/*/point/*",
        "type": "float"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/volume_selection/*/normal",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/volume_selection/*/normal/*",
        "type": "float"
    },
    {
        "max": 2,
        "min": 2,
        "pointer": "/geometry/*/volume_selection/*/box",
        "type": "list"
    },
    {
        "default": [],
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/volume_selection/*/box/*",
        "type": "list"
    },
    {
        "default": 0,
        "pointer": "/geometry/*/volume_selection/*/box/*/*",
        "type": "float"
    },
    {
        "#type_name": "id_offset",
        "default": null,
        "doc": "Offsets the volume IDs loaded from the mesh.",
        "optional": [
            "id_offset"
        ],
        "pointer": "/geometry/*/volume_selection",
        "type": "object"
    },
    {
        "default": 0,
        "doc": "Offsets the volume IDs loaded from the mesh.",
        "pointer": "/geometry/*/volume_selection/id_offset",
        "type": "int"
    },
    {
        "doc": "List of selection (ID assignment) operations to apply to the geometry; operations can be box, sphere, etc.",
        "pointer": "/geometry/*/volume_selection",
        "type": "list"
    },
    {
        "doc": "Assign specified ID to all elements of the geometry.",
        "pointer": "/geometry/*/surface_selection",
        "type": "int"
    },
    {
        "doc": "Load ids from a file; the file is required to have one ID per element of the geometry",
        "extensions": [
            ".txt"
        ],
        "pointer": "/geometry/*/surface_selection",
        "type": "file"
    },
    {
        "doc": "Threshold for box side selection.",
        "pointer": "/geometry/*/surface_selection/*/threshold",
        "type": "float"
    },
    {
        "#type_name": "file",
        "default": null,
        "doc": "Load ids from a file; the file is required to have one ID per element of the geometry",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/surface_selection/*",
        "required": [
            "file"
        ],
        "type": "object"
    },
    {
        "#type_name": "box",
        "default": null,
        "doc": "Assign the ID to all elements with barycenters inside an axis-aligned box given by the list of its 2 corners, one with min, the other with max coordinates along all axes.  If relative option is set to true, the coordinates of the box corners are specified in bilinear/trilinear coordinates  with respect to the bounding box of the geometry.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/surface_selection/*",
        "required": [
            "id",
            "box"
        ],
        "type": "object"
    },
    {
        "#type_name": "sphere",
        "default": null,
        "doc": "Assign the ID to all elements with barycenters inside a sphere with specified center and radius.  If relative option is set to true, the coordinates of the  center are specified in bilinear/trilinear coordinates with respect to the bounding box of the geometry, and the radius is specified relative to the bounding box diagonal length.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/surface_selection/*",
        "required": [
            "id",
            "radius",
            "center"
        ],
        "type": "object"
    },
    {
        "#type_name": "cylinder",
        "default": null,
        "doc": "Assign the ID to all elements with barycenters inside a cylinder with specified axis (p1, p2) and radius.  If relative option is set to true, the coordinates of the  center are specified in bilinear/trilinear coordinates with respect to the bounding box of the geometry, and the radius is specified relative to the bounding box diagonal length.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/surface_selection/*",
        "required": [
            "id",
            "radius",
            "p1",
            "p2"
        ],
        "type": "object"
    },
    {
        "#type_name": "plane",
        "default": null,
        "doc": "Assign the ID to all elements with barycenters in a halfspace. The halfspace boundary plane is defined by a point in the plane and the normal, which points to the halfspace. The option relative set to true indicates that the point position is specified in bilinear/trilinear coordinates with respect to the bounding box of the geometry.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/surface_selection/*",
        "required": [
            "id",
            "point",
            "normal"
        ],
        "type": "object"
    },
    {
        "#type_name": "axis",
        "default": null,
        "doc": "Same as halfspace, but the boundary plane is axis-aligned. The choice of axis is specified either by a string matching the regexp r\"[+-][xyzXYZ]\" or an int matching the regular expression [+-]?[123] where the sign is the side of the plane to select and letter or number indicates the axis to which the plane is perpendicular. The offset is the plane offset from the origin. If the relative option is set to true, the offset is with respect to the center of the bounding box.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/surface_selection/*",
        "required": [
            "id",
            "axis",
            "position"
        ],
        "type": "object"
    },
    {
        "default": true,
        "doc": "If true, only elements with at least a face on the geometry boundary are selected.",
        "pointer": "/geometry/*/surface_selection/*/boundary_only",
        "type": "bool"
    },
    {
        "doc": "Load ids from a file; the file is required to have one ID per element of the geometry",
        "extensions": [
            ".txt"
        ],
        "pointer": "/geometry/*/surface_selection/*/file",
        "type": "file"
    },
    {
        "pointer": "/geometry/*/surface_selection/*/id",
        "type": "int"
    },
    {
        "pointer": "/geometry/*/surface_selection/*/radius",
        "type": "float"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/surface_selection/*/center",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/surface_selection/*/center/*",
        "type": "float"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/surface_selection/*/p1",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/surface_selection/*/p1/*",
        "type": "float"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/surface_selection/*/p2",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/surface_selection/*/p2/*",
        "type": "float"
    },
    {
        "pointer": "/geometry/*/surface_selection/*/axis",
        "type": "int"
    },
    {
        "pointer": "/geometry/*/surface_selection/*/axis",
        "type": "string"
    },
    {
        "pointer": "/geometry/*/surface_selection/*/offset",
        "type": "float"
    },
    {
        "pointer": "/geometry/*/surface_selection/*/posi)JSE_JSON";
        text += R"JSE_JSON(tion",
        "type": "float"
    },
    {
        "default": false,
        "pointer": "/geometry/*/surface_selection/*/relative",
        "type": "bool"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/surface_selection/*/point",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/surface_selection/*/point/*",
        "type": "float"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/surface_selection/*/normal",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/surface_selection/*/normal/*",
        "type": "float"
    },
    {
        "max": 2,
        "min": 2,
        "pointer": "/geometry/*/surface_selection/*/box",
        "type": "list"
    },
    {
        "default": [],
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/surface_selection/*/box/*",
        "type": "list"
    },
    {
        "default": 0,
        "pointer": "/geometry/*/surface_selection/*/box/*/*",
        "type": "float"
    },
    {
        "default": "skip",
        "doc": "List of selection (ID assignment) operations to apply to the geometry; operations can be box, sphere, etc.",
        "pointer": "/geometry/*/surface_selection",
        "type": "list"
    },
    {
        "#type_name": "box_side",
        "default": null,
        "doc": "Assigns ids to sides touching the bbox of the model using a threshold. Assigns 1+offset to left, 2+offset to bottom, 3+offset to right, 4+offset to top, 5+offset to front, 6+offset to back, 7+offset to everything else.",
        "optional": [
            "id_offset"
        ],
        "pointer": "/geometry/*/surface_selection/*",
        "required": [
            "threshold"
        ],
        "type": "object"
    },
    {
        "default": 0,
        "doc": "ID offset of box side selection.",
        "pointer": "/geometry/*/point_selection/*/id_offset",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Selection of curves",
        "pointer": "/geometry/*/curve_selection",
        "type": "object"
    },
    {
        "doc": "Assign specified ID to all elements of the geometry.",
        "pointer": "/geometry/*/point_selection",
        "type": "int"
    },
    {
        "doc": "Load ids from a file; the file is required to have one ID per element of the geometry",
        "extensions": [
            ".txt"
        ],
        "pointer": "/geometry/*/point_selection",
        "type": "file"
    },
    {
        "doc": "Threshold for box side selection.",
        "pointer": "/geometry/*/point_selection/*/threshold",
        "type": "float"
    },
    {
        "#type_name": "file",
        "default": null,
        "doc": "Load ids from a file; the file is required to have one ID per element of the geometry",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/point_selection/*",
        "required": [
            "file"
        ],
        "type": "object"
    },
    {
        "#type_name": "box",
        "default": null,
        "doc": "Assign the ID to all elements with barycenters inside an axis-aligned box given by the list of its 2 corners, one with min, the other with max coordinates along all axes.  If relative option is set to true, the coordinates of the box corners are specified in bilinear/trilinear coordinates  with respect to the bounding box of the geometry.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/point_selection/*",
        "required": [
            "id",
            "box"
        ],
        "type": "object"
    },
    {
        "#type_name": "sphere",
        "default": null,
        "doc": "Assign the ID to all elements with barycenters inside a sphere with specified center and radius.  If relative option is set to true, the coordinates of the  center are specified in bilinear/trilinear coordinates with respect to the bounding box of the geometry, and the radius is specified relative to the bounding box diagonal length.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/point_selection/*",
        "required": [
            "id",
            "radius",
            "center"
        ],
        "type": "object"
    },
    {
        "#type_name": "cylinder",
        "default": null,
        "doc": "Assign the ID to all elements with barycenters inside a cylinder with specified axis (p1, p2) and radius.  If relative option is set to true, the coordinates of the  center are specified in bilinear/trilinear coordinates with respect to the bounding box of the geometry, and the radius is specified relative to the bounding box diagonal length.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/point_selection/*",
        "required": [
            "id",
            "radius",
            "p1",
            "p2"
        ],
        "type": "object"
    },
    {
        "#type_name": "plane",
        "default": null,
        "doc": "Assign the ID to all elements with barycenters in a halfspace. The halfspace boundary plane is defined by a point in the plane and the normal, which points to the halfspace. The option relative set to true indicates that the point position is specified in bilinear/trilinear coordinates with respect to the bounding box of the geometry.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/point_selection/*",
        "required": [
            "id",
            "point",
            "normal"
        ],
        "type": "object"
    },
    {
        "#type_name": "axis",
        "default": null,
        "doc": "Same as halfspace, but the boundary plane is axis-aligned. The choice of axis is specified either by a string matching the regexp r\"[+-][xyzXYZ]\" or an int matching the regular expression [+-]?[123] where the sign is the side of the plane to select and letter or number indicates the axis to which the plane is perpendicular. The offset is the plane offset from the origin. If the relative option is set to true, the offset is with respect to the center of the bounding box.",
        "optional": [
            "relative",
            "boundary_only"
        ],
        "pointer": "/geometry/*/point_selection/*",
        "required": [
            "id",
            "axis",
            "position"
        ],
        "type": "object"
    },
    {
        "default": true,
        "doc": "If true, only elements with at least a face on the geometry boundary are selected.",
        "pointer": "/geometry/*/point_selection/*/boundary_only",
        "type": "bool"
    },
    {
        "doc": "Load ids from a file; the file is required to have one ID per element of the geometry",
        "extensions": [
            ".txt"
        ],
        "pointer": "/geometry/*/point_selection/*/file",
        "type": "file"
    },
    {
        "pointer": "/geometry/*/point_selection/*/id",
        "type": "int"
    },
    {
        "pointer": "/geometry/*/point_selection/*/radius",
        "type": "float"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/point_selection/*/center",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/point_selection/*/center/*",
        "type": "float"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/point_selection/*/p1",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/point_selection/*/p1/*",
        "type": "float"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/point_selection/*/p2",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/point_selection/*/p2/*",
        "type": "float"
    },
    {
        "pointer": "/geometry/*/point_selection/*/axis",
        "type": "int"
    },)JSE_JSON";
        text += R"JSE_JSON(
    {
        "pointer": "/geometry/*/point_selection/*/axis",
        "type": "string"
    },
    {
        "pointer": "/geometry/*/point_selection/*/offset",
        "type": "float"
    },
    {
        "pointer": "/geometry/*/point_selection/*/position",
        "type": "float"
    },
    {
        "default": false,
        "pointer": "/geometry/*/point_selection/*/relative",
        "type": "bool"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/point_selection/*/point",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/point_selection/*/point/*",
        "type": "float"
    },
    {
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/point_selection/*/normal",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/point_selection/*/normal/*",
        "type": "float"
    },
    {
        "max": 2,
        "min": 2,
        "pointer": "/geometry/*/point_selection/*/box",
        "type": "list"
    },
    {
        "default": [],
        "max": 3,
        "min": 2,
        "pointer": "/geometry/*/point_selection/*/box/*",
        "type": "list"
    },
    {
        "default": 0,
        "pointer": "/geometry/*/point_selection/*/box/*/*",
        "type": "float"
    },
    {
        "default": "skip",
        "doc": "List of selection (ID assignment) operations to apply to the geometry; operations can be box, sphere, etc.",
        "pointer": "/geometry/*/point_selection",
        "type": "list"
    },
    {
        "#type_name": "box_side",
        "default": null,
        "doc": "Assigns ids to sides touching the bbox of the model using a threshold. Assigns 1+offset to left, 2+offset to bottom, 3+offset to right, 4+offset to top, 5+offset to front, 6+offset to back, 7+offset to everything else.",
        "optional": [
            "id_offset"
        ],
        "pointer": "/geometry/*/point_selection/*",
        "required": [
            "threshold"
        ],
        "type": "object"
    },
    {
        "default": 0,
        "doc": "ID offset of box side selection.",
        "pointer": "/geometry/*/surface_selection/*/id_offset",
        "type": "int"
    },
    {
        "default": 0,
        "doc": "number of uniform refinements",
        "pointer": "/geometry/*/n_refs",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Advanced options for geometry",
        "optional": [
            "normalize_mesh",
            "force_linear_geometry",
            "refinement_location",
            "min_component"
        ],
        "pointer": "/geometry/*/advanced",
        "type": "object"
    },
    {
        "default": false,
        "doc": "Rescale the mesh to it fits in the biunit cube",
        "pointer": "/geometry/*/advanced/normalize_mesh",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "Discard high-order nodes for curved geometries",
        "pointer": "/geometry/*/advanced/force_linear_geometry",
        "type": "bool"
    },
    {
        "default": 0.5,
        "doc": "parametric location of the refinement",
        "pointer": "/geometry/*/advanced/refinement_location",
        "type": "float"
    },
    {
        "default": -1,
        "doc": "Size of the minimum component for collision",
        "pointer": "/geometry/*/advanced/min_component",
        "type": "int"
    },
    {
        "default": false,
        "doc": "The geometry elements are not included in deforming geometry, only in collision computations",
        "pointer": "/geometry/*/is_obstacle",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "Skips the geometry if false",
        "pointer": "/geometry/*/enabled",
        "type": "bool"
    },
    {
        "doc": "Point on plane (two entries for 2D problems or three entries for 3D problems).",
        "pointer": "/geometry/*/point",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/point/*",
        "type": "float"
    },
    {
        "doc": "Normal of plane (two entries for 2D problems or three entries for 3D problems).",
        "pointer": "/geometry/*/normal",
        "type": "list"
    },
    {
        "pointer": "/geometry/*/normal/*",
        "type": "float"
    },
    {
        "doc": "Height of ground plane.",
        "pointer": "/geometry/*/height",
        "type": "float"
    },
    {
        "doc": "Directory (or GLOB) of meshes for the mesh sequence.",
        "pointer": "/geometry/*/mesh_sequence",
        "type": "string"
    },
    {
        "doc": "List of mesh files for the mesh sequence.",
        "pointer": "/geometry/*/mesh_sequence",
        "type": "list"
    },
    {
        "doc": "Path of the mesh file to load.",
        "extensions": [
            ".obj",
            ".msh",
            ".stl",
            ".ply",
            ".mesh"
        ],
        "pointer": "/geometry/*/mesh_sequence/*",
        "type": "file"
    },
    {
        "doc": "Frames of the mesh sequence per second.",
        "pointer": "/geometry/*/fps",
        "type": "int"
    },
    {
        "doc": "Array of meshes",
        "optional": [
            "relative"
        ],
        "pointer": "/geometry/*/array",
        "required": [
            "offset",
            "size"
        ],
        "type": "object"
    },
    {
        "default": false,
        "doc": "Is the offset value relative to the mesh's dimensions.",
        "pointer": "/geometry/*/array/relative",
        "type": "bool"
    },
    {
        "doc": "Offset of the mesh in the array.",
        "pointer": "/geometry/*/array/offset",
        "type": "float"
    },
    {
        "doc": "Size of the array (two entries for 2D problems or three entries for 3D problems).",
        "pointer": "/geometry/*/array/size",
        "type": "list"
    },
    {
        "doc": "Size of the array (two entries for 2D problems or three entries for 3D problems).",
        "min": 1,
        "pointer": "/geometry/*/array/size/*",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Options related to the FE space.",
        "optional": [
            "discr_order",
            "discr_orderq",
            "pressure_discr_order",
            "basis_type",
            "poly_basis_type",
            "use_p_ref",
            "remesh",
            "advanced"
        ],
        "pointer": "/space",
        "type": "object"
    },
    {
        "default": 1,
        "doc": "Lagrange element order at height dimension for the space for the main unknown, for prism.",
        "pointer": "/space/discr_orderq",
        "type": "int"
    },
    {
        "default": 1,
        "doc": "Lagrange element order for the space for the main unknown, for all elements.",
        "pointer": "/space/discr_order",
        "type": "int"
    },
    {
        "doc": "Path to file containing Lagrange element order for the space for the main unknown per element.",
        "extensions": [
            ".txt",
            ".bin"
        ],
        "pointer": "/space/discr_order",
        "type": "file"
    },
    {
        "doc": "List of Lagrange element order for the space for the main unknown with volume IDs.",
        "pointer": "/space/discr_order",
        "type": "list"
    },
    {
        "doc": "Lagrange element order for the a space tagged with volume ID for the main unknown.",
        "pointer": "/space/discr_order/*",
        "required": [
            "id",
            "order"
        ],
        "type": "object"
    },
    {
        "doc": "Volume selection ID to apply the discr_order to.",
        "pointer": "/space/discr_order/*/id",
        "type": "int"
    },
    {
        "doc": "List of volume selection IDs to apply the discr_order to.",
        "pointer": "/space/discr_order/*/id",
        "type": "list"
    },
    {
        "doc": "Volume selection ID to apply the discr_order to.",
        "pointer": "/space/discr_order/*/id/*",
        "type": "int"
    },
    {
        "doc": "Lagrange element o)JSE_JSON";
        text += R"JSE_JSON(rder for the space for the main unknown, for all elements.",
        "pointer": "/space/discr_order/*/order",
        "type": "int"
    },
    {
        "default": 1,
        "doc": " Lagrange element order for the space for the pressure unknown, for all elements.",
        "pointer": "/space/pressure_discr_order",
        "type": "int"
    },
    {
        "default": "Lagrange",
        "doc": "Type of basis to use for non polygonal element, one of Lagrange, Spline, or Serendipity. Spline or Serendipity work only for quad/hex meshes",
        "options": [
            "Lagrange",
            "Spline",
            "Serendipity",
            "Bernstein"
        ],
        "pointer": "/space/basis_type",
        "type": "string"
    },
    {
        "default": "MFSHarmonic",
        "doc": "Type of basis to use for a polygonal element, one of MFSHarmonic, MeanValue, or Wachspress see 'PolySpline..' paper for details.",
        "options": [
            "MFSHarmonic",
            "MeanValue",
            "Wachspress"
        ],
        "pointer": "/space/poly_basis_type",
        "type": "string"
    },
    {
        "default": false,
        "doc": "Perform a priori p-refinement based on element shape, as described in 'Decoupling..' paper.",
        "pointer": "/space/use_p_ref",
        "type": "bool"
    },
    {
        "default": null,
        "doc": "Settings for adaptive remeshing",
        "optional": [
            "enabled",
            "split",
            "collapse",
            "swap",
            "smooth",
            "local_relaxation",
            "type"
        ],
        "pointer": "/space/remesh",
        "type": "object"
    },
    {
        "default": false,
        "doc": "Whether to do adaptive remeshing",
        "pointer": "/space/remesh/enabled",
        "type": "bool"
    },
    {
        "default": null,
        "doc": "Settings for adaptive remeshing edge splitting operations",
        "optional": [
            "enabled",
            "acceptance_tolerance",
            "culling_threshold",
            "max_depth",
            "min_edge_length"
        ],
        "pointer": "/space/remesh/split",
        "type": "object"
    },
    {
        "default": true,
        "doc": "Whether to do edge splitting in adaptive remeshing",
        "pointer": "/space/remesh/split/enabled",
        "type": "bool"
    },
    {
        "default": 0.001,
        "doc": "Accept split operation if energy decreased by at least x",
        "min": 0,
        "pointer": "/space/remesh/split/acceptance_tolerance",
        "type": "float"
    },
    {
        "default": 0.95,
        "doc": "Split operation culling threshold on energy",
        "max": 1,
        "min": 0,
        "pointer": "/space/remesh/split/culling_threshold",
        "type": "float"
    },
    {
        "default": 3,
        "doc": "Maximum depth split per time-step",
        "min": 1,
        "pointer": "/space/remesh/split/max_depth",
        "type": "int"
    },
    {
        "default": 1e-06,
        "doc": "Minimum edge length to split",
        "min": 0,
        "pointer": "/space/remesh/split/min_edge_length",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Settings for adaptive remeshing edge collapse operations",
        "optional": [
            "enabled",
            "acceptance_tolerance",
            "culling_threshold",
            "max_depth",
            "rel_max_edge_length",
            "abs_max_edge_length"
        ],
        "pointer": "/space/remesh/collapse",
        "type": "object"
    },
    {
        "default": true,
        "doc": "Whether to do edge collapse in adaptive remeshing",
        "pointer": "/space/remesh/collapse/enabled",
        "type": "bool"
    },
    {
        "default": -1e-08,
        "doc": "Accept collapse operation if energy decreased by at least x",
        "max": 0,
        "pointer": "/space/remesh/collapse/acceptance_tolerance",
        "type": "float"
    },
    {
        "default": 0.01,
        "doc": "Collapse operation culling threshold on energy",
        "max": 1,
        "min": 0,
        "pointer": "/space/remesh/collapse/culling_threshold",
        "type": "float"
    },
    {
        "default": 3,
        "doc": "Maximum depth collapse per time-step",
        "min": 1,
        "pointer": "/space/remesh/collapse/max_depth",
        "type": "int"
    },
    {
        "default": 1,
        "doc": "Length of maximum edge length to collapse relative to initial minimum edge length",
        "min": 0,
        "pointer": "/space/remesh/collapse/rel_max_edge_length",
        "type": "float"
    },
    {
        "default": 1e+100,
        "doc": "Length of maximum edge length to collapse in absolute units of distance",
        "min": 0,
        "pointer": "/space/remesh/collapse/abs_max_edge_length",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Settings for adaptive remeshing edge/face swap operations",
        "optional": [
            "enabled",
            "acceptance_tolerance",
            "max_depth"
        ],
        "pointer": "/space/remesh/swap",
        "type": "object"
    },
    {
        "default": false,
        "doc": "Whether to do edge/face swap in adaptive remeshing",
        "pointer": "/space/remesh/swap/enabled",
        "type": "bool"
    },
    {
        "default": -1e-08,
        "doc": "Accept swap operation if energy decreased by at least x",
        "max": 0,
        "pointer": "/space/remesh/swap/acceptance_tolerance",
        "type": "float"
    },
    {
        "default": 3,
        "doc": "Maximum depth swap per time-step",
        "min": 1,
        "pointer": "/space/remesh/swap/max_depth",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Settings for adaptive remeshing vertex smoothing operations",
        "optional": [
            "enabled",
            "acceptance_tolerance",
            "max_iters"
        ],
        "pointer": "/space/remesh/smooth",
        "type": "object"
    },
    {
        "default": false,
        "doc": "Whether to do vertex smoothing in adaptive remeshing",
        "pointer": "/space/remesh/smooth/enabled",
        "type": "bool"
    },
    {
        "default": -1e-08,
        "doc": "Accept smooth operation if energy decreased by at least x",
        "max": 0,
        "pointer": "/space/remesh/smooth/acceptance_tolerance",
        "type": "float"
    },
    {
        "default": 1,
        "doc": "Maximum number of smoothing iterations per time-step",
        "min": 1,
        "pointer": "/space/remesh/smooth/max_iters",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Settings for adaptive remeshing local relaxation",
        "optional": [
            "local_mesh_n_ring",
            "local_mesh_rel_area",
            "max_nl_iterations"
        ],
        "pointer": "/space/remesh/local_relaxation",
        "type": "object"
    },
    {
        "default": 2,
        "doc": "Size of n-ring for local relaxation",
        "pointer": "/space/remesh/local_relaxation/local_mesh_n_ring",
        "type": "int"
    },
    {
        "default": 0.01,
        "doc": "Minimum area for local relaxation",
        "pointer": "/space/remesh/local_relaxation/local_mesh_rel_area",
        "type": "float"
    },
    {
        "default": 1,
        "doc": "Maximum number of nonlinear solver iterations before acceptance check",
        "pointer": "/space/remesh/local_relaxation/max_nl_iterations",
        "type": "int"
    },
    {
        "default": "physics",
        "doc": "Type of adaptive remeshing to use.",
        "options": [
            "physics",
            "sizing_field"
        ],
        "pointer": "/space/remesh/type",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Advanced settings for the FE space.",
        "optional": [
            "discr_order_max",
            "isoparametric",
            "bc_method",
            "n_)JSE_JSON";
        text += R"JSE_JSON(boundary_samples",
            "quadrature_order",
            "mass_quadrature_order",
            "use_corner_quadrature",
            "integral_constraints",
            "n_harmonic_samples",
            "force_no_ref_for_harmonic",
            "B",
            "h1_formula",
            "count_flipped_els",
            "count_flipped_els_continuous",
            "use_particle_advection"
        ],
        "pointer": "/space/advanced",
        "type": "object"
    },
    {
        "default": 4,
        "doc": "Maximal discretization order in adaptive p-refinement and hp-refinement",
        "pointer": "/space/advanced/discr_order_max",
        "type": "int"
    },
    {
        "default": false,
        "doc": "Forces geometric map basis to be the same degree as the main variable basis, irrespective of the degree associated with the geom. map degrees associated with the elements of the geometry.",
        "pointer": "/space/advanced/isoparametric",
        "type": "bool"
    },
    {
        "default": "sample",
        "doc": "Method for imposing analytic Dirichet boundary conditions. If 'lsq' (least-squares fit), then the bc function is sampled at quadrature points, and the FEspace nodal values on the boundary are determined by minimizing L2 norm of the difference. If 'sample', then the analytic bc function is sampled at the boundary nodes.",
        "options": [
            "lsq",
            "sample"
        ],
        "pointer": "/space/advanced/bc_method",
        "type": "string"
    },
    {
        "default": -1,
        "doc": "Per-element number of boundary samples for analytic Dirichlet and Neumann boundary conditions.",
        "pointer": "/space/advanced/n_boundary_samples",
        "type": "int"
    },
    {
        "default": -1,
        "doc": "Minimal quadrature order to use in matrix and rhs assembly; the actual order is determined as min(2*(p-1)+1,quadrature_order).",
        "pointer": "/space/advanced/quadrature_order",
        "type": "int"
    },
    {
        "default": false,
        "doc": "Use quadrature rules that always include all the vertices of the element.",
        "pointer": "/space/advanced/use_corner_quadrature",
        "type": "bool"
    },
    {
        "default": -1,
        "doc": "Minimal quadrature order to use in mass matrix assembler; the actual order is determined as min(2*p+1,quadrature_order)",
        "pointer": "/space/advanced/mass_quadrature_order",
        "type": "int"
    },
    {
        "default": 2,
        "doc": "Number of constraints for non-conforming polygonal basis;  0, 1, or 2; see 'PolySpline..' paper for details.",
        "pointer": "/space/advanced/integral_constraints",
        "type": "int"
    },
    {
        "default": 10,
        "doc": "If MFSHarmonics is used for a polygonal element, number of collocation samples used in the basis construction;see 'PolySpline..' paper for details.",
        "pointer": "/space/advanced/n_harmonic_samples",
        "type": "int"
    },
    {
        "default": false,
        "doc": "If true, do not do uniform global refinement if the mesh contains polygonal elements.",
        "pointer": "/space/advanced/force_no_ref_for_harmonic",
        "type": "bool"
    },
    {
        "default": 3,
        "doc": "The target deviation of the error on elements from perfect element error, for a priori geometry-dependent p-refinement, see 'Decoupling .. ' paper.",
        "pointer": "/space/advanced/B",
        "type": "int"
    },
    {
        "default": false,
        "doc": "",
        "pointer": "/space/advanced/h1_formula",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "Count the number of elements with Jacobian of the geometric map not positive at quadrature points.",
        "pointer": "/space/advanced/count_flipped_els",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "Count the number of elements with Jacobian of the geometric map not positive at any point.",
        "pointer": "/space/advanced/count_flipped_els_continuous",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "Use particle advection in splitting method for solving NS equation.",
        "pointer": "/space/advanced/use_particle_advection",
        "type": "bool"
    },
    {
        "default": "skip",
        "doc": "The time parameters: start time `t0`, end time `tend`, time step `dt`.",
        "optional": [
            "t0",
            "integrator",
            "quasistatic"
        ],
        "pointer": "/time",
        "required": [
            "tend",
            "dt"
        ],
        "type": "object"
    },
    {
        "doc": "The time parameters: start time `t0`, time step `dt`, number of time steps.",
        "optional": [
            "t0",
            "integrator",
            "quasistatic"
        ],
        "pointer": "/time",
        "required": [
            "time_steps",
            "dt"
        ],
        "type": "object"
    },
    {
        "doc": "The time parameters: start time `t0`, end time `tend`, number of time steps.",
        "optional": [
            "t0",
            "integrator",
            "quasistatic"
        ],
        "pointer": "/time",
        "required": [
            "time_steps",
            "tend"
        ],
        "type": "object"
    },
    {
        "default": 0,
        "doc": "Startning time",
        "min": 0,
        "pointer": "/time/t0",
        "type": "float"
    },
    {
        "doc": "Ending time",
        "min": 0,
        "pointer": "/time/tend",
        "type": "float"
    },
    {
        "doc": "Time step size $\\Delta t$",
        "min": 0,
        "pointer": "/time/dt",
        "type": "float"
    },
    {
        "doc": "Number of time steps",
        "min": 0,
        "pointer": "/time/time_steps",
        "type": "int"
    },
    {
        "default": "ImplicitEuler",
        "doc": "Time integrator",
        "options": [
            "ImplicitEuler",
            "BDF1",
            "BDF2",
            "BDF3",
            "BDF4",
            "BDF5",
            "BDF6",
            "ImplicitNewmark"
        ],
        "pointer": "/time/integrator",
        "type": "string"
    },
    {
        "doc": "Implicit Euler time integration",
        "pointer": "/time/integrator",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "ImplicitEuler"
    },
    {
        "doc": "Backwards differentiation formula time integration",
        "optional": [
            "steps"
        ],
        "pointer": "/time/integrator",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "BDF"
    },
    {
        "doc": "Implicit Newmark time integration",
        "optional": [
            "gamma",
            "beta"
        ],
        "pointer": "/time/integrator",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "ImplicitNewmark"
    },
    {
        "doc": "Type of time integrator to use",
        "options": [
            "ImplicitEuler",
            "BDF",
            "ImplicitNewmark"
        ],
        "pointer": "/time/integrator/type",
        "type": "string"
    },
    {
        "default": 0.5,
        "doc": "Newmark gamma",
        "max": 1,
        "min": 0,
        "pointer": "/time/integrator/gamma",
        "type": "float"
    },
    {
        "default": 0.25,
        "doc": "Newmark beta",
        "max": 0.5,
        "min": 0,
        "pointer": "/time/integrator/beta",
        "type": "float"
    },
    {
        "default": 1,
        "doc": "BDF order",
        "max": 6,
        "min": 1,
        "pointer": "/time/integrator/steps",
        "type": "int"
    },
    {
        "default": false,
        "doc": "Ignore inertia in time dependent. Used for doing incremental load.",
        "pointer": "/time/quasistatic",
        "type": "bool"
    },
    {
        "default": null,
        "doc": ")JSE_JSON";
        text += R"JSE_JSON(Contact handling parameters.",
        "optional": [
            "enabled",
            "dhat",
            "dhat_percentage",
            "epsv",
            "friction_coefficient",
            "use_convergent_formulation",
            "use_area_weighting",
            "use_improved_max_operator",
            "use_physical_barrier",
            "collision_mesh",
            "use_gcp_formulation",
            "alpha_n",
            "alpha_t",
            "min_distance_ratio",
            "use_adaptive_dhat",
            "periodic",
            "adhesion"
        ],
        "pointer": "/contact",
        "type": "object"
    },
    {
        "default": false,
        "doc": "True if adaptive epsilon is used.",
        "pointer": "/contact/use_adaptive_dhat",
        "type": "bool"
    },
    {
        "default": 0.5,
        "doc": "Ratio of the minimum distance to contact to define local epsilon.",
        "min": 0,
        "pointer": "/contact/min_distance_ratio",
        "type": "float"
    },
    {
        "default": false,
        "doc": "True if the smooth contact formulation is used.",
        "pointer": "/contact/use_gcp_formulation",
        "type": "bool"
    },
    {
        "default": 0.5,
        "doc": "Control the smoothness of tangent angle contraints of contact pairs.",
        "max": 1,
        "min": -1,
        "pointer": "/contact/alpha_t",
        "type": "float"
    },
    {
        "default": 0.5,
        "doc": "Control the smoothness of normal angle contraints of contact pairs.",
        "max": 1,
        "min": -1,
        "pointer": "/contact/alpha_n",
        "type": "float"
    },
    {
        "default": false,
        "doc": "True if contact handling is enabled.",
        "pointer": "/contact/enabled",
        "type": "bool"
    },
    {
        "default": 0.001,
        "doc": "Contact barrier activation distance.",
        "min": 0,
        "pointer": "/contact/dhat",
        "type": "float"
    },
    {
        "default": 0.8,
        "doc": "$\\hat{d}$ as percentage of the diagonal of the bounding box",
        "pointer": "/contact/dhat_percentage",
        "type": "float"
    },
    {
        "default": 0.001,
        "doc": "Friction smoothing parameter.",
        "min": 0,
        "pointer": "/contact/epsv",
        "type": "float"
    },
    {
        "default": 0,
        "doc": "Coefficient of friction (global)",
        "pointer": "/contact/friction_coefficient",
        "type": "float"
    },
    {
        "default": false,
        "doc": "Whether to use the convergent (area weighted) formulation of IPC.",
        "pointer": "/contact/use_convergent_formulation",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "If using the convergent formulation, whether or not to use area weighting. Currently not implemented.",
        "pointer": "/contact/use_area_weighting",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "If using the convergent formulation, whether or not to use improved max operator. Currently not implemented.",
        "pointer": "/contact/use_improved_max_operator",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "If using the convergent formulation, whether or not to use physical barrier stiffness. Currently not implemented.",
        "pointer": "/contact/use_physical_barrier",
        "type": "bool"
    },
    {
        "default": "skip",
        "doc": "Load a preconstructed collision mesh.",
        "optional": [
            "enabled"
        ],
        "pointer": "/contact/collision_mesh",
        "required": [
            "mesh",
            "linear_map"
        ],
        "type": "object"
    },
    {
        "doc": "Construct a collision mesh with a maximum edge length.",
        "optional": [
            "tessellation_type",
            "enabled"
        ],
        "pointer": "/contact/collision_mesh",
        "required": [
            "max_edge_length"
        ],
        "type": "object"
    },
    {
        "doc": "Construct a collision mesh.",
        "optional": [
            "enabled"
        ],
        "pointer": "/contact/collision_mesh",
        "type": "object"
    },
    {
        "doc": "Path to preconstructed collision mesh.",
        "pointer": "/contact/collision_mesh/mesh",
        "type": "string"
    },
    {
        "doc": "HDF file storing the linear mapping of displacements.",
        "pointer": "/contact/collision_mesh/linear_map",
        "type": "string"
    },
    {
        "doc": "Maximum edge length to use for building the collision mesh.",
        "pointer": "/contact/collision_mesh/max_edge_length",
        "type": "float"
    },
    {
        "default": "regular",
        "doc": "Type of tessellation to use for building the collision mesh.",
        "options": [
            "regular",
            "irregular"
        ],
        "pointer": "/contact/collision_mesh/tessellation_type",
        "type": "string"
    },
    {
        "default": true,
        "doc": "",
        "pointer": "/contact/collision_mesh/enabled",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "Set to true to check collision between adjacent periodic cells.",
        "pointer": "/contact/periodic",
        "type": "bool"
    },
    {
        "default": null,
        "doc": "Adhesion settings.",
        "optional": [
            "adhesion_enabled",
            "dhat_p",
            "dhat_a",
            "adhesion_strength",
            "tangential_adhesion_coefficient",
            "epsa"
        ],
        "pointer": "/contact/adhesion",
        "type": "object"
    },
    {
        "default": false,
        "doc": "Set to true to enable normal adhesion forces.",
        "pointer": "/contact/adhesion/adhesion_enabled",
        "type": "bool"
    },
    {
        "default": 0.001,
        "doc": "Distance at which normal adhesion force reaches its maximum.",
        "pointer": "/contact/adhesion/dhat_p",
        "type": "float"
    },
    {
        "default": 0.01,
        "doc": "Distance at which normal adhesion force is activated.",
        "pointer": "/contact/adhesion/dhat_a",
        "type": "float"
    },
    {
        "default": 0.001,
        "doc": "Parameter that sets the strength of the normal adhesion force.",
        "pointer": "/contact/adhesion/adhesion_strength",
        "type": "float"
    },
    {
        "default": 0.0,
        "doc": "Coefficient of tangential adhesion (global)",
        "pointer": "/contact/adhesion/tangential_adhesion_coefficient",
        "type": "float"
    },
    {
        "default": 0.001,
        "doc": "Tangential adhesion smoothing parameter.",
        "min": 0,
        "pointer": "/contact/adhesion/epsa",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Settings for the linear solver.",
        "optional": [
            "enable_overwrite_solver",
            "solver",
            "precond",
            "Eigen::LeastSquaresConjugateGradient",
            "Eigen::DGMRES",
            "Eigen::ConjugateGradient",
            "Eigen::BiCGSTAB",
            "Eigen::GMRES",
            "Eigen::MINRES",
            "Pardiso",
            "Hypre",
            "AMGCL"
        ],
        "pointer": "/solver/linear",
        "type": "object"
    },
    {
        "default": false,
        "doc": "If solver name is not present, falls back to default",
        "pointer": "/solver/linear/enable_overwrite_solver",
        "type": "bool"
    },
    {
        "default": "",
        "doc": "Linear solver type.",
        "options": [
            "Eigen::SimplicialLDLT",
            "Eigen::SparseLU",
            "Eigen::CholmodSupernodalLLT",
            "Eigen::UmfPackLU",
            "Eigen::SuperLU",
            "Eigen::PardisoLDLT",
            "Eigen::PardisoLLT",
            "Eigen::PardisoLU",
            "Pardiso",
            "Hypre",
            "AMGCL",
            "Eigen::LeastSquaresConjugateGradient",
           )JSE_JSON";
        text += R"JSE_JSON( "Eigen::DGMRES",
            "Eigen::ConjugateGradient",
            "Eigen::BiCGSTAB",
            "Eigen::GMRES",
            "Eigen::MINRES"
        ],
        "pointer": "/solver/linear/solver",
        "type": "string"
    },
    {
        "default": "",
        "doc": "Preconditioner used if using an iterative linear solver.",
        "options": [
            "Eigen::IdentityPreconditioner",
            "Eigen::DiagonalPreconditioner",
            "Eigen::IncompleteCholesky",
            "Eigen::LeastSquareDiagonalPreconditioner",
            "Eigen::IncompleteLUT"
        ],
        "pointer": "/solver/linear/precond",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's Least Squares Conjugate Gradient solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/linear/Eigen::LeastSquaresConjugateGradient",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's DGMRES solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/linear/Eigen::DGMRES",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's Conjugate Gradient solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/linear/Eigen::ConjugateGradient",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's BiCGSTAB solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/linear/Eigen::BiCGSTAB",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's GMRES solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/linear/Eigen::GMRES",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's MINRES solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/linear/Eigen::MINRES",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Pardiso solver.",
        "optional": [
            "mtype"
        ],
        "pointer": "/solver/linear/Pardiso",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Hypre solver.",
        "optional": [
            "max_iter",
            "pre_max_iter",
            "tolerance"
        ],
        "pointer": "/solver/linear/Hypre",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the AMGCL solver.",
        "optional": [
            "solver",
            "precond"
        ],
        "pointer": "/solver/linear/AMGCL",
        "type": "object"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/Eigen::LeastSquaresConjugateGradient/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/Eigen::LeastSquaresConjugateGradient/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/Eigen::DGMRES/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/Eigen::DGMRES/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/Eigen::ConjugateGradient/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/Eigen::ConjugateGradient/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/Eigen::BiCGSTAB/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/Eigen::BiCGSTAB/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/Eigen::GMRES/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/Eigen::GMRES/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/Eigen::MINRES/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/Eigen::MINRES/tolerance",
        "type": "float"
    },
    {
        "default": 11,
        "doc": "Matrix type.",
        "options": [
            1,
            2,
            -2,
            3,
            4,
            -4,
            6,
            11,
            13
        ],
        "pointer": "/solver/linear/Pardiso/mtype",
        "type": "int"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/Hypre/max_iter",
        "type": "int"
    },
    {
        "default": 1,
        "doc": "Maximum number of pre iterations.",
        "pointer": "/solver/linear/Hypre/pre_max_iter",
        "type": "int"
    },
    {
        "default": 1e-10,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/Hypre/tolerance",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Solver settings for the AMGCL.",
        "optional": [
            "tol",
            "maxiter",
            "type"
        ],
        "pointer": "/solver/linear/AMGCL/solver",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Preconditioner settings for the AMGCL.",
        "optional": [
            "relax",
            "class",
            "max_levels",
            "direct_coarse",
            "ncycle",
            "coarsening"
        ],
        "pointer": "/solver/linear/AMGCL/precond",
        "type": "object"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/AMGCL/solver/maxiter",
        "type": "int"
    },
    {
        "default": 1e-10,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/AMGCL/solver/tol",
        "type": "float"
    },
    {
        "default": "cg",
        "doc": "Type of solver to use.",
        "pointer": "/solver/linear/AMGCL/solver/type",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Preconditioner settings for the AMGCL.",
        "optional": [
            "degree",
            "type",
            "power_iters",
            "higher",
            "lower",
            "scale"
        ],
        "pointer": "/solver/linear/AMGCL/precond/relax",
        "type": "object"
    },
    {
        "default": "amg",
        "doc": "Type of preconditioner to use.",
        "pointer": "/solver/linear/AMGCL/precond/class",
        "type": "string"
    },
    {
        "default": 6,
        "doc": "Maximum number of levels.",
        "pointer": "/solver/linear/AMGCL/precond/max_levels",
        "type": "int"
    },
    {
        "default": false,
        "doc": "Use direct solver for the coarsest level.",
        "pointer": "/solver/linear/AMGCL/precond/direct_coarse",
        "type": "bool"
    },
    {
        "default": 2,
        "doc": "Number of cycles.",
        "pointer": "/solver/linear/AMGCL/precond/ncycle",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Coarsening parameters.",
 )JSE_JSON";
        text += R"JSE_JSON(       "optional": [
            "type",
            "estimate_spectral_radius",
            "relax",
            "aggr"
        ],
        "pointer": "/solver/linear/AMGCL/precond/coarsening",
        "type": "object"
    },
    {
        "default": 16,
        "doc": "Degree of the polynomial.",
        "pointer": "/solver/linear/AMGCL/precond/relax/degree",
        "type": "int"
    },
    {
        "default": "chebyshev",
        "doc": "Type of relaxation to use.",
        "pointer": "/solver/linear/AMGCL/precond/relax/type",
        "type": "string"
    },
    {
        "default": 100,
        "doc": "Number of power iterations.",
        "pointer": "/solver/linear/AMGCL/precond/relax/power_iters",
        "type": "int"
    },
    {
        "default": 2,
        "doc": "Higher level relaxation.",
        "pointer": "/solver/linear/AMGCL/precond/relax/higher",
        "type": "float"
    },
    {
        "default": 0.008333333333,
        "doc": "Lower level relaxation.",
        "pointer": "/solver/linear/AMGCL/precond/relax/lower",
        "type": "float"
    },
    {
        "default": true,
        "doc": "Scale.",
        "pointer": "/solver/linear/AMGCL/precond/relax/scale",
        "type": "bool"
    },
    {
        "default": "smoothed_aggregation",
        "doc": "Coarsening type.",
        "pointer": "/solver/linear/AMGCL/precond/coarsening/type",
        "type": "string"
    },
    {
        "default": true,
        "doc": "Should the spectral radius be estimated.",
        "pointer": "/solver/linear/AMGCL/precond/coarsening/estimate_spectral_radius",
        "type": "bool"
    },
    {
        "default": 1,
        "doc": "Coarsening relaxation.",
        "pointer": "/solver/linear/AMGCL/precond/coarsening/relax",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Aggregation settings.",
        "optional": [
            "eps_strong"
        ],
        "pointer": "/solver/linear/AMGCL/precond/coarsening/aggr",
        "type": "object"
    },
    {
        "default": 0,
        "doc": "Aggregation epsilon strong.",
        "pointer": "/solver/linear/AMGCL/precond/coarsening/aggr/eps_strong",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Settings for the linear solver.",
        "optional": [
            "enable_overwrite_solver",
            "solver",
            "precond",
            "Eigen::LeastSquaresConjugateGradient",
            "Eigen::DGMRES",
            "Eigen::ConjugateGradient",
            "Eigen::BiCGSTAB",
            "Eigen::GMRES",
            "Eigen::MINRES",
            "Pardiso",
            "Hypre",
            "AMGCL"
        ],
        "pointer": "/solver/adjoint_linear",
        "type": "object"
    },
    {
        "default": false,
        "doc": "If solver name is not present, falls back to default",
        "pointer": "/solver/adjoint_linear/enable_overwrite_solver",
        "type": "bool"
    },
    {
        "default": "",
        "doc": "Linear solver type.",
        "options": [
            "Eigen::SimplicialLDLT",
            "Eigen::SparseLU",
            "Eigen::CholmodSupernodalLLT",
            "Eigen::UmfPackLU",
            "Eigen::SuperLU",
            "Eigen::PardisoLDLT",
            "Eigen::PardisoLLT",
            "Eigen::PardisoLU",
            "Pardiso",
            "Hypre",
            "AMGCL",
            "Eigen::LeastSquaresConjugateGradient",
            "Eigen::DGMRES",
            "Eigen::ConjugateGradient",
            "Eigen::BiCGSTAB",
            "Eigen::GMRES",
            "Eigen::MINRES"
        ],
        "pointer": "/solver/adjoint_linear/solver",
        "type": "string"
    },
    {
        "default": "",
        "doc": "Preconditioner used if using an iterative linear solver.",
        "options": [
            "Eigen::IdentityPreconditioner",
            "Eigen::DiagonalPreconditioner",
            "Eigen::IncompleteCholesky",
            "Eigen::LeastSquareDiagonalPreconditioner",
            "Eigen::IncompleteLUT"
        ],
        "pointer": "/solver/adjoint_linear/precond",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's Least Squares Conjugate Gradient solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/adjoint_linear/Eigen::LeastSquaresConjugateGradient",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's DGMRES solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/adjoint_linear/Eigen::DGMRES",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's Conjugate Gradient solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/adjoint_linear/Eigen::ConjugateGradient",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's BiCGSTAB solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/adjoint_linear/Eigen::BiCGSTAB",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's GMRES solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/adjoint_linear/Eigen::GMRES",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's MINRES solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/adjoint_linear/Eigen::MINRES",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Pardiso solver.",
        "optional": [
            "mtype"
        ],
        "pointer": "/solver/adjoint_linear/Pardiso",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Hypre solver.",
        "optional": [
            "max_iter",
            "pre_max_iter",
            "tolerance"
        ],
        "pointer": "/solver/adjoint_linear/Hypre",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the AMGCL solver.",
        "optional": [
            "solver",
            "precond"
        ],
        "pointer": "/solver/adjoint_linear/AMGCL",
        "type": "object"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/adjoint_linear/Eigen::LeastSquaresConjugateGradient/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/adjoint_linear/Eigen::LeastSquaresConjugateGradient/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/adjoint_linear/Eigen::DGMRES/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/adjoint_linear/Eigen::DGMRES/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/adjoint_linear/Eigen::ConjugateGradient/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/adjoint_linear/Eigen::ConjugateGradient/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/adjoint_linear/Eigen::BiCGSTAB/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/adjoint_linear/Eigen::BiCGSTAB/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
    )JSE_JSON";
        text += R"JSE_JSON(    "doc": "Maximum number of iterations.",
        "pointer": "/solver/adjoint_linear/Eigen::GMRES/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/adjoint_linear/Eigen::GMRES/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/adjoint_linear/Eigen::MINRES/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/adjoint_linear/Eigen::MINRES/tolerance",
        "type": "float"
    },
    {
        "default": 11,
        "doc": "Matrix type.",
        "options": [
            1,
            2,
            -2,
            3,
            4,
            -4,
            6,
            11,
            13
        ],
        "pointer": "/solver/adjoint_linear/Pardiso/mtype",
        "type": "int"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/adjoint_linear/Hypre/max_iter",
        "type": "int"
    },
    {
        "default": 1,
        "doc": "Maximum number of pre iterations.",
        "pointer": "/solver/adjoint_linear/Hypre/pre_max_iter",
        "type": "int"
    },
    {
        "default": 1e-10,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/adjoint_linear/Hypre/tolerance",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Solver settings for the AMGCL.",
        "optional": [
            "tol",
            "maxiter",
            "type"
        ],
        "pointer": "/solver/adjoint_linear/AMGCL/solver",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Preconditioner settings for the AMGCL.",
        "optional": [
            "relax",
            "class",
            "max_levels",
            "direct_coarse",
            "ncycle",
            "coarsening"
        ],
        "pointer": "/solver/adjoint_linear/AMGCL/precond",
        "type": "object"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/adjoint_linear/AMGCL/solver/maxiter",
        "type": "int"
    },
    {
        "default": 1e-10,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/adjoint_linear/AMGCL/solver/tol",
        "type": "float"
    },
    {
        "default": "cg",
        "doc": "Type of solver to use.",
        "pointer": "/solver/adjoint_linear/AMGCL/solver/type",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Preconditioner settings for the AMGCL.",
        "optional": [
            "degree",
            "type",
            "power_iters",
            "higher",
            "lower",
            "scale"
        ],
        "pointer": "/solver/adjoint_linear/AMGCL/precond/relax",
        "type": "object"
    },
    {
        "default": "amg",
        "doc": "Type of preconditioner to use.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/class",
        "type": "string"
    },
    {
        "default": 6,
        "doc": "Maximum number of levels.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/max_levels",
        "type": "int"
    },
    {
        "default": false,
        "doc": "Use direct solver for the coarsest level.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/direct_coarse",
        "type": "bool"
    },
    {
        "default": 2,
        "doc": "Number of cycles.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/ncycle",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Coarsening parameters.",
        "optional": [
            "type",
            "estimate_spectral_radius",
            "relax",
            "aggr"
        ],
        "pointer": "/solver/adjoint_linear/AMGCL/precond/coarsening",
        "type": "object"
    },
    {
        "default": 16,
        "doc": "Degree of the polynomial.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/relax/degree",
        "type": "int"
    },
    {
        "default": "chebyshev",
        "doc": "Type of relaxation to use.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/relax/type",
        "type": "string"
    },
    {
        "default": 100,
        "doc": "Number of power iterations.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/relax/power_iters",
        "type": "int"
    },
    {
        "default": 2,
        "doc": "Higher level relaxation.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/relax/higher",
        "type": "float"
    },
    {
        "default": 0.008333333333,
        "doc": "Lower level relaxation.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/relax/lower",
        "type": "float"
    },
    {
        "default": true,
        "doc": "Scale.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/relax/scale",
        "type": "bool"
    },
    {
        "default": "smoothed_aggregation",
        "doc": "Coarsening type.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/coarsening/type",
        "type": "string"
    },
    {
        "default": true,
        "doc": "Should the spectral radius be estimated.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/coarsening/estimate_spectral_radius",
        "type": "bool"
    },
    {
        "default": 1,
        "doc": "Coarsening relaxation.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/coarsening/relax",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Aggregation settings.",
        "optional": [
            "eps_strong"
        ],
        "pointer": "/solver/adjoint_linear/AMGCL/precond/coarsening/aggr",
        "type": "object"
    },
    {
        "default": 0,
        "doc": "Aggregation epsilon strong.",
        "pointer": "/solver/adjoint_linear/AMGCL/precond/coarsening/aggr/eps_strong",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Settings for nonlinear solver. Interior-loop linear solver settings are defined in the solver/linear section.",
        "optional": [
            "solver",
            "x_delta_tol",
            "grad_norm_tol",
            "rel_grad_norm_tol",
            "newton_decrement_tol",
            "rel_x_delta_tol",
            "first_grad_norm_tol",
            "norm_type",
            "max_iterations",
            "iterations_per_strategy",
            "line_search",
            "allow_out_of_iterations",
            "L-BFGS",
            "L-BFGS-B",
            "Newton",
            "ADAM",
            "StochasticADAM",
            "StochasticGradientDescent",
            "box_constraints",
            "advanced"
        ],
        "pointer": "/solver/nonlinear",
        "type": "object"
    },
    {
        "default": "Newton",
        "doc": "Nonlinear solver type",
        "options": [
            "Newton",
            "DenseNewton",
            "GradientDescent",
            "ADAM",
            "StochasticADAM",
            "StochasticGradientDescent",
            "L-BFGS",
            "BFGS",
            "L-BFGS-B",
            "MMA"
        ],
        "pointer": "/solver/nonlinear/solver",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "Stopping criterion: minimal change of the variables x for the iterations to continue. Computed as the L2 norm of x divide by the time step.",
        "min": 0,
        "pointer": "/solver/nonlinear/x_delta_tol",
        "type": "float"
    },
    {
        "default": 0,
        "doc": "Stopping criterion: minimal change of the variables x for the iterations to continue relative to first step in nonlinear solve.",
        "min": 0,
        "pointer": "/solver/nonlinear/rel_x_delta_tol",
        "type": "float"
    },
    {
        "default": 1e-10,
        "doc": "Stopping criterion: minimal gradient for the iterations t)JSE_JSON";
        text += R"JSE_JSON(o continue relative to first step in nonlinear solve.",
        "min": 0,
        "pointer": "/solver/nonlinear/rel_grad_norm_tol",
        "type": "float"
    },
    {
        "default": 0,
        "doc": "Stopping criterion: minimal change of energy (as estimated by Newton decrement) for the iterations to continue.",
        "min": 0,
        "pointer": "/solver/nonlinear/newton_decrement_tol",
        "type": "float"
    },
    {
        "default": 1e-10,
        "doc": "Stopping criterion: Minimal gradient norm for the iterations to continue.",
        "min": 0,
        "pointer": "/solver/nonlinear/grad_norm_tol",
        "type": "float"
    },
    {
        "default": 1e-12,
        "doc": "Minimal gradient norm for the iterations to not start, assume we already are at a minimum.",
        "pointer": "/solver/nonlinear/first_grad_norm_tol",
        "type": "float"
    },
    {
        "default": "L2",
        "doc": "Norm to use when computing stopping criteria in nonlinear solve.",
        "options": [
            "Euclidean",
            "L2",
            "Linf"
        ],
        "pointer": "/solver/nonlinear/norm_type",
        "type": "string"
    },
    {
        "default": 500,
        "doc": "Maximum number of iterations for a nonlinear solve.",
        "pointer": "/solver/nonlinear/max_iterations",
        "type": "int"
    },
    {
        "default": 5,
        "doc": "Number of iterations for every substrategy before reset.",
        "pointer": "/solver/nonlinear/iterations_per_strategy",
        "type": "int"
    },
    {
        "doc": "Number of iterations for every substrategy before reset.",
        "pointer": "/solver/nonlinear/iterations_per_strategy",
        "type": "list"
    },
    {
        "default": 5,
        "doc": "Number of iterations for every substrategy before reset.",
        "pointer": "/solver/nonlinear/iterations_per_strategy/*",
        "type": "int"
    },
    {
        "default": false,
        "doc": "If false (default), an exception will be thrown when the nonlinear solver reaches the maximum number of iterations.",
        "pointer": "/solver/nonlinear/allow_out_of_iterations",
        "type": "bool"
    },
    {
        "default": null,
        "doc": "Options for LBFGS.",
        "optional": [
            "history_size"
        ],
        "pointer": "/solver/nonlinear/L-BFGS",
        "type": "object"
    },
    {
        "default": 6,
        "doc": "The number of corrections to approximate the inverse Hessian matrix.",
        "pointer": "/solver/nonlinear/L-BFGS/history_size",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Options for the boxed L-BFGS.",
        "optional": [
            "history_size"
        ],
        "pointer": "/solver/nonlinear/L-BFGS-B",
        "type": "object"
    },
    {
        "default": 6,
        "doc": "The number of corrections to approximate the inverse Hessian matrix.",
        "pointer": "/solver/nonlinear/L-BFGS-B/history_size",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Options for Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc",
            "force_psd_projection",
            "use_psd_projection",
            "use_psd_projection_in_regularized"
        ],
        "pointer": "/solver/nonlinear/Newton",
        "type": "object"
    },
    {
        "default": 1e-05,
        "doc": "Tolerance of the linear system residual. If residual is above, the direction is rejected.",
        "pointer": "/solver/nonlinear/Newton/residual_tolerance",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Minimum regulariztion weight.",
        "pointer": "/solver/nonlinear/Newton/reg_weight_min",
        "type": "float"
    },
    {
        "default": 100000000.0,
        "doc": "Maximum regulariztion weight.",
        "pointer": "/solver/nonlinear/Newton/reg_weight_max",
        "type": "float"
    },
    {
        "default": 10,
        "doc": "Regulariztion weight increment.",
        "pointer": "/solver/nonlinear/Newton/reg_weight_inc",
        "type": "float"
    },
    {
        "default": false,
        "doc": "Force the Hessian to be PSD when using second order solvers (i.e., Newton's method).",
        "pointer": "/solver/nonlinear/Newton/force_psd_projection",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "Use PSD as fallback using second order solvers (i.e., Newton's method).",
        "pointer": "/solver/nonlinear/Newton/use_psd_projection",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "Use PSD in regularized Newton.",
        "pointer": "/solver/nonlinear/Newton/use_psd_projection_in_regularized",
        "type": "bool"
    },
    {
        "default": null,
        "doc": "Options for ADAM.",
        "optional": [
            "alpha",
            "beta_1",
            "beta_2",
            "epsilon"
        ],
        "pointer": "/solver/nonlinear/ADAM",
        "type": "object"
    },
    {
        "default": 0.001,
        "doc": "Parameter alpha for ADAM.",
        "pointer": "/solver/nonlinear/ADAM/alpha",
        "type": "float"
    },
    {
        "default": 0.9,
        "doc": "Parameter beta_1 for ADAM.",
        "pointer": "/solver/nonlinear/ADAM/beta_1",
        "type": "float"
    },
    {
        "default": 0.999,
        "doc": "Parameter beta_2 for ADAM.",
        "pointer": "/solver/nonlinear/ADAM/beta_2",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Parameter epsilon for ADAM.",
        "pointer": "/solver/nonlinear/ADAM/epsilon",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Options for ADAM.",
        "optional": [
            "alpha",
            "beta_1",
            "beta_2",
            "epsilon",
            "erase_component_probability"
        ],
        "pointer": "/solver/nonlinear/StochasticADAM",
        "type": "object"
    },
    {
        "default": 0.001,
        "doc": "Parameter alpha for ADAM.",
        "pointer": "/solver/nonlinear/StochasticADAM/alpha",
        "type": "float"
    },
    {
        "default": 0.9,
        "doc": "Parameter beta_1 for ADAM.",
        "pointer": "/solver/nonlinear/StochasticADAM/beta_1",
        "type": "float"
    },
    {
        "default": 0.999,
        "doc": "Parameter beta_2 for ADAM.",
        "pointer": "/solver/nonlinear/StochasticADAM/beta_2",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Parameter epsilon for ADAM.",
        "pointer": "/solver/nonlinear/StochasticADAM/epsilon",
        "type": "float"
    },
    {
        "default": 0.3,
        "doc": "Probability of erasing a component on the gradient for ADAM.",
        "pointer": "/solver/nonlinear/StochasticADAM/erase_component_probability",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Options for Stochastic Gradient Descent.",
        "optional": [
            "erase_component_probability"
        ],
        "pointer": "/solver/nonlinear/StochasticGradientDescent",
        "type": "object"
    },
    {
        "default": 0.3,
        "doc": "Probability of erasing a component on the gradient for StochasticGradientDescent.",
        "pointer": "/solver/nonlinear/StochasticGradientDescent/erase_component_probability",
        "type": "float"
    },
    {
        "doc": "List of solvers for ballback. Eg, [{'type':'Newton'}, {'type':'L-BFGS'}, {'type':'GradientDescent'}] will solve using Newton, in case of failure will fallback to L-BFGS and eventually to GradientDescent",
        "pointer": "/solver/nonlinear/solver",
        "type": "list"
    },
    {
        "doc": "Options for Newton.",
        "optional": [
            "residual_tolerance"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
   )JSE_JSON";
        text += R"JSE_JSON(     ],
        "type": "object",
        "type_name": "Newton"
    },
    {
        "doc": "Options for projected Newton.",
        "optional": [
            "residual_tolerance"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "ProjectedNewton"
    },
    {
        "doc": "Options for regularized Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "RegularizedNewton"
    },
    {
        "doc": "Options for regularized projected Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "RegularizedProjectedNewton"
    },
    {
        "doc": "Options for Newton.",
        "optional": [
            "residual_tolerance"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DenseNewton"
    },
    {
        "doc": "Options for projected Newton.",
        "optional": [
            "residual_tolerance"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DenseProjectedNewton"
    },
    {
        "doc": "Options for regularized Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DenseRegularizedNewton"
    },
    {
        "doc": "Options for projected regularized Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DenseRegularizedProjectedNewton"
    },
    {
        "doc": "Options for Gradient Descent.",
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "GradientDescent"
    },
    {
        "doc": "Options for Stochastic Gradient Descent.",
        "optional": [
            "erase_component_probability"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "StochasticGradientDescent"
    },
    {
        "doc": "Options for L-BFGS.",
        "optional": [
            "history_size"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "L-BFGS"
    },
    {
        "doc": "Options for BFGS.",
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "BFGS"
    },
    {
        "doc": "Options for ADAM.",
        "optional": [
            "alpha",
            "beta_1",
            "beta_2",
            "epsilon"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "ADAM"
    },
    {
        "doc": "Options for ADAM.",
        "optional": [
            "alpha",
            "beta_1",
            "beta_2",
            "epsilon",
            "erase_component_probability"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "StochasticADAM"
    },
    {
        "doc": "Nonlinear solver type",
        "options": [
            "Newton",
            "DenseNewton",
            "ProjectedNewton",
            "DenseProjectedNewton",
            "RegularizedNewton",
            "DenseRegularizedNewton",
            "RegularizedProjectedNewton",
            "DenseRegularizedProjectedNewton",
            "GradientDescent",
            "StochasticGradientDescent",
            "ADAM",
            "StochasticADAM",
            "L-BFGS",
            "BFGS"
        ],
        "pointer": "/solver/nonlinear/solver/*/type",
        "type": "string"
    },
    {
        "default": 1e-05,
        "doc": "Tolerance of the linear system residual. If residual is above, the direction is rejected.",
        "pointer": "/solver/nonlinear/solver/*/residual_tolerance",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Minimum regulariztion weight.",
        "pointer": "/solver/nonlinear/solver/*/reg_weight_min",
        "type": "float"
    },
    {
        "default": 100000000.0,
        "doc": "Maximum regulariztion weight.",
        "pointer": "/solver/nonlinear/solver/*/reg_weight_max",
        "type": "float"
    },
    {
        "default": 10,
        "doc": "Regulariztion weight increment.",
        "pointer": "/solver/nonlinear/solver/*/reg_weight_inc",
        "type": "float"
    },
    {
        "default": 0.3,
        "doc": "Probability of erasing a component on the gradient for stochastic solvers.",
        "pointer": "/solver/nonlinear/solver/*/erase_component_probability",
        "type": "float"
    },
    {
        "default": 6,
        "doc": "The number of corrections to approximate the inverse Hessian matrix.",
        "pointer": "/solver/nonlinear/solver/*/history_size",
        "type": "int"
    },
    {
        "default": 0.001,
        "doc": "Parameter alpha for ADAM.",
        "pointer": "/solver/nonlinear/solver/*/alpha",
        "type": "float"
    },
    {
        "default": 0.9,
        "doc": "Parameter beta_1 for ADAM.",
        "pointer": "/solver/nonlinear/solver/*/beta_1",
        "type": "float"
    },
    {
        "default": 0.999,
        "doc": "Parameter beta_2 for ADAM.",
        "pointer": "/solver/nonlinear/solver/*/beta_2",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Parameter epsilon for ADAM.",
        "pointer": "/solver/nonlinear/solver/*/epsilon",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Settings for line-search in the nonlinear solver",
        "optional": [
            "method",
            "use_grad_norm_tol",
            "min_step_size",
            "max_step_size_iter",
            "min_step_size_final",
            "max_step_size_iter_final",
            "default_init_step_size",
            "step_ratio",
            "Armijo",
            "RobustArmijo"
        ],
        "pointer": "/solver/nonlinear/line_search",
        "type": "object"
    },
    {
        "default": "RobustArmijo",
        "doc": "Line-search type",
        "options": [
            "Armijo",
            "RobustArmijo",
            "Backtracking",
            "None"
        ],
        "pointer": "/solver/nonlinear/line_search/method",
        "type": "string"
    },
    {
        "default": 1e-06,
        "doc": "When the energy is smaller than use_grad_norm_tol, line-search uses norm of gradient instead of energy",
        "pointer": "/solver/nonlinear/line_search/use_grad_norm_tol",
        "type": "float"
    },
    {
        "default": 1e-10,
        "doc": "Mimimum step size",
        "pointer": "/solver/nonlinear/line_search/min_step_size",
        "type": "float"
    },
    {
        "default": 30,
        "doc)JSE_JSON";
        text += R"JSE_JSON(": "Number of iterations",
        "pointer": "/solver/nonlinear/line_search/max_step_size_iter",
        "type": "int"
    },
    {
        "default": 1e-20,
        "doc": "Mimimum step size for last descent strategy",
        "pointer": "/solver/nonlinear/line_search/min_step_size_final",
        "type": "float"
    },
    {
        "default": 100,
        "doc": "Number of iterations for last descent strategy",
        "pointer": "/solver/nonlinear/line_search/max_step_size_iter_final",
        "type": "int"
    },
    {
        "default": 1,
        "doc": "Initial step size",
        "pointer": "/solver/nonlinear/line_search/default_init_step_size",
        "type": "float"
    },
    {
        "default": 0.5,
        "doc": "Ratio used to decrease the step",
        "pointer": "/solver/nonlinear/line_search/step_ratio",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Options for Armijo.",
        "optional": [
            "c"
        ],
        "pointer": "/solver/nonlinear/line_search/Armijo",
        "type": "object"
    },
    {
        "default": 0.0001,
        "doc": "Armijo c parameter.",
        "min_value": 0,
        "pointer": "/solver/nonlinear/line_search/Armijo/c",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Options for RobustArmijo.",
        "optional": [
            "delta_relative_tolerance"
        ],
        "pointer": "/solver/nonlinear/line_search/RobustArmijo",
        "type": "object"
    },
    {
        "default": 0.1,
        "doc": "Relative tolerance on E to switch to approximate.",
        "min_value": 0,
        "pointer": "/solver/nonlinear/line_search/RobustArmijo/delta_relative_tolerance",
        "type": "float"
    },
    {
        "default": null,
        "optional": [
            "bounds",
            "max_change"
        ],
        "pointer": "/solver/nonlinear/box_constraints",
        "type": "object"
    },
    {
        "default": [],
        "doc": "Box constraints on optimization variables.",
        "pointer": "/solver/nonlinear/box_constraints/bounds",
        "type": "list"
    },
    {
        "doc": "Box constraint values on optimization variables.",
        "pointer": "/solver/nonlinear/box_constraints/bounds/*",
        "type": "list"
    },
    {
        "doc": "Box constraint values on optimization variables.",
        "pointer": "/solver/nonlinear/box_constraints/bounds/*/*",
        "type": "float"
    },
    {
        "doc": "Box constraint values on optimization variables.",
        "pointer": "/solver/nonlinear/box_constraints/bounds/*",
        "type": "float"
    },
    {
        "default": -1,
        "doc": "Maximum change of optimization variables in one iteration, only for solvers with box constraints. Negative value to disable this constraint.",
        "pointer": "/solver/nonlinear/box_constraints/max_change",
        "type": "float"
    },
    {
        "doc": "Maximum change of optimization variables in one iteration, only for solvers with box constraints.",
        "pointer": "/solver/nonlinear/box_constraints/max_change",
        "type": "list"
    },
    {
        "doc": "Maximum change of every optimization variable in one iteration, only for solvers with box constraints.",
        "pointer": "/solver/nonlinear/box_constraints/max_change/*",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Nonlinear solver advanced options",
        "optional": [
            "f_delta_tol",
            "f_delta_step_tol",
            "derivative_along_delta_x_tol",
            "apply_gradient_fd",
            "gradient_fd_eps"
        ],
        "pointer": "/solver/nonlinear/advanced",
        "type": "object"
    },
    {
        "default": 0,
        "doc": "Dangerous Option: Quit the optimization if the solver reduces the energy by less than f_delta for consecutive f_delta_step_tol steps.",
        "min": 0,
        "pointer": "/solver/nonlinear/advanced/f_delta_tol",
        "type": "float"
    },
    {
        "default": 100,
        "doc": "Dangerous Option: Quit the optimization if the solver reduces the energy by less than f_delta for consecutive f_delta_step_tol steps.",
        "pointer": "/solver/nonlinear/advanced/f_delta_step_tol",
        "type": "int"
    },
    {
        "default": 0,
        "doc": "Quit the optimization if the directional derivative along the descent direction is smaller than this tolerance.",
        "min": 0,
        "pointer": "/solver/nonlinear/advanced/derivative_along_delta_x_tol",
        "type": "float"
    },
    {
        "default": "None",
        "doc": "Expensive Option: For every iteration of the nonlinear solver, run finite difference to verify gradient of energy.",
        "options": [
            "None",
            "DirectionalDerivative",
            "FullFiniteDiff"
        ],
        "pointer": "/solver/nonlinear/advanced/apply_gradient_fd",
        "type": "string"
    },
    {
        "default": 1e-07,
        "doc": "Expensive Option: Eps for finite difference to verify gradient of energy.",
        "pointer": "/solver/nonlinear/advanced/gradient_fd_eps",
        "type": "float"
    },
    {
        "default": null,
        "doc": "The settings for the solver including linear solver, nonlinear solver, and some advanced options.",
        "optional": [
            "max_threads",
            "linear",
            "adjoint_linear",
            "nonlinear",
            "augmented_lagrangian",
            "contact",
            "rayleigh_damping",
            "advanced"
        ],
        "pointer": "/solver",
        "type": "object"
    },
    {
        "default": 0,
        "doc": "Maximum number of threads used; 0 is unlimited.",
        "min": 0,
        "pointer": "/solver/max_threads",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Settings for the linear solver.",
        "optional": [
            "enable_overwrite_solver",
            "solver",
            "precond",
            "Eigen::LeastSquaresConjugateGradient",
            "Eigen::DGMRES",
            "Eigen::ConjugateGradient",
            "Eigen::BiCGSTAB",
            "Eigen::GMRES",
            "Eigen::MINRES",
            "Pardiso",
            "Hypre",
            "AMGCL"
        ],
        "pointer": "/solver/linear/adjoint_solver",
        "type": "object"
    },
    {
        "default": false,
        "doc": "If solver name is not present, falls back to default",
        "pointer": "/solver/linear/adjoint_solver/enable_overwrite_solver",
        "type": "bool"
    },
    {
        "default": "",
        "doc": "Linear solver type.",
        "options": [
            "Eigen::SimplicialLDLT",
            "Eigen::SparseLU",
            "Eigen::CholmodSupernodalLLT",
            "Eigen::UmfPackLU",
            "Eigen::SuperLU",
            "Eigen::PardisoLDLT",
            "Eigen::PardisoLLT",
            "Eigen::PardisoLU",
            "Pardiso",
            "Hypre",
            "AMGCL",
            "Eigen::LeastSquaresConjugateGradient",
            "Eigen::DGMRES",
            "Eigen::ConjugateGradient",
            "Eigen::BiCGSTAB",
            "Eigen::GMRES",
            "Eigen::MINRES"
        ],
        "pointer": "/solver/linear/adjoint_solver/solver",
        "type": "string"
    },
    {
        "default": "",
        "doc": "Preconditioner used if using an iterative linear solver.",
        "options": [
            "Eigen::IdentityPreconditioner",
            "Eigen::DiagonalPreconditioner",
            "Eigen::IncompleteCholesky",
            "Eigen::LeastSquareDiagonalPreconditioner",
            "Eigen::IncompleteLUT"
        ],
        "pointer": "/solver/linear/adjoint_solver/precond",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's Least Squares Conjugate Gradient solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": ")JSE_JSON";
        text += R"JSE_JSON(/solver/linear/adjoint_solver/Eigen::LeastSquaresConjugateGradient",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's DGMRES solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/linear/adjoint_solver/Eigen::DGMRES",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's Conjugate Gradient solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/linear/adjoint_solver/Eigen::ConjugateGradient",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's BiCGSTAB solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/linear/adjoint_solver/Eigen::BiCGSTAB",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's GMRES solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/linear/adjoint_solver/Eigen::GMRES",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Eigen's MINRES solver.",
        "optional": [
            "max_iter",
            "tolerance"
        ],
        "pointer": "/solver/linear/adjoint_solver/Eigen::MINRES",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Pardiso solver.",
        "optional": [
            "mtype"
        ],
        "pointer": "/solver/linear/adjoint_solver/Pardiso",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the Hypre solver.",
        "optional": [
            "max_iter",
            "pre_max_iter",
            "tolerance"
        ],
        "pointer": "/solver/linear/adjoint_solver/Hypre",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for the AMGCL solver.",
        "optional": [
            "solver",
            "precond"
        ],
        "pointer": "/solver/linear/adjoint_solver/AMGCL",
        "type": "object"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/adjoint_solver/Eigen::LeastSquaresConjugateGradient/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/adjoint_solver/Eigen::LeastSquaresConjugateGradient/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/adjoint_solver/Eigen::DGMRES/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/adjoint_solver/Eigen::DGMRES/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/adjoint_solver/Eigen::ConjugateGradient/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/adjoint_solver/Eigen::ConjugateGradient/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/adjoint_solver/Eigen::BiCGSTAB/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/adjoint_solver/Eigen::BiCGSTAB/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/adjoint_solver/Eigen::GMRES/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/adjoint_solver/Eigen::GMRES/tolerance",
        "type": "float"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/adjoint_solver/Eigen::MINRES/max_iter",
        "type": "int"
    },
    {
        "default": 1e-12,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/adjoint_solver/Eigen::MINRES/tolerance",
        "type": "float"
    },
    {
        "default": 11,
        "doc": "Matrix type.",
        "options": [
            1,
            2,
            -2,
            3,
            4,
            -4,
            6,
            11,
            13
        ],
        "pointer": "/solver/linear/adjoint_solver/Pardiso/mtype",
        "type": "int"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/adjoint_solver/Hypre/max_iter",
        "type": "int"
    },
    {
        "default": 1,
        "doc": "Maximum number of pre iterations.",
        "pointer": "/solver/linear/adjoint_solver/Hypre/pre_max_iter",
        "type": "int"
    },
    {
        "default": 1e-10,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/adjoint_solver/Hypre/tolerance",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Solver settings for the AMGCL.",
        "optional": [
            "tol",
            "maxiter",
            "type"
        ],
        "pointer": "/solver/linear/adjoint_solver/AMGCL/solver",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Preconditioner settings for the AMGCL.",
        "optional": [
            "relax",
            "class",
            "max_levels",
            "direct_coarse",
            "ncycle",
            "coarsening"
        ],
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond",
        "type": "object"
    },
    {
        "default": 1000,
        "doc": "Maximum number of iterations.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/solver/maxiter",
        "type": "int"
    },
    {
        "default": 1e-10,
        "doc": "Convergence tolerance.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/solver/tol",
        "type": "float"
    },
    {
        "default": "cg",
        "doc": "Type of solver to use.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/solver/type",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Preconditioner settings for the AMGCL.",
        "optional": [
            "degree",
            "type",
            "power_iters",
            "higher",
            "lower",
            "scale"
        ],
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/relax",
        "type": "object"
    },
    {
        "default": "amg",
        "doc": "Type of preconditioner to use.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/class",
        "type": "string"
    },
    {
        "default": 6,
        "doc": "Maximum number of levels.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/max_levels",
        "type": "int"
    },
    {
        "default": false,
        "doc": "Use direct solver for the coarsest level.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/direct_coarse",
        "type": "bool"
    },
    {
        "default": 2,
        "doc": "Number of cycles.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/ncycle",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Coarsening parameters.",
        "optional": [
            "type",
            "estimate_spectral_radius",
            "relax",
            "aggr"
        ],
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/coarsening",
        "type": "object"
    },
    {
        "default": 16,
        "doc": "Degree of the polynomial.",
        "pointer": "/solver/linear/adjoint_s)JSE_JSON";
        text += R"JSE_JSON(olver/AMGCL/precond/relax/degree",
        "type": "int"
    },
    {
        "default": "chebyshev",
        "doc": "Type of relaxation to use.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/relax/type",
        "type": "string"
    },
    {
        "default": 100,
        "doc": "Number of power iterations.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/relax/power_iters",
        "type": "int"
    },
    {
        "default": 2,
        "doc": "Higher level relaxation.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/relax/higher",
        "type": "float"
    },
    {
        "default": 0.008333333333,
        "doc": "Lower level relaxation.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/relax/lower",
        "type": "float"
    },
    {
        "default": true,
        "doc": "Scale.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/relax/scale",
        "type": "bool"
    },
    {
        "default": "smoothed_aggregation",
        "doc": "Coarsening type.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/coarsening/type",
        "type": "string"
    },
    {
        "default": true,
        "doc": "Should the spectral radius be estimated.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/coarsening/estimate_spectral_radius",
        "type": "bool"
    },
    {
        "default": 1,
        "doc": "Coarsening relaxation.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/coarsening/relax",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Aggregation settings.",
        "optional": [
            "eps_strong"
        ],
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/coarsening/aggr",
        "type": "object"
    },
    {
        "default": 0,
        "doc": "Aggregation epsilon strong.",
        "pointer": "/solver/linear/adjoint_solver/AMGCL/precond/coarsening/aggr/eps_strong",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Parameters for the AL for imposing Dirichlet BCs. If the bc are not imposable, we add $w\\|u - bc\\|^2$ to the energy ($u$ is the solution at the Dirichlet nodes and $bc$ are the Dirichlet values). After convergence, we try to impose bc again. The algorithm computes E + a/2*AL^2 - lambda AL, where E is the current energy (elastic, inertia, contact, etc.) and AL is the augmented Lagrangian energy. a starts at `initial_weight` and, in case DBC cannot be imposed, we update a as `a *= scaling` until `max_weight`. See IPC additional material",
        "optional": [
            "initial_weight",
            "scaling",
            "max_weight",
            "eta",
            "nonlinear"
        ],
        "pointer": "/solver/augmented_lagrangian",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Settings for nonlinear solver. Interior-loop linear solver settings are defined in the solver/linear section.",
        "optional": [
            "solver",
            "x_delta_tol",
            "grad_norm_tol",
            "rel_grad_norm_tol",
            "newton_decrement_tol",
            "rel_x_delta_tol",
            "first_grad_norm_tol",
            "norm_type",
            "max_iterations",
            "iterations_per_strategy",
            "line_search",
            "allow_out_of_iterations",
            "L-BFGS",
            "L-BFGS-B",
            "Newton",
            "ADAM",
            "StochasticADAM",
            "StochasticGradientDescent",
            "box_constraints",
            "advanced"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear",
        "type": "object"
    },
    {
        "default": "Newton",
        "doc": "Nonlinear solver type",
        "options": [
            "Newton",
            "DenseNewton",
            "GradientDescent",
            "ADAM",
            "StochasticADAM",
            "StochasticGradientDescent",
            "L-BFGS",
            "BFGS",
            "L-BFGS-B",
            "MMA"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "Stopping criterion: minimal change of the variables x for the iterations to continue. Computed as the L2 norm of x divide by the time step.",
        "min": 0,
        "pointer": "/solver/augmented_lagrangian/nonlinear/x_delta_tol",
        "type": "float"
    },
    {
        "default": 0,
        "doc": "Stopping criterion: minimal change of the variables x for the iterations to continue relative to first step in nonlinear solve.",
        "min": 0,
        "pointer": "/solver/augmented_lagrangian/nonlinear/rel_x_delta_tol",
        "type": "float"
    },
    {
        "default": 1e-10,
        "doc": "Stopping criterion: minimal gradient for the iterations to continue relative to first step in nonlinear solve.",
        "min": 0,
        "pointer": "/solver/augmented_lagrangian/nonlinear/rel_grad_norm_tol",
        "type": "float"
    },
    {
        "default": 0,
        "doc": "Stopping criterion: minimal change of energy (as estimated by Newton decrement) for the iterations to continue.",
        "min": 0,
        "pointer": "/solver/augmented_lagrangian/nonlinear/newton_decrement_tol",
        "type": "float"
    },
    {
        "default": 1e-10,
        "doc": "Stopping criterion: Minimal gradient norm for the iterations to continue.",
        "min": 0,
        "pointer": "/solver/augmented_lagrangian/nonlinear/grad_norm_tol",
        "type": "float"
    },
    {
        "default": 1e-12,
        "doc": "Minimal gradient norm for the iterations to not start, assume we already are at a minimum.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/first_grad_norm_tol",
        "type": "float"
    },
    {
        "default": "L2",
        "doc": "Norm to use when computing stopping criteria in nonlinear solve.",
        "options": [
            "Euclidean",
            "L2",
            "Linf"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/norm_type",
        "type": "string"
    },
    {
        "default": 500,
        "doc": "Maximum number of iterations for a nonlinear solve.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/max_iterations",
        "type": "int"
    },
    {
        "default": 5,
        "doc": "Number of iterations for every substrategy before reset.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/iterations_per_strategy",
        "type": "int"
    },
    {
        "doc": "Number of iterations for every substrategy before reset.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/iterations_per_strategy",
        "type": "list"
    },
    {
        "default": 5,
        "doc": "Number of iterations for every substrategy before reset.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/iterations_per_strategy/*",
        "type": "int"
    },
    {
        "default": false,
        "doc": "If false (default), an exception will be thrown when the nonlinear solver reaches the maximum number of iterations.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/allow_out_of_iterations",
        "type": "bool"
    },
    {
        "default": null,
        "doc": "Options for LBFGS.",
        "optional": [
            "history_size"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/L-BFGS",
        "type": "object"
    },
    {
        "default": 6,
        "doc": "The number of corrections to approximate the inverse Hessian matrix.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/L-BFGS/history_size",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Options for the boxed L-BFGS.",
        "optional": [
            "history_size"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/L-BFGS-B",
        "type": "object"
    },
 )JSE_JSON";
        text += R"JSE_JSON(   {
        "default": 6,
        "doc": "The number of corrections to approximate the inverse Hessian matrix.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/L-BFGS-B/history_size",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Options for Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc",
            "force_psd_projection",
            "use_psd_projection",
            "use_psd_projection_in_regularized"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/Newton",
        "type": "object"
    },
    {
        "default": 1e-05,
        "doc": "Tolerance of the linear system residual. If residual is above, the direction is rejected.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/Newton/residual_tolerance",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Minimum regulariztion weight.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/Newton/reg_weight_min",
        "type": "float"
    },
    {
        "default": 100000000.0,
        "doc": "Maximum regulariztion weight.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/Newton/reg_weight_max",
        "type": "float"
    },
    {
        "default": 10,
        "doc": "Regulariztion weight increment.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/Newton/reg_weight_inc",
        "type": "float"
    },
    {
        "default": false,
        "doc": "Force the Hessian to be PSD when using second order solvers (i.e., Newton's method).",
        "pointer": "/solver/augmented_lagrangian/nonlinear/Newton/force_psd_projection",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "Use PSD as fallback using second order solvers (i.e., Newton's method).",
        "pointer": "/solver/augmented_lagrangian/nonlinear/Newton/use_psd_projection",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "Use PSD in regularized Newton.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/Newton/use_psd_projection_in_regularized",
        "type": "bool"
    },
    {
        "default": null,
        "doc": "Options for ADAM.",
        "optional": [
            "alpha",
            "beta_1",
            "beta_2",
            "epsilon"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/ADAM",
        "type": "object"
    },
    {
        "default": 0.001,
        "doc": "Parameter alpha for ADAM.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/ADAM/alpha",
        "type": "float"
    },
    {
        "default": 0.9,
        "doc": "Parameter beta_1 for ADAM.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/ADAM/beta_1",
        "type": "float"
    },
    {
        "default": 0.999,
        "doc": "Parameter beta_2 for ADAM.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/ADAM/beta_2",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Parameter epsilon for ADAM.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/ADAM/epsilon",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Options for ADAM.",
        "optional": [
            "alpha",
            "beta_1",
            "beta_2",
            "epsilon",
            "erase_component_probability"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/StochasticADAM",
        "type": "object"
    },
    {
        "default": 0.001,
        "doc": "Parameter alpha for ADAM.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/StochasticADAM/alpha",
        "type": "float"
    },
    {
        "default": 0.9,
        "doc": "Parameter beta_1 for ADAM.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/StochasticADAM/beta_1",
        "type": "float"
    },
    {
        "default": 0.999,
        "doc": "Parameter beta_2 for ADAM.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/StochasticADAM/beta_2",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Parameter epsilon for ADAM.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/StochasticADAM/epsilon",
        "type": "float"
    },
    {
        "default": 0.3,
        "doc": "Probability of erasing a component on the gradient for ADAM.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/StochasticADAM/erase_component_probability",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Options for Stochastic Gradient Descent.",
        "optional": [
            "erase_component_probability"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/StochasticGradientDescent",
        "type": "object"
    },
    {
        "default": 0.3,
        "doc": "Probability of erasing a component on the gradient for StochasticGradientDescent.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/StochasticGradientDescent/erase_component_probability",
        "type": "float"
    },
    {
        "doc": "List of solvers for ballback. Eg, [{'type':'Newton'}, {'type':'L-BFGS'}, {'type':'GradientDescent'}] will solve using Newton, in case of failure will fallback to L-BFGS and eventually to GradientDescent",
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver",
        "type": "list"
    },
    {
        "doc": "Options for Newton.",
        "optional": [
            "residual_tolerance"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Newton"
    },
    {
        "doc": "Options for projected Newton.",
        "optional": [
            "residual_tolerance"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "ProjectedNewton"
    },
    {
        "doc": "Options for regularized Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "RegularizedNewton"
    },
    {
        "doc": "Options for regularized projected Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "RegularizedProjectedNewton"
    },
    {
        "doc": "Options for Newton.",
        "optional": [
            "residual_tolerance"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DenseNewton"
    },
    {
        "doc": "Options for projected Newton.",
        "optional": [
            "residual_tolerance"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DenseProjectedNewton"
    },
    {
        "doc": "Options for regularized Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DenseRegularizedNewton"
    },
    {
        "doc": "Options for projected regularized Newton.",
   )JSE_JSON";
        text += R"JSE_JSON(     "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DenseRegularizedProjectedNewton"
    },
    {
        "doc": "Options for Gradient Descent.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "GradientDescent"
    },
    {
        "doc": "Options for Stochastic Gradient Descent.",
        "optional": [
            "erase_component_probability"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "StochasticGradientDescent"
    },
    {
        "doc": "Options for L-BFGS.",
        "optional": [
            "history_size"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "L-BFGS"
    },
    {
        "doc": "Options for BFGS.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "BFGS"
    },
    {
        "doc": "Options for ADAM.",
        "optional": [
            "alpha",
            "beta_1",
            "beta_2",
            "epsilon"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "ADAM"
    },
    {
        "doc": "Options for ADAM.",
        "optional": [
            "alpha",
            "beta_1",
            "beta_2",
            "epsilon",
            "erase_component_probability"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "StochasticADAM"
    },
    {
        "doc": "Nonlinear solver type",
        "options": [
            "Newton",
            "DenseNewton",
            "ProjectedNewton",
            "DenseProjectedNewton",
            "RegularizedNewton",
            "DenseRegularizedNewton",
            "RegularizedProjectedNewton",
            "DenseRegularizedProjectedNewton",
            "GradientDescent",
            "StochasticGradientDescent",
            "ADAM",
            "StochasticADAM",
            "L-BFGS",
            "BFGS"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*/type",
        "type": "string"
    },
    {
        "default": 1e-05,
        "doc": "Tolerance of the linear system residual. If residual is above, the direction is rejected.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*/residual_tolerance",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Minimum regulariztion weight.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*/reg_weight_min",
        "type": "float"
    },
    {
        "default": 100000000.0,
        "doc": "Maximum regulariztion weight.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*/reg_weight_max",
        "type": "float"
    },
    {
        "default": 10,
        "doc": "Regulariztion weight increment.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*/reg_weight_inc",
        "type": "float"
    },
    {
        "default": 0.3,
        "doc": "Probability of erasing a component on the gradient for stochastic solvers.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*/erase_component_probability",
        "type": "float"
    },
    {
        "default": 6,
        "doc": "The number of corrections to approximate the inverse Hessian matrix.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*/history_size",
        "type": "int"
    },
    {
        "default": 0.001,
        "doc": "Parameter alpha for ADAM.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*/alpha",
        "type": "float"
    },
    {
        "default": 0.9,
        "doc": "Parameter beta_1 for ADAM.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*/beta_1",
        "type": "float"
    },
    {
        "default": 0.999,
        "doc": "Parameter beta_2 for ADAM.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*/beta_2",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Parameter epsilon for ADAM.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/solver/*/epsilon",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Settings for line-search in the nonlinear solver",
        "optional": [
            "method",
            "use_grad_norm_tol",
            "min_step_size",
            "max_step_size_iter",
            "min_step_size_final",
            "max_step_size_iter_final",
            "default_init_step_size",
            "step_ratio",
            "Armijo",
            "RobustArmijo"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/line_search",
        "type": "object"
    },
    {
        "default": "RobustArmijo",
        "doc": "Line-search type",
        "options": [
            "Armijo",
            "RobustArmijo",
            "Backtracking",
            "None"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/line_search/method",
        "type": "string"
    },
    {
        "default": 1e-06,
        "doc": "When the energy is smaller than use_grad_norm_tol, line-search uses norm of gradient instead of energy",
        "pointer": "/solver/augmented_lagrangian/nonlinear/line_search/use_grad_norm_tol",
        "type": "float"
    },
    {
        "default": 1e-10,
        "doc": "Mimimum step size",
        "pointer": "/solver/augmented_lagrangian/nonlinear/line_search/min_step_size",
        "type": "float"
    },
    {
        "default": 30,
        "doc": "Number of iterations",
        "pointer": "/solver/augmented_lagrangian/nonlinear/line_search/max_step_size_iter",
        "type": "int"
    },
    {
        "default": 1e-20,
        "doc": "Mimimum step size for last descent strategy",
        "pointer": "/solver/augmented_lagrangian/nonlinear/line_search/min_step_size_final",
        "type": "float"
    },
    {
        "default": 100,
        "doc": "Number of iterations for last descent strategy",
        "pointer": "/solver/augmented_lagrangian/nonlinear/line_search/max_step_size_iter_final",
        "type": "int"
    },
    {
        "default": 1,
        "doc": "Initial step size",
        "pointer": "/solver/augmented_lagrangian/nonlinear/line_search/default_init_step_size",
        "type": "float"
    },
    {
        "default": 0.5,
        "doc": "Ratio used to decrease the step",
        "pointer": "/solver/augmented_lagrangian/nonlinear/line_search/step_ratio",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Options for Armijo.",
        "optional": [
            "c"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/line_search/Armijo",
        "type": "object"
    },
    {
        "default": 0.0001,
        "doc": "Armijo c parameter.",
        "min_value": 0,
        "pointer": "/solver/augmented_lagrangian/nonlinear/line_search/Armijo/c",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Options for RobustArmijo.",
        "optional": [
            "delta_relative_tolerance"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/line_search/RobustArmijo",
        "type": "object"
    },
    {
        "default": 0.1,
        "doc": "Relative t)JSE_JSON";
        text += R"JSE_JSON(olerance on E to switch to approximate.",
        "min_value": 0,
        "pointer": "/solver/augmented_lagrangian/nonlinear/line_search/RobustArmijo/delta_relative_tolerance",
        "type": "float"
    },
    {
        "default": null,
        "optional": [
            "bounds",
            "max_change"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/box_constraints",
        "type": "object"
    },
    {
        "default": [],
        "doc": "Box constraints on optimization variables.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/box_constraints/bounds",
        "type": "list"
    },
    {
        "doc": "Box constraint values on optimization variables.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/box_constraints/bounds/*",
        "type": "list"
    },
    {
        "doc": "Box constraint values on optimization variables.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/box_constraints/bounds/*/*",
        "type": "float"
    },
    {
        "doc": "Box constraint values on optimization variables.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/box_constraints/bounds/*",
        "type": "float"
    },
    {
        "default": -1,
        "doc": "Maximum change of optimization variables in one iteration, only for solvers with box constraints. Negative value to disable this constraint.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/box_constraints/max_change",
        "type": "float"
    },
    {
        "doc": "Maximum change of optimization variables in one iteration, only for solvers with box constraints.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/box_constraints/max_change",
        "type": "list"
    },
    {
        "doc": "Maximum change of every optimization variable in one iteration, only for solvers with box constraints.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/box_constraints/max_change/*",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Nonlinear solver advanced options",
        "optional": [
            "f_delta_tol",
            "f_delta_step_tol",
            "derivative_along_delta_x_tol",
            "apply_gradient_fd",
            "gradient_fd_eps"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/advanced",
        "type": "object"
    },
    {
        "default": 0,
        "doc": "Dangerous Option: Quit the optimization if the solver reduces the energy by less than f_delta for consecutive f_delta_step_tol steps.",
        "min": 0,
        "pointer": "/solver/augmented_lagrangian/nonlinear/advanced/f_delta_tol",
        "type": "float"
    },
    {
        "default": 100,
        "doc": "Dangerous Option: Quit the optimization if the solver reduces the energy by less than f_delta for consecutive f_delta_step_tol steps.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/advanced/f_delta_step_tol",
        "type": "int"
    },
    {
        "default": 0,
        "doc": "Quit the optimization if the directional derivative along the descent direction is smaller than this tolerance.",
        "min": 0,
        "pointer": "/solver/augmented_lagrangian/nonlinear/advanced/derivative_along_delta_x_tol",
        "type": "float"
    },
    {
        "default": "None",
        "doc": "Expensive Option: For every iteration of the nonlinear solver, run finite difference to verify gradient of energy.",
        "options": [
            "None",
            "DirectionalDerivative",
            "FullFiniteDiff"
        ],
        "pointer": "/solver/augmented_lagrangian/nonlinear/advanced/apply_gradient_fd",
        "type": "string"
    },
    {
        "default": 1e-07,
        "doc": "Expensive Option: Eps for finite difference to verify gradient of energy.",
        "pointer": "/solver/augmented_lagrangian/nonlinear/advanced/gradient_fd_eps",
        "type": "float"
    },
    {
        "default": 1000000.0,
        "doc": "Initial weight for AL",
        "min": 0,
        "pointer": "/solver/augmented_lagrangian/initial_weight",
        "type": "float"
    },
    {
        "default": 0.01,
        "doc": "Don't stop AL unless the error is smaller than this number.",
        "min": 0,
        "pointer": "/solver/augmented_lagrangian/error",
        "type": "float"
    },
    {
        "default": 2.0,
        "doc": "Multiplication factor",
        "pointer": "/solver/augmented_lagrangian/scaling",
        "type": "float"
    },
    {
        "default": 100000000.0,
        "doc": "Maximum weight",
        "pointer": "/solver/augmented_lagrangian/max_weight",
        "type": "float"
    },
    {
        "default": 0.99,
        "doc": "Tolerance for increasing the weight or updating the lagrangian",
        "max": 1,
        "min": 0,
        "pointer": "/solver/augmented_lagrangian/eta",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Settings for contact handling in the solver.",
        "optional": [
            "CCD",
            "friction_iterations",
            "tangential_adhesion_iterations",
            "friction_convergence_tol",
            "barrier_stiffness",
            "initial_barrier_stiffness"
        ],
        "pointer": "/solver/contact",
        "type": "object"
    },
    {
        "default": 1,
        "doc": "Initial barrier stiffness if adaptive barrier is used.",
        "pointer": "/solver/contact/initial_barrier_stiffness",
        "type": "float"
    },
    {
        "default": null,
        "doc": "CCD options",
        "optional": [
            "broad_phase",
            "tolerance",
            "max_iterations"
        ],
        "pointer": "/solver/contact/CCD",
        "type": "object"
    },
    {
        "default": "hash_grid",
        "doc": "Broad phase collision-detection algorithm to use",
        "options": [
            "hash_grid",
            "HG",
            "brute_force",
            "BF",
            "spatial_hash",
            "SH",
            "bvh",
            "BVH",
            "sweep_and_prune",
            "SAP",
            "sweep_and_tiniest_queue",
            "STQ"
        ],
        "pointer": "/solver/contact/CCD/broad_phase",
        "type": "string"
    },
    {
        "default": 1e-06,
        "doc": "CCD tolerance",
        "pointer": "/solver/contact/CCD/tolerance",
        "type": "float"
    },
    {
        "default": 1000000,
        "doc": "Maximum number of iterations for continuous collision detection",
        "pointer": "/solver/contact/CCD/max_iterations",
        "type": "int"
    },
    {
        "default": 1,
        "doc": "Maximum number of update iterations for lagged friction formulation (see IPC paper).",
        "pointer": "/solver/contact/friction_iterations",
        "type": "int"
    },
    {
        "default": 1,
        "doc": "Maximum number of update iterations for lagged tangential adhesion formulation (see IPC paper).",
        "pointer": "/solver/contact/tangential_adhesion_iterations",
        "type": "int"
    },
    {
        "default": 0.01,
        "doc": "Tolerence for friction convergence",
        "pointer": "/solver/contact/friction_convergence_tol",
        "type": "float"
    },
    {
        "default": "adaptive",
        "doc": "How coefficient of clamped log-barrier function for contact is updated",
        "options": [
            "adaptive"
        ],
        "pointer": "/solver/contact/barrier_stiffness",
        "type": "string"
    },
    {
        "doc": "The coefficient of clamped log-barrier function value when not adaptive",
        "pointer": "/solver/contact/barrier_stiffness",
        "type": "float"
    },
    {
        "default": [],
        "doc": "Apply Rayleigh damping.",
        "pointer": "/solver/rayleigh_damping",
        "type": "list"
    },
    {
        "doc": "Apply Rayleigh damping to the given Form with a stiffness ratio.",
        "optional": [
            "lagging_iterations"
        )JSE_JSON";
        text += R"JSE_JSON(],
        "pointer": "/solver/rayleigh_damping/*",
        "required": [
            "form",
            "stiffness_ratio"
        ],
        "type": "object"
    },
    {
        "doc": "Apply Rayleigh damping to the given Form with a stiffness.",
        "optional": [
            "lagging_iterations"
        ],
        "pointer": "/solver/rayleigh_damping/*",
        "required": [
            "form",
            "stiffness"
        ],
        "type": "object"
    },
    {
        "doc": "Form to damp.",
        "options": [
            "elasticity",
            "contact",
            "friction"
        ],
        "pointer": "/solver/rayleigh_damping/*/form",
        "type": "string"
    },
    {
        "doc": "Ratio of to damp (stiffness = 0.75 * stiffness_ratio * \u0394t\u00b3).",
        "min": 0,
        "pointer": "/solver/rayleigh_damping/*/stiffness_ratio",
        "type": "float"
    },
    {
        "doc": "Ratio of to damp.",
        "min": 0,
        "pointer": "/solver/rayleigh_damping/*/stiffness",
        "type": "float"
    },
    {
        "default": 1,
        "doc": "Maximum number of update iterations for lagging.",
        "pointer": "/solver/rayleigh_damping/*/lagging_iterations",
        "type": "int"
    },
    {
        "default": "Discrete",
        "doc": "The method for checking if any element is flipped.",
        "options": [
            "Discrete",
            "Conservative"
        ],
        "pointer": "/solver/advanced/check_inversion",
        "type": "string"
    },
    {
        "default": 0,
        "doc": ".",
        "pointer": "/solver/advanced/jacobian_threshold",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Advanced settings for the solver",
        "optional": [
            "cache_size",
            "lump_mass_matrix",
            "lagged_regularization_weight",
            "lagged_regularization_iterations",
            "check_inversion",
            "jacobian_threshold",
            "characteristic_length",
            "characteristic_force_density"
        ],
        "pointer": "/solver/advanced",
        "type": "object"
    },
    {
        "default": -1,
        "doc": "Characteristic length, used for tolerances. Defaults to bounding box diagonal if not specified.",
        "pointer": "/solver/advanced/characteristic_length",
        "type": "float"
    },
    {
        "default": 10000,
        "doc": "Characteristic force density, used for tolerances.",
        "pointer": "/solver/advanced/characteristic_force_density",
        "type": "float"
    },
    {
        "default": 900000,
        "doc": "Maximum number of elements when the assembly values are cached.",
        "pointer": "/solver/advanced/cache_size",
        "type": "int"
    },
    {
        "default": false,
        "doc": "If true, use diagonal mass matrix with entries on the diagonal equal to the sum of entries in each row of the full mass matrix.}",
        "pointer": "/solver/advanced/lump_mass_matrix",
        "type": "bool"
    },
    {
        "default": 0,
        "doc": "Weight used to regularize singular static problems.",
        "pointer": "/solver/advanced/lagged_regularization_weight",
        "type": "float"
    },
    {
        "default": 1,
        "doc": "Number of regularize singular static problems.",
        "pointer": "/solver/advanced/lagged_regularization_iterations",
        "type": "int"
    },
    {
        "doc": "Material Parameters lists including ID pointing to volume selection, Young's modulus ($E$), Poisson's ratio ($\\nu$), Density ($\\rho$), or Lam\u00e9 constants ($\\lambda$ and $\\mu$).",
        "pointer": "/materials",
        "type": "list"
    },
    {
        "doc": "Type of material",
        "options": [
            "LinearElasticity",
            "HookeLinearElasticity",
            "SaintVenant",
            "NeoHookean",
            "MooneyRivlin",
            "MooneyRivlin3Param",
            "MooneyRivlin3ParamSymbolic",
            "UnconstrainedOgden",
            "IncompressibleOgden",
            "Stokes",
            "ActiveFiber",
            "HGOFiber",
            "IsochoricNeoHookean",
            "NavierStokes",
            "OperatorSplitting",
            "Electrostatics",
            "MaterialSum",
            "IncompressibleLinearElasticity",
            "Laplacian",
            "Helmholtz",
            "Bilaplacian",
            "AMIPS",
            "FixedCorotational",
            "VolumePenalty"
        ],
        "pointer": "/materials/*/type",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "Volume selection ID",
        "pointer": "/materials/*/id",
        "type": "int"
    },
    {
        "doc": "Volume selection IDs",
        "pointer": "/materials/*/id",
        "type": "list"
    },
    {
        "doc": "Volume selection ID",
        "pointer": "/materials/*/id/*",
        "type": "int"
    },
    {
        "doc": "Material Parameters including ID, Young's modulus ($E$), Poisson's ratio ($\\nu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "NeoHookean"
    },
    {
        "doc": "Material Parameters including ID, Lam\u00e9 first ($\\lambda$), Lam\u00e9 second ($\\mu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "NeoHookean"
    },
    {
        "doc": "Material Parameters including ID, Young's modulus ($E$), Poisson's ratio ($\\nu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "IsochoricNeoHookean"
    },
    {
        "doc": "Material Parameters including ID, Lam\u00e9 first ($\\lambda$), Lam\u00e9 second ($\\mu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "IsochoricNeoHookean"
    },
    {
        "doc": "Material Parameters including ID, for Mooney-Rivlin",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "c1",
            "c2",
            "k"
        ],
        "type": "object",
        "type_name": "MooneyRivlin"
    },
    {
        "doc": "Material Parameters including ID, for Mooney-Rivlin",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "c1",
            "c2",
            "c3",
            "d1"
        ],
        "type": "object",
        "type_name": "MooneyRivlin3Param"
    },
    {
        "doc": "Material Parameters including ID, for Mooney-Rivlin",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "c1",
            "c2",
            "c3",
            "d1"
        ],
        "type": "object",
        "type_name": "MooneyRivlin3ParamSymbolic"
    },
    {
        "doc": "Material Parameters including ID, for [Ogden](https://en.wikipedia.org/wiki/Ogden_hyperelastic_model).",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/m)JSE_JSON";
        text += R"JSE_JSON(aterials/*",
        "required": [
            "type",
            "alphas",
            "mus",
            "Ds"
        ],
        "type": "object",
        "type_name": "UnconstrainedOgden"
    },
    {
        "doc": "Material Parameters including ID, for [Ogden](https://en.wikipedia.org/wiki/Ogden_hyperelastic_model).",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "c",
            "m",
            "k"
        ],
        "type": "object",
        "type_name": "IncompressibleOgden"
    },
    {
        "doc": "Material Parameters including ID, Young's modulus ($E$), Poisson's ratio ($\\nu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "LinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, Lam\u00e9 first ($\\lambda$), Lam\u00e9 second ($\\mu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "LinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, E, nu, density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "fiber_direction"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "HookeLinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, E, nu, density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "fiber_direction"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "elasticity_tensor"
        ],
        "type": "object",
        "type_name": "HookeLinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, E, nu, density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "fiber_direction",
            "psi"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "SaintVenant"
    },
    {
        "doc": "Material Parameters including ID, E, nu, density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "fiber_direction",
            "psi"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "elasticity_tensor"
        ],
        "type": "object",
        "type_name": "SaintVenant"
    },
    {
        "doc": "Material Parameters including ID, viscosity, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "viscosity"
        ],
        "type": "object",
        "type_name": "Stokes"
    },
    {
        "doc": "Material Parameters including ID, viscosity, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "viscosity"
        ],
        "type": "object",
        "type_name": "NavierStokes"
    },
    {
        "doc": "Material Parameters including ID, viscosity, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "viscosity"
        ],
        "type": "object",
        "type_name": "OperatorSplitting"
    },
    {
        "doc": "Material Parameters including ID, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "epsilon"
        ],
        "type": "object",
        "type_name": "Electrostatics"
    },
    {
        "doc": "Material Parameters including ID, Young's modulus ($E$), Poisson's ratio ($\\nu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "IncompressibleLinearElasticity"
    },
    {
        "doc": "Model that is a sum of other models",
        "optional": [
            "id",
            "models",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "MaterialSum"
    },
    {
        "doc": "Material Parameters including ID, Lam\u00e9 first ($\\lambda$), Lam\u00e9 second ($\\mu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "IncompressibleLinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Laplacian"
    },
    {
        "doc": "Material Parameters including ID, k, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "k"
        ],
        "type": "object",
        "type_name": "Helmholtz"
    },
    {
        "doc": "Material Parameters including ID, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Bilaplacian"
    },
    {
        "doc": "Material Parameters including ID",
        "optional": [
            "use_rest_pose",
            "weight",
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "AMIPS"
    },
    {
        "doc": "Material Parameters including ID",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "FixedCorotational"
    },
    {
        "doc": "Material Parameters including ID",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "FixedCorotational"
    },
    {
        "doc": "Material Parameters including ID",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "k"
        ],
        "type": "object",
        "type_name": "VolumePenalty"
    },
    {
        "doc": "Material Parameters",
        "optional": [
            "id",
            "rho",
            "fiber_direction"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "k1",
           )JSE_JSON";
        text += R"JSE_JSON( "k2"
        ],
        "type": "object",
        "type_name": "HGOFiber"
    },
    {
        "doc": "Material Parameters",
        "optional": [
            "id",
            "rho",
            "Tmax",
            "fiber_direction"
        ],
        "pointer": "/materials/*",
        "required": [
            "type",
            "activation"
        ],
        "type": "object",
        "type_name": "ActiveFiber"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/E",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/E",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/E",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/E",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/E/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/E/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/E/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/E/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/E/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/nu",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/nu",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/nu",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/nu",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/nu/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/nu/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/nu/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/nu/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/nu/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/viscosity",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/viscosity",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/viscosity",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/viscosity",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/viscosity/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/viscosity/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/viscosity/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/viscosity/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/viscosity/function_name",
        "type": "string"
    },
    {
        "doc": "Symmetric elasticity tensor",
        "pointer": "/materials/*/elasticity_tensor",
        "type": "list"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/materials/*/elasticity_tensor/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/elasticity_tensor/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/elasticity_tensor/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/elasticity_tensor/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/elasticity_tensor/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/elasticity_tensor/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/elasticity_tensor/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/elasticity_tensor/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/elasticity_tensor/*/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "Fiber direction",
        "pointer": "/materials/*/fiber_direction",
        "type": "list"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/materials/*/fiber_direction/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/fiber_direction/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/fiber_direction/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/fiber_direction/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/fiber_direction/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/fiber_direction/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/fiber_direction/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/fiber_direction/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/fiber_direction/*/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "Fiber direction row",
        "pointer": "/materials/*/fiber_direction/*",
        "type": "list"
    },
    {
        "default")JSE_JSON";
        text += R"JSE_JSON(: 0,
        "doc": "Value as a constant float",
        "pointer": "/materials/*/fiber_direction/*/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/fiber_direction/*/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/fiber_direction/*/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/fiber_direction/*/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/fiber_direction/*/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/fiber_direction/*/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/fiber_direction/*/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/fiber_direction/*/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/fiber_direction/*/*/function_name",
        "type": "string"
    },
    {
        "default": 1,
        "doc": "Value as a constant float",
        "pointer": "/materials/*/rho",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/rho",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/rho",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/rho",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/rho/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/rho/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/rho/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/rho/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/rho/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/epsilon",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/epsilon",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/epsilon",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/epsilon",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/epsilon/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/epsilon/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/epsilon/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/epsilon/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/epsilon/function_name",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/materials/*/phi",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/phi",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/phi",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/phi",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/phi/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/phi/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/phi/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/phi/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/phi/function_name",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/materials/*/psi",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/psi",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/psi",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/psi",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/psi/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/psi/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/psi/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/psi/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/psi/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/k",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/k",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/k",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/k",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/k/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/k/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/mate)JSE_JSON";
        text += R"JSE_JSON(rials/*/k/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/k/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/k/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/mu",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/mu",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/mu",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/mu",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/mu/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/mu/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/mu/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/mu/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/mu/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/lambda",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/lambda",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/lambda",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/lambda",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/lambda/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/lambda/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/lambda/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/lambda/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/lambda/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/c1",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/c1",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/c1",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/c1",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/c1/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/c1/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/c1/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/c1/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/c1/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/c2",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/c2",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/c2",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/c2",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/c2/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/c2/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/c2/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/c2/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/c2/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/c3",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/c3",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/c3",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/c3",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/c3/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/c3/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/c3/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/c3/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/c3/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/d1",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/d1",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/d1",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/d1",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/d1/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/d1/value",
  )JSE_JSON";
        text += R"JSE_JSON(      "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/d1/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/d1/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/d1/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/k1",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/k1",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/k1",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/k1",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/k1/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/k1/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/k1/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/k1/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/k1/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/k2",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/k2",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/k2",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/k2",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/k2/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/k2/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/k2/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/k2/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/k2/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/alphas",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/alphas",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/alphas",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/alphas",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/alphas/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/alphas/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/alphas/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/alphas/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/alphas/function_name",
        "type": "string"
    },
    {
        "doc": "Ogden mu",
        "pointer": "/materials/*/mus",
        "type": "list"
    },
    {
        "doc": "Ogden D",
        "pointer": "/materials/*/Ds",
        "type": "list"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/alphas/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/alphas/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/alphas/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/alphas/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/alphas/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/alphas/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/alphas/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/alphas/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/alphas/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/mus/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/mus/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/mus/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/mus/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/mus/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/mus/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/mus/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/mus/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/mus/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/Ds/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/Ds/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/Ds/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "do)JSE_JSON";
        text += R"JSE_JSON(c": "Value from python function",
        "pointer": "/materials/*/Ds/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/Ds/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/Ds/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/Ds/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/Ds/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/Ds/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/c",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/c",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/c",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/c",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/c/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/c/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/c/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/c/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/c/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/m",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/m",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/m",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/m",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/m/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/m/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/m/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/m/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/m/function_name",
        "type": "string"
    },
    {
        "doc": "Coefficient(s) of Incompressible Ogden",
        "pointer": "/materials/*/c",
        "type": "list"
    },
    {
        "doc": "Exponent(s) of Incompressible Ogden",
        "pointer": "/materials/*/m",
        "type": "list"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/c/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/c/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/c/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/c/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/c/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/c/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/c/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/c/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/c/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/m/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/m/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/m/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/m/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/m/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/m/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/m/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/m/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/m/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/activation",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/activation",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/activation",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/activation",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/activation/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/activation/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/activation/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/activation/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/activation/function_name",
        "type": "string"
    },
    {
        "default": 1,
        )JSE_JSON";
        text += R"JSE_JSON("doc": "Value as a constant float",
        "pointer": "/materials/*/Tmax",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/Tmax",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/Tmax",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/Tmax",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/Tmax/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/Tmax/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/Tmax/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/Tmax/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/Tmax/function_name",
        "type": "string"
    },
    {
        "default": false,
        "doc": "Use amips wrt to rest pose or the regular element",
        "pointer": "/materials/*/use_rest_pose",
        "type": "bool"
    },
    {
        "default": 1.0,
        "doc": "Scale factor for the AMIPS energy.",
        "pointer": "/materials/*/weight",
        "type": "float"
    },
    {
        "default": [],
        "doc": "List of models",
        "pointer": "/materials/*/models",
        "type": "list"
    },
    {
        "doc": "Type of material",
        "options": [
            "LinearElasticity",
            "HookeLinearElasticity",
            "SaintVenant",
            "NeoHookean",
            "MooneyRivlin",
            "MooneyRivlin3Param",
            "MooneyRivlin3ParamSymbolic",
            "UnconstrainedOgden",
            "IncompressibleOgden",
            "Stokes",
            "ActiveFiber",
            "HGOFiber",
            "IsochoricNeoHookean",
            "NavierStokes",
            "OperatorSplitting",
            "Electrostatics",
            "MaterialSum",
            "IncompressibleLinearElasticity",
            "Laplacian",
            "Helmholtz",
            "Bilaplacian",
            "AMIPS",
            "FixedCorotational",
            "VolumePenalty"
        ],
        "pointer": "/materials/*/models/*/type",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "Volume selection ID",
        "pointer": "/materials/*/models/*/id",
        "type": "int"
    },
    {
        "doc": "Volume selection IDs",
        "pointer": "/materials/*/models/*/id",
        "type": "list"
    },
    {
        "doc": "Volume selection ID",
        "pointer": "/materials/*/models/*/id/*",
        "type": "int"
    },
    {
        "doc": "Material Parameters including ID, Young's modulus ($E$), Poisson's ratio ($\\nu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "NeoHookean"
    },
    {
        "doc": "Material Parameters including ID, Lam\u00e9 first ($\\lambda$), Lam\u00e9 second ($\\mu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "NeoHookean"
    },
    {
        "doc": "Material Parameters including ID, Young's modulus ($E$), Poisson's ratio ($\\nu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "IsochoricNeoHookean"
    },
    {
        "doc": "Material Parameters including ID, Lam\u00e9 first ($\\lambda$), Lam\u00e9 second ($\\mu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "IsochoricNeoHookean"
    },
    {
        "doc": "Material Parameters including ID, for Mooney-Rivlin",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "c1",
            "c2",
            "k"
        ],
        "type": "object",
        "type_name": "MooneyRivlin"
    },
    {
        "doc": "Material Parameters including ID, for Mooney-Rivlin",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "c1",
            "c2",
            "c3",
            "d1"
        ],
        "type": "object",
        "type_name": "MooneyRivlin3Param"
    },
    {
        "doc": "Material Parameters including ID, for Mooney-Rivlin",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "c1",
            "c2",
            "c3",
            "d1"
        ],
        "type": "object",
        "type_name": "MooneyRivlin3ParamSymbolic"
    },
    {
        "doc": "Material Parameters including ID, for [Ogden](https://en.wikipedia.org/wiki/Ogden_hyperelastic_model).",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "alphas",
            "mus",
            "Ds"
        ],
        "type": "object",
        "type_name": "UnconstrainedOgden"
    },
    {
        "doc": "Material Parameters including ID, for [Ogden](https://en.wikipedia.org/wiki/Ogden_hyperelastic_model).",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "c",
            "m",
            "k"
        ],
        "type": "object",
        "type_name": "IncompressibleOgden"
    },
    {
        "doc": "Material Parameters including ID, Young's modulus ($E$), Poisson's ratio ($\\nu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "LinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, Lam\u00e9 first ($\\lambda$), Lam\u00e9 second ($\\mu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "LinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, E, nu, density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "fiber_direction"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "E",
            "nu"
        )JSE_JSON";
        text += R"JSE_JSON(],
        "type": "object",
        "type_name": "HookeLinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, E, nu, density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "fiber_direction"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "elasticity_tensor"
        ],
        "type": "object",
        "type_name": "HookeLinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, E, nu, density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "fiber_direction",
            "psi"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "SaintVenant"
    },
    {
        "doc": "Material Parameters including ID, E, nu, density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "fiber_direction",
            "psi"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "elasticity_tensor"
        ],
        "type": "object",
        "type_name": "SaintVenant"
    },
    {
        "doc": "Material Parameters including ID, viscosity, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "viscosity"
        ],
        "type": "object",
        "type_name": "Stokes"
    },
    {
        "doc": "Material Parameters including ID, viscosity, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "viscosity"
        ],
        "type": "object",
        "type_name": "NavierStokes"
    },
    {
        "doc": "Material Parameters including ID, viscosity, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "viscosity"
        ],
        "type": "object",
        "type_name": "OperatorSplitting"
    },
    {
        "doc": "Material Parameters including ID, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "epsilon"
        ],
        "type": "object",
        "type_name": "Electrostatics"
    },
    {
        "doc": "Material Parameters including ID, Young's modulus ($E$), Poisson's ratio ($\\nu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "IncompressibleLinearElasticity"
    },
    {
        "doc": "Model that is a sum of other models",
        "optional": [
            "id",
            "models",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "MaterialSum"
    },
    {
        "doc": "Material Parameters including ID, Lam\u00e9 first ($\\lambda$), Lam\u00e9 second ($\\mu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "IncompressibleLinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Laplacian"
    },
    {
        "doc": "Material Parameters including ID, k, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "k"
        ],
        "type": "object",
        "type_name": "Helmholtz"
    },
    {
        "doc": "Material Parameters including ID, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Bilaplacian"
    },
    {
        "doc": "Material Parameters including ID",
        "optional": [
            "use_rest_pose",
            "weight",
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "AMIPS"
    },
    {
        "doc": "Material Parameters including ID",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "FixedCorotational"
    },
    {
        "doc": "Material Parameters including ID",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "FixedCorotational"
    },
    {
        "doc": "Material Parameters including ID",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "k"
        ],
        "type": "object",
        "type_name": "VolumePenalty"
    },
    {
        "doc": "Material Parameters",
        "optional": [
            "id",
            "rho",
            "fiber_direction"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "k1",
            "k2"
        ],
        "type": "object",
        "type_name": "HGOFiber"
    },
    {
        "doc": "Material Parameters",
        "optional": [
            "id",
            "rho",
            "Tmax",
            "fiber_direction"
        ],
        "pointer": "/materials/*/models/*",
        "required": [
            "type",
            "activation"
        ],
        "type": "object",
        "type_name": "ActiveFiber"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/E",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/E",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/E",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/E",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/E/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/E/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/E/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the f)JSE_JSON";
        text += R"JSE_JSON(unction to evaluate the value",
        "pointer": "/materials/*/models/*/E/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/E/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/nu",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/nu",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/nu",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/nu",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/nu/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/nu/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/nu/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/nu/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/nu/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/viscosity",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/viscosity",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/viscosity",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/viscosity",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/viscosity/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/viscosity/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/viscosity/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/viscosity/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/viscosity/function_name",
        "type": "string"
    },
    {
        "doc": "Symmetric elasticity tensor",
        "pointer": "/materials/*/models/*/elasticity_tensor",
        "type": "list"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/elasticity_tensor/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/elasticity_tensor/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/elasticity_tensor/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/elasticity_tensor/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/elasticity_tensor/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/elasticity_tensor/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/elasticity_tensor/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/elasticity_tensor/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/elasticity_tensor/*/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "Fiber direction",
        "pointer": "/materials/*/models/*/fiber_direction",
        "type": "list"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/fiber_direction/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/fiber_direction/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/fiber_direction/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/fiber_direction/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/fiber_direction/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/fiber_direction/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/fiber_direction/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/fiber_direction/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/fiber_direction/*/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "Fiber direction row",
        "pointer": "/materials/*/models/*/fiber_direction/*",
        "type": "list"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/fiber_direction/*/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/fiber_direction/*/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/fiber_direction/*/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/fiber_direction/*/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/fiber_direction/*/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/fiber_direction/*/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/mater)JSE_JSON";
        text += R"JSE_JSON(ials/*/models/*/fiber_direction/*/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/fiber_direction/*/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/fiber_direction/*/*/function_name",
        "type": "string"
    },
    {
        "default": 1,
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/rho",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/rho",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/rho",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/rho",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/rho/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/rho/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/rho/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/rho/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/rho/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/epsilon",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/epsilon",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/epsilon",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/epsilon",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/epsilon/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/epsilon/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/epsilon/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/epsilon/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/epsilon/function_name",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/phi",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/phi",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/phi",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/phi",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/phi/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/phi/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/phi/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/phi/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/phi/function_name",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/psi",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/psi",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/psi",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/psi",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/psi/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/psi/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/psi/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/psi/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/psi/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/k",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/k",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/k",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/k",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/k/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/k/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/k/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/k/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/k/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/mu",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/mu",
        "type": "string"
    },
    {
     )JSE_JSON";
        text += R"JSE_JSON(   "doc": "Value with unit",
        "pointer": "/materials/*/models/*/mu",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/mu",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/mu/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/mu/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/mu/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/mu/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/mu/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/lambda",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/lambda",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/lambda",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/lambda",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/lambda/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/lambda/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/lambda/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/lambda/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/lambda/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/c1",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/c1",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/c1",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/c1",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/c1/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/c1/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/c1/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/c1/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/c1/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/c2",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/c2",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/c2",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/c2",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/c2/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/c2/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/c2/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/c2/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/c2/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/c3",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/c3",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/c3",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/c3",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/c3/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/c3/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/c3/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/c3/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/c3/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/d1",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/d1",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/d1",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/d1",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/d1/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/d1/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/d1/value",
        "type": "string"
    },
    {
        "do)JSE_JSON";
        text += R"JSE_JSON(c": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/d1/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/d1/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/k1",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/k1",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/k1",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/k1",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/k1/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/k1/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/k1/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/k1/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/k1/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/k2",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/k2",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/k2",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/k2",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/k2/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/k2/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/k2/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/k2/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/k2/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/alphas",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/alphas",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/alphas",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/alphas",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/alphas/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/alphas/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/alphas/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/alphas/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/alphas/function_name",
        "type": "string"
    },
    {
        "doc": "Ogden mu",
        "pointer": "/materials/*/models/*/mus",
        "type": "list"
    },
    {
        "doc": "Ogden D",
        "pointer": "/materials/*/models/*/Ds",
        "type": "list"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/alphas/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/alphas/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/alphas/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/alphas/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/alphas/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/alphas/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/alphas/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/alphas/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/alphas/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/mus/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/mus/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/mus/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/mus/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/mus/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/mus/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/mus/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/mus/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/mus/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/Ds/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
       )JSE_JSON";
        text += R"JSE_JSON( "pointer": "/materials/*/models/*/Ds/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/Ds/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/Ds/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/Ds/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/Ds/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/Ds/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/Ds/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/Ds/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/c",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/c",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/c",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/c",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/c/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/c/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/c/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/c/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/c/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/m",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/m",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/m",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/m",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/m/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/m/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/m/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/m/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/m/function_name",
        "type": "string"
    },
    {
        "doc": "Coefficient(s) of Incompressible Ogden",
        "pointer": "/materials/*/models/*/c",
        "type": "list"
    },
    {
        "doc": "Exponent(s) of Incompressible Ogden",
        "pointer": "/materials/*/models/*/m",
        "type": "list"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/c/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/c/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/c/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/c/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/c/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/c/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/c/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/c/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/c/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/m/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/m/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/m/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/m/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/m/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/m/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/m/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/m/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/m/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/activation",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/activation",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/activation",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/activation",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "po)JSE_JSON";
        text += R"JSE_JSON(inter": "/materials/*/models/*/activation/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/activation/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/activation/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/activation/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/activation/function_name",
        "type": "string"
    },
    {
        "default": 1,
        "doc": "Value as a constant float",
        "pointer": "/materials/*/models/*/Tmax",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/materials/*/models/*/Tmax",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/materials/*/models/*/Tmax",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/materials/*/models/*/Tmax",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/materials/*/models/*/Tmax/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/materials/*/models/*/Tmax/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/materials/*/models/*/Tmax/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/materials/*/models/*/Tmax/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/materials/*/models/*/Tmax/function_name",
        "type": "string"
    },
    {
        "default": false,
        "doc": "Use amips wrt to rest pose or the regular element",
        "pointer": "/materials/*/models/*/use_rest_pose",
        "type": "bool"
    },
    {
        "default": 1.0,
        "doc": "Scale factor for the AMIPS energy.",
        "pointer": "/materials/*/models/*/weight",
        "type": "float"
    },
    {
        "default": null,
        "doc": "The settings for boundary conditions.",
        "optional": [
            "rhs",
            "dirichlet_boundary",
            "neumann_boundary",
            "normal_aligned_neumann_boundary",
            "pressure_boundary",
            "pressure_cavity",
            "obstacle_displacements",
            "periodic_boundary"
        ],
        "pointer": "/boundary_conditions",
        "type": "object"
    },
    {
        "default": null,
        "doc": "Options for periodic boundary conditions.",
        "optional": [
            "enabled",
            "tolerance",
            "correspondence",
            "linear_displacement_offset",
            "fixed_macro_strain",
            "force_zero_mean"
        ],
        "pointer": "/boundary_conditions/periodic_boundary",
        "type": "object"
    },
    {
        "default": false,
        "doc": "The periodic solution is not unique, set to true to find the solution with zero mean.",
        "pointer": "/boundary_conditions/periodic_boundary/force_zero_mean",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "",
        "pointer": "/boundary_conditions/periodic_boundary/enabled",
        "type": "bool"
    },
    {
        "default": 1e-05,
        "doc": "Relative tolerance of deciding periodic correspondence",
        "pointer": "/boundary_conditions/periodic_boundary/tolerance",
        "type": "float"
    },
    {
        "default": [],
        "doc": "Periodic directions for periodic boundary conditions. If not specified, default to axis-aligned directions.",
        "pointer": "/boundary_conditions/periodic_boundary/correspondence",
        "type": "list"
    },
    {
        "default": [],
        "doc": "One periodic direction.",
        "pointer": "/boundary_conditions/periodic_boundary/correspondence/*",
        "type": "list"
    },
    {
        "doc": "One entry of a periodic direction.",
        "pointer": "/boundary_conditions/periodic_boundary/correspondence/*/*",
        "type": "float"
    },
    {
        "default": [],
        "doc": "",
        "pointer": "/boundary_conditions/periodic_boundary/fixed_macro_strain",
        "type": "list"
    },
    {
        "doc": "",
        "pointer": "/boundary_conditions/periodic_boundary/fixed_macro_strain/*",
        "type": "int"
    },
    {
        "default": [],
        "doc": "",
        "pointer": "/boundary_conditions/periodic_boundary/linear_displacement_offset",
        "type": "list"
    },
    {
        "doc": "",
        "pointer": "/boundary_conditions/periodic_boundary/linear_displacement_offset/*",
        "type": "list"
    },
    {
        "doc": "",
        "pointer": "/boundary_conditions/periodic_boundary/linear_displacement_offset/*/*",
        "type": "float"
    },
    {
        "doc": "",
        "pointer": "/boundary_conditions/periodic_boundary/linear_displacement_offset/*/*",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/boundary_conditions/rhs",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/boundary_conditions/rhs",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/boundary_conditions/rhs",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/boundary_conditions/rhs",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/boundary_conditions/rhs/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/boundary_conditions/rhs/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/boundary_conditions/rhs/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/boundary_conditions/rhs/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/boundary_conditions/rhs/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "Right-hand side of the system being solved for vector-valued PDEs.",
        "pointer": "/boundary_conditions/rhs",
        "type": "list"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/boundary_conditions/rhs/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/boundary_conditions/rhs/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/boundary_conditions/rhs/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/boundary_conditions/rhs/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/boundary_conditions/rhs/*/unit",
        "type": "string"
    },
    {
       )JSE_JSON";
        text += R"JSE_JSON( "doc": "The value of the constant",
        "pointer": "/boundary_conditions/rhs/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/boundary_conditions/rhs/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/boundary_conditions/rhs/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/boundary_conditions/rhs/*/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "The list of boundary conditions for the main variable. Elements of the list are assignment pairs (ID, value) where ID is assigned by surface selection.",
        "pointer": "/boundary_conditions/dirichlet_boundary",
        "type": "list"
    },
    {
        "doc": "ID of boundary condition from surface selection.",
        "max": 2147483646,
        "min": 0,
        "pointer": "/boundary_conditions/dirichlet_boundary/*/id",
        "type": "int"
    },
    {
        "doc": "select all ids.",
        "options": [
            "all"
        ],
        "pointer": "/boundary_conditions/dirichlet_boundary/*/id",
        "type": "string"
    },
    {
        "doc": "Values of boundary condition, length 1 for scalar-valued pde, 2/3 for vector-valued PDEs depending on the dimension.",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value",
        "type": "list"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "interpolation of boundary condition",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation",
        "type": "list"
    },
    {
        "default": {
            "type": "none"
        },
        "doc": "interpolation of boundary condition",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "none"
    },
    {
        "doc": "interpolation of boundary condition",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "linear"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "from"
        ],
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*",
        "required": [
            "type",
            "to"
        ],
        "type": "object",
        "type_name": "linear_ramp"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "extend"
        ],
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*",
        "required": [
            "type",
            "points",
            "values"
        ],
        "type": "object",
        "type_name": "piecewise_constant"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "extend"
        ],
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*",
        "required": [
            "type",
            "points",
            "values"
        ],
        "type": "object",
        "type_name": "piecewise_linear"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "extend"
        ],
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*",
        "required": [
            "type",
            "points",
            "values"
        ],
        "type": "object",
        "type_name": "piecewise_cubic"
    },
    {
        "doc": "type of interpolation of boundary condition",
        "options": [
            "none",
            "linear",
            "linear_ramp",
            "piecewise_constant",
            "piecewise_linear",
            "piecewise_cubic"
        ],
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*/type",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "interpolation starting time",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*/from",
        "type": "float"
    },
    {
        "doc": "interpolation ending time",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*/to",
        "type": "float"
    },
    {
        "doc": "interpolation time points",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*/points",
        "type": "list"
    },
    {
        "doc": "interpolation time point",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*/points/*",
        "type": "float"
    },
    {
        "doc": "interpolation values",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*/values",
        "type": "list"
    },
    {
        "doc": "interpolation value",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*/values/*",
        "type": "float"
    },
    {
        "default": "constant",
        "doc": "how to extend the piecewise interpolation",
        "options": [
            "constant",
            "extrapolate",
            "repeat",
            "repeat_offset"
        ],
        "pointer": "/boundary_conditions/dirichlet_boundary/*/interpolation/*/extend",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Dirichlet boundary condition.",
        "optional": [
            "time_reference",
            "interpolation",
            "dimension"
        ],
        "pointer": "/boundary_conditions/dirichlet_boundary/*",
        "required": [
            "id",
            "value"
        ],
        "type": "object"
    },
    {
        "doc": "Dirichlet boundary condition loaded from a file, <node_id> <bc values>, 1 for scalar, 2/3 for tensor depending on dimension.",
        "pointer": "/boundary_conditions/dirichlet_boundary/*",
        "type": "string"
    },
    {
        "doc": "Dirichlet boundary condition specified per timestep.",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*",
        "type": "list"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/)JSE_JSON";
        text += R"JSE_JSON(boundary_conditions/dirichlet_boundary/*/value/*/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/value/*/*/function_name",
        "type": "string"
    },
    {
        "default": [
            true,
            true,
            true
        ],
        "doc": "List of 2 (2D) or 3 (3D) boolean values indicating if the Dirichlet boundary condition  is applied for a particular dimension.",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/dimension",
        "type": "list"
    },
    {
        "default": true,
        "doc": "value",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/dimension/*",
        "type": "bool"
    },
    {
        "default": [],
        "doc": "List of times when the Dirichlet boundary condition is specified",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/time_reference",
        "type": "list"
    },
    {
        "doc": "Values of Dirichlet boundary condition for timestep",
        "pointer": "/boundary_conditions/dirichlet_boundary/*/time_reference/*",
        "type": "float"
    },
    {
        "default": [],
        "doc": "The list of boundary conditions for the main variable. Elements of the list are assignment pairs (ID, value) where ID is assigned by surface selection.",
        "pointer": "/boundary_conditions/neumann_boundary",
        "type": "list"
    },
    {
        "doc": "ID of boundary condition from surface selection.",
        "max": 2147483646,
        "min": 0,
        "pointer": "/boundary_conditions/neumann_boundary/*/id",
        "type": "int"
    },
    {
        "doc": "select all ids.",
        "options": [
            "all"
        ],
        "pointer": "/boundary_conditions/neumann_boundary/*/id",
        "type": "string"
    },
    {
        "doc": "Values of boundary condition, length 1 for scalar-valued pde, 2/3 for vector-valued PDEs depending on the dimension.",
        "pointer": "/boundary_conditions/neumann_boundary/*/value",
        "type": "list"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/boundary_conditions/neumann_boundary/*/value/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/boundary_conditions/neumann_boundary/*/value/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/boundary_conditions/neumann_boundary/*/value/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/boundary_conditions/neumann_boundary/*/value/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/boundary_conditions/neumann_boundary/*/value/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/boundary_conditions/neumann_boundary/*/value/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/boundary_conditions/neumann_boundary/*/value/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/boundary_conditions/neumann_boundary/*/value/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/boundary_conditions/neumann_boundary/*/value/*/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "interpolation of boundary condition",
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation",
        "type": "list"
    },
    {
        "default": {
            "type": "none"
        },
        "doc": "interpolation of boundary condition",
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "none"
    },
    {
        "doc": "interpolation of boundary condition",
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "linear"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "from"
        ],
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*",
        "required": [
            "type",
            "to"
        ],
        "type": "object",
        "type_name": "linear_ramp"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "extend"
        ],
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*",
        "required": [
            "type",
            "points",
            "values"
        ],
        "type": "object",
        "type_name": "piecewise_constant"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "extend"
        ],
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*",
        "required": [
            "type",
            "points",
            "values"
        ],
        "type": "object",
        "type_name": "piecewise_linear"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "extend"
        ],
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*",
        "required": [
            "type",
            "points",
            "values"
        ],
        "type": "object",
        "type_name": "piecewise_cubic"
    },
    {
        "doc": "type of interpolation of boundary condition",
        "options": [
            "none",
            "linear",
            "linear_ramp",
            "piecewise_constant",
            "piecewise_linear",
            "piecewise_cubic"
        ],
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*/type",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "interpolation starting time",
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*/from",
        "type": "float"
    },
    {
        "doc": "interpolation ending time",
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*/to",
        "type": "float"
    },
    {
        "doc": "interpolatio)JSE_JSON";
        text += R"JSE_JSON(n time points",
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*/points",
        "type": "list"
    },
    {
        "doc": "interpolation time point",
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*/points/*",
        "type": "float"
    },
    {
        "doc": "interpolation values",
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*/values",
        "type": "list"
    },
    {
        "doc": "interpolation value",
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*/values/*",
        "type": "float"
    },
    {
        "default": "constant",
        "doc": "how to extend the piecewise interpolation",
        "options": [
            "constant",
            "extrapolate",
            "repeat",
            "repeat_offset"
        ],
        "pointer": "/boundary_conditions/neumann_boundary/*/interpolation/*/extend",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Neumann boundary condition",
        "optional": [
            "interpolation"
        ],
        "pointer": "/boundary_conditions/neumann_boundary/*",
        "required": [
            "id",
            "value"
        ],
        "type": "object"
    },
    {
        "default": [],
        "doc": "Neumann boundary condition for normal times value for vector-valued PDEs.",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary",
        "type": "list"
    },
    {
        "default": null,
        "doc": "pressure BC entry",
        "optional": [
            "interpolation"
        ],
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*",
        "required": [
            "id",
            "value"
        ],
        "type": "object"
    },
    {
        "doc": "ID for the pressure Neumann boundary condition",
        "max": 2147483646,
        "min": 0,
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/id",
        "type": "int"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/value",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/value",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/value",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/value",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/value/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/value/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/value/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/value/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/value/function_name",
        "type": "string"
    },
    {
        "default": {
            "type": "none"
        },
        "doc": "interpolation of boundary condition",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "none"
    },
    {
        "doc": "interpolation of boundary condition",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "linear"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "from"
        ],
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation",
        "required": [
            "type",
            "to"
        ],
        "type": "object",
        "type_name": "linear_ramp"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "extend"
        ],
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation",
        "required": [
            "type",
            "points",
            "values"
        ],
        "type": "object",
        "type_name": "piecewise_constant"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "extend"
        ],
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation",
        "required": [
            "type",
            "points",
            "values"
        ],
        "type": "object",
        "type_name": "piecewise_linear"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "extend"
        ],
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation",
        "required": [
            "type",
            "points",
            "values"
        ],
        "type": "object",
        "type_name": "piecewise_cubic"
    },
    {
        "doc": "type of interpolation of boundary condition",
        "options": [
            "none",
            "linear",
            "linear_ramp",
            "piecewise_constant",
            "piecewise_linear",
            "piecewise_cubic"
        ],
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation/type",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "interpolation starting time",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation/from",
        "type": "float"
    },
    {
        "doc": "interpolation ending time",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation/to",
        "type": "float"
    },
    {
        "doc": "interpolation time points",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation/points",
        "type": "list"
    },
    {
        "doc": "interpolation time point",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation/points/*",
        "type": "float"
    },
    {
        "doc": "interpolation values",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation/values",
        "type": "list"
    },
    {
        "doc": "interpolation value",
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation/values/*",
        "type": "float"
    },
    {
        "default": "constant",
        "doc": "how to extend the piecewise interpolation",
        "options": [
            "constant",
            "extrapolate",
            "repeat",
            "repeat_offset"
        ],
        "pointer": "/boundary_conditions/normal_aligned_neumann_boundary/*/interpolation/extend",
        "type": "string"
    },
    {
        "default": [],
        "doc": "Neumann boundary condition for normal times value for vector-valued PDEs.",
        "pointer": "/boundary_conditions/pressure_boundary",
        "type": "list"
    },
    {
        "default": null,
        "doc": "pressure BC entry",
        "optional": [
            "time_reference"
  )JSE_JSON";
        text += R"JSE_JSON(      ],
        "pointer": "/boundary_conditions/pressure_boundary/*",
        "required": [
            "id",
            "value"
        ],
        "type": "object"
    },
    {
        "doc": "ID for the pressure Neumann boundary condition",
        "max": 2147483646,
        "min": 0,
        "pointer": "/boundary_conditions/pressure_boundary/*/id",
        "type": "int"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/boundary_conditions/pressure_boundary/*/value",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/boundary_conditions/pressure_boundary/*/value",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/boundary_conditions/pressure_boundary/*/value",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/boundary_conditions/pressure_boundary/*/value",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/function_name",
        "type": "string"
    },
    {
        "doc": "Values of pressure boundary condition specified per timestep",
        "pointer": "/boundary_conditions/pressure_boundary/*/value",
        "type": "list"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/boundary_conditions/pressure_boundary/*/value/*/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "List of times when the pressure boundary condition is specified",
        "pointer": "/boundary_conditions/pressure_boundary/*/time_reference",
        "type": "list"
    },
    {
        "doc": "Values of pressure boundary condition for timestep",
        "pointer": "/boundary_conditions/pressure_boundary/*/time_reference/*",
        "type": "float"
    },
    {
        "default": [],
        "doc": "Neumann boundary condition for normal times value for vector-valued PDEs.",
        "pointer": "/boundary_conditions/pressure_cavity",
        "type": "list"
    },
    {
        "default": null,
        "doc": "pressure BC entry",
        "pointer": "/boundary_conditions/pressure_cavity/*",
        "required": [
            "id",
            "value"
        ],
        "type": "object"
    },
    {
        "doc": "ID for the pressure Neumann boundary condition",
        "max": 2147483646,
        "min": 0,
        "pointer": "/boundary_conditions/pressure_cavity/*/id",
        "type": "int"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/boundary_conditions/pressure_cavity/*/value",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/boundary_conditions/pressure_cavity/*/value",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/boundary_conditions/pressure_cavity/*/value",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/boundary_conditions/pressure_cavity/*/value",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/boundary_conditions/pressure_cavity/*/value/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/boundary_conditions/pressure_cavity/*/value/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/boundary_conditions/pressure_cavity/*/value/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/boundary_conditions/pressure_cavity/*/value/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/boundary_conditions/pressure_cavity/*/value/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "The list of boundary conditions for the main variable. Elements of the list are assignment pairs (ID, value) where ID is assigned by surface selection.",
        "pointer": "/boundary_conditions/obstacle_displacements",
        "type": "list"
    },
    {
        "doc": "ID of boundary condition from surface selection.",
        "max": 2147483646,
        "min": 0,
        "pointer": "/boundary_conditions/obstacle_displacements/*/id",
        "type": "int"
    },
    {
        "doc": "select all ids.",
        "options": [
            "all"
        ],
        "pointer": "/boundary_conditions/obstacle_displacements/*/id",
        "type": "string"
    },
    {
        "doc": "Values of boundary condition, length 1 for scalar-valued pde, 2/3 for vector-valued PDEs depending on the dimension.",
        "pointer": "/boundary_conditions/obstacle_displacements/*/value",
        "type": "list"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/boundary_conditions/obstacle_displacements/*/value/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/boundary_conditions/obstacle_displacements/*/value/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/boundary_conditions/obstacle_displacements/*/value/*",
        "required": [
            "value",
            "unit"
        ],
)JSE_JSON";
        text += R"JSE_JSON(        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/boundary_conditions/obstacle_displacements/*/value/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/boundary_conditions/obstacle_displacements/*/value/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/boundary_conditions/obstacle_displacements/*/value/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/boundary_conditions/obstacle_displacements/*/value/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/boundary_conditions/obstacle_displacements/*/value/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/boundary_conditions/obstacle_displacements/*/value/*/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "interpolation of boundary condition",
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation",
        "type": "list"
    },
    {
        "default": {
            "type": "none"
        },
        "doc": "interpolation of boundary condition",
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "none"
    },
    {
        "doc": "interpolation of boundary condition",
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "linear"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "from"
        ],
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*",
        "required": [
            "type",
            "to"
        ],
        "type": "object",
        "type_name": "linear_ramp"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "extend"
        ],
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*",
        "required": [
            "type",
            "points",
            "values"
        ],
        "type": "object",
        "type_name": "piecewise_constant"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "extend"
        ],
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*",
        "required": [
            "type",
            "points",
            "values"
        ],
        "type": "object",
        "type_name": "piecewise_linear"
    },
    {
        "doc": "interpolation of boundary condition",
        "optional": [
            "extend"
        ],
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*",
        "required": [
            "type",
            "points",
            "values"
        ],
        "type": "object",
        "type_name": "piecewise_cubic"
    },
    {
        "doc": "type of interpolation of boundary condition",
        "options": [
            "none",
            "linear",
            "linear_ramp",
            "piecewise_constant",
            "piecewise_linear",
            "piecewise_cubic"
        ],
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*/type",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "interpolation starting time",
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*/from",
        "type": "float"
    },
    {
        "doc": "interpolation ending time",
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*/to",
        "type": "float"
    },
    {
        "doc": "interpolation time points",
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*/points",
        "type": "list"
    },
    {
        "doc": "interpolation time point",
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*/points/*",
        "type": "float"
    },
    {
        "doc": "interpolation values",
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*/values",
        "type": "list"
    },
    {
        "doc": "interpolation value",
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*/values/*",
        "type": "float"
    },
    {
        "default": "constant",
        "doc": "how to extend the piecewise interpolation",
        "options": [
            "constant",
            "extrapolate",
            "repeat",
            "repeat_offset"
        ],
        "pointer": "/boundary_conditions/obstacle_displacements/*/interpolation/*/extend",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Obstacle displacements",
        "optional": [
            "interpolation"
        ],
        "pointer": "/boundary_conditions/obstacle_displacements/*",
        "required": [
            "id",
            "value"
        ],
        "type": "object"
    },
    {
        "default": null,
        "doc": "Initial conditions for the time-dependent problem, imposed on the main variable, its derivative or second derivative",
        "optional": [
            "solution",
            "velocity",
            "acceleration"
        ],
        "pointer": "/initial_conditions",
        "type": "object"
    },
    {
        "default": [],
        "doc": "initial solution",
        "pointer": "/initial_conditions/solution",
        "type": "list"
    },
    {
        "default": null,
        "doc": "A list of (ID, value) pairs defining the initial conditions for the main variable values. Ids are set by selection, and values can be floats or formulas.",
        "pointer": "/initial_conditions/solution/*",
        "required": [
            "id",
            "value"
        ],
        "type": "object"
    },
    {
        "doc": "ID from volume selections",
        "pointer": "/initial_conditions/solution/*/id",
        "type": "int"
    },
    {
        "doc": "value of the solution",
        "pointer": "/initial_conditions/solution/*/value",
        "type": "list"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/initial_conditions/solution/*/value/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/initial_conditions/solution/*/value/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/initial_conditions/solution/*/value/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/initial_conditions/solution/*/value/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/initial_conditions/solution/*/value/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/initial_conditions/solution/*/value/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/initial_conditions/solution/*/value/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/initial_conditions/solution/*/value/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name t)JSE_JSON";
        text += R"JSE_JSON(o evaluate the value",
        "pointer": "/initial_conditions/solution/*/value/*/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "initial velocity",
        "pointer": "/initial_conditions/velocity",
        "type": "list"
    },
    {
        "default": null,
        "doc": "A list of (ID, value) pairs defining the initial conditions for the first derivative of the main variable values. Ids are set by selection, and values can be floats or formulas.",
        "pointer": "/initial_conditions/velocity/*",
        "required": [
            "id",
            "value"
        ],
        "type": "object"
    },
    {
        "doc": "ID from volume selections",
        "pointer": "/initial_conditions/velocity/*/id",
        "type": "int"
    },
    {
        "doc": "value od the initial velocity",
        "max": 3,
        "min": 2,
        "pointer": "/initial_conditions/velocity/*/value",
        "type": "list"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/initial_conditions/velocity/*/value/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/initial_conditions/velocity/*/value/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/initial_conditions/velocity/*/value/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/initial_conditions/velocity/*/value/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/initial_conditions/velocity/*/value/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/initial_conditions/velocity/*/value/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/initial_conditions/velocity/*/value/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/initial_conditions/velocity/*/value/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/initial_conditions/velocity/*/value/*/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "initial acceleration",
        "pointer": "/initial_conditions/acceleration",
        "type": "list"
    },
    {
        "default": null,
        "doc": "entries",
        "pointer": "/initial_conditions/acceleration/*",
        "required": [
            "id",
            "value"
        ],
        "type": "object"
    },
    {
        "doc": "ID from volume selections",
        "pointer": "/initial_conditions/acceleration/*/id",
        "type": "int"
    },
    {
        "doc": "value",
        "max": 3,
        "min": 2,
        "pointer": "/initial_conditions/acceleration/*/value",
        "type": "list"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/initial_conditions/acceleration/*/value/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/initial_conditions/acceleration/*/value/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/initial_conditions/acceleration/*/value/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/initial_conditions/acceleration/*/value/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/initial_conditions/acceleration/*/value/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/initial_conditions/acceleration/*/value/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/initial_conditions/acceleration/*/value/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/initial_conditions/acceleration/*/value/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/initial_conditions/acceleration/*/value/*/function_name",
        "type": "string"
    },
    {
        "default": null,
        "doc": "soft and hard constraints",
        "optional": [
            "soft",
            "hard"
        ],
        "pointer": "/constraints",
        "type": "object"
    },
    {
        "default": [],
        "doc": "list of file containing hard constraints",
        "pointer": "/constraints/hard",
        "type": "list"
    },
    {
        "default": "",
        "doc": "constraint hdf5 file for hard constraint Ax=b. The file must contain these datasets: local2global, dense/sparse matrix A, and vector b. The colums of b nees to be the same as the dimentionality of the problem. if A is sparse it should contain A_triplets/value A_triplets/cols A_triplets/rows A_triplets/shape",
        "pointer": "/constraints/hard/*",
        "type": "string"
    },
    {
        "default": [],
        "doc": "list of file containing soft constraints",
        "pointer": "/constraints/soft",
        "type": "list"
    },
    {
        "default": "",
        "doc": "constraint hdf5 file for soft constraint w||Ax-b||^2. The file must contain these datasets: weight w, local2global, dense/sparse matrix A, and vector b. The colums of b nees to be the same as the dimentionality of the problem. if A is sparse it should contain A_triplets/value A_triplets/cols A_triplets/rows A_triplets/shape",
        "optional": [
            "weight",
            "data"
        ],
        "pointer": "/constraints/soft/*",
        "type": "object"
    },
    {
        "default": 0,
        "doc": "weight",
        "pointer": "/constraints/soft/*/weight",
        "type": "float"
    },
    {
        "default": "",
        "doc": "constraint hdf5 file for soft constraint w||Ax-b||^2. The file must contain these datasets: local2global, dense/sparse matrix A, and vector b. The colums of b nees to be the same as the dimentionality of the problem. if A is sparse it should contain A_triplets/value A_triplets/col A_triplets/rows A_triplets/shape",
        "pointer": "/constraints/soft/*/data",
        "type": "string"
    },
    {
        "default": null,
        "doc": "output settings",
        "optional": [
            "directory",
            "log",
            "json",
            "restart_json",
            "paraview",
            "data",
            "advanced",
            "reference",
            "stats"
        ],
        "pointer": "/output",
        "type": "object"
    },
    {
        "default": "",
        "doc": "Directory for output files.",
        "pointer": "/output/directory",
        "type": "string"
    },
    {
        "default": false,
        "doc": "Saves csv for energy and stats of the non linear solver.",
        "pointer": "/output/stats",
        "type": "bool"
    },
    {
        "default": null,
        "doc": "Setting for the output log.",
        "optional": [
            "level",
            "file_level",
            "path",
            "quiet"
        ],
        "pointer": "/output/log",
        "type": "object"
    },
    {
        "doc": "Level of logging, 0 trace, 1 debug, 2 info, 3 warning, 4 error, 5 critical, and 6 off.",
        "max": 6,
        "min": 0,
        "pointer": "/output/log/level",
        "type": "int"
    )JSE_JSON";
        text += R"JSE_JSON(},
    {
        "default": "debug",
        "doc": "Level of logging.",
        "options": [
            "trace",
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "off"
        ],
        "pointer": "/output/log/level",
        "type": "string"
    },
    {
        "doc": "Level of logging to a file, 0 trace, 1 debug, 2 info, 3 warning, 4 error, 5 critical, and 6 off.",
        "max": 6,
        "min": 0,
        "pointer": "/output/log/file_level",
        "type": "int"
    },
    {
        "default": "trace",
        "doc": "Level of logging.",
        "options": [
            "trace",
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "off"
        ],
        "pointer": "/output/log/file_level",
        "type": "string"
    },
    {
        "default": "",
        "doc": "File where to save the log; empty string is output to terminal.",
        "pointer": "/output/log/path",
        "type": "string"
    },
    {
        "default": false,
        "doc": "Disable cout for logging.",
        "pointer": "/output/log/quiet",
        "type": "bool"
    },
    {
        "default": "",
        "doc": "File name for JSON output statistics on time/error/etc.",
        "pointer": "/output/json",
        "type": "string"
    },
    {
        "default": "",
        "doc": "File name for JSON output to restart the simulation.",
        "pointer": "/output/restart_json",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Output in paraview format",
        "optional": [
            "file_name",
            "vismesh_rel_area",
            "skip_frame",
            "high_order_mesh",
            "volume",
            "surface",
            "wireframe",
            "fields",
            "points",
            "options"
        ],
        "pointer": "/output/paraview",
        "type": "object"
    },
    {
        "default": "",
        "doc": "Paraview output file name",
        "pointer": "/output/paraview/file_name",
        "type": "string"
    },
    {
        "default": 1e-05,
        "doc": "relative area for the upsampled visualisation mesh",
        "pointer": "/output/paraview/vismesh_rel_area",
        "type": "float"
    },
    {
        "default": 1,
        "doc": "export every skip_frame-th frames for time dependent simulations",
        "pointer": "/output/paraview/skip_frame",
        "type": "int"
    },
    {
        "default": true,
        "doc": "Enables/disables high-order output for paraview. Supported only for isoparametric or linear meshes with high-order solutions.",
        "pointer": "/output/paraview/high_order_mesh",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "Export volumetric mesh",
        "pointer": "/output/paraview/volume",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "Export surface mesh (in 2d polygon)",
        "pointer": "/output/paraview/surface",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "Export the wireframe of the mesh",
        "pointer": "/output/paraview/wireframe",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "Export the Dirichlet points",
        "pointer": "/output/paraview/points",
        "type": "bool"
    },
    {
        "default": [],
        "doc": "list of names of fields to export. If empty, all fields are exported.",
        "pointer": "/output/paraview/fields",
        "type": "list"
    },
    {
        "default": "",
        "doc": "field name",
        "pointer": "/output/paraview/fields/*",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Optional fields in the output",
        "optional": [
            "use_hdf5",
            "material",
            "body_ids",
            "contact_forces",
            "friction_forces",
            "normal_adhesion_forces",
            "tangential_adhesion_forces",
            "velocity",
            "acceleration",
            "scalar_values",
            "tensor_values",
            "discretization_order",
            "nodes",
            "forces",
            "force_high_order",
            "jacobian_validity"
        ],
        "pointer": "/output/paraview/options",
        "type": "object"
    },
    {
        "default": false,
        "doc": "If true, export the data as hdf5, compatible with paraview >5.11",
        "pointer": "/output/paraview/options/use_hdf5",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "If true, write out material values sampled on the vertices of the mesh",
        "pointer": "/output/paraview/options/material",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "Export volumes ids",
        "pointer": "/output/paraview/options/body_ids",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "If true, write out contact forces for surface",
        "pointer": "/output/paraview/options/contact_forces",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "If true, write out friction forces for surface",
        "pointer": "/output/paraview/options/friction_forces",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "If true, write out normal adhesion forces for surface",
        "pointer": "/output/paraview/options/normal_adhesion_forces",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "If true, write out tangential adhesion forces for surface",
        "pointer": "/output/paraview/options/tangential_adhesion_forces",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "If true, write out velocities",
        "pointer": "/output/paraview/options/velocity",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "If true, write out accelerations",
        "pointer": "/output/paraview/options/acceleration",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "If true, write out scalar values",
        "pointer": "/output/paraview/options/scalar_values",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "If true, write out tensor values",
        "pointer": "/output/paraview/options/tensor_values",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "If true, write out discretization order",
        "pointer": "/output/paraview/options/discretization_order",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "If true, write out node order",
        "pointer": "/output/paraview/options/nodes",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "If true, write out all variational forces on the FE mesh",
        "pointer": "/output/paraview/options/forces",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "If true, force write out high-order mesh, might break the output",
        "pointer": "/output/paraview/options/force_high_order",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "If true, perform robust Jacobian check on the deformed elements and mark elements with non-positive Jacobian.",
        "pointer": "/output/paraview/options/jacobian_validity",
        "type": "bool"
    },
    {
        "default": null,
        "doc": "File names to write output data to.",
        "optional": [
            "solution",
            "full_mat",
            "stiffness_mat",
            "stress_mat",
            "state",
            "rest_mesh",
            "mises",
            "nodes",
            "advanced",
            "file_index_offset"
        ],
        "pointer": "/output/data",
        "type": "object"
    },
    {
        "default": "",
        "doc": "Main variable solution. Unrolled [xyz, xyz, ...] using PolyFEM ordering. If reorder_n)JSE_JSON";
        text += R"JSE_JSON(odes exports the solution with the same order the vertices of the input mesh as a #n x d file",
        "pointer": "/output/data/solution",
        "type": "string"
    },
    {
        "default": "",
        "doc": "System matrix without boundary conditions. Doesn't work for nonlinear problems",
        "pointer": "/output/data/full_mat",
        "type": "string"
    },
    {
        "default": "",
        "doc": "System matrix with boundary conditions. Doesn't work for nonlinear problems",
        "pointer": "/output/data/stiffness_mat",
        "type": "string"
    },
    {
        "default": "",
        "doc": "Exports stress",
        "pointer": "/output/data/stress_mat",
        "type": "string"
    },
    {
        "default": "",
        "doc": "Writes the complete state in PolyFEM hdf5 format, used to restart the sim",
        "pointer": "/output/data/state",
        "type": "string"
    },
    {
        "default": "",
        "doc": "Writes the rest mesh in MSH format, used to restart the sim",
        "pointer": "/output/data/rest_mesh",
        "type": "string"
    },
    {
        "default": "",
        "doc": "File name to write per-node Von Mises stress values to.",
        "pointer": "/output/data/mises",
        "type": "string"
    },
    {
        "default": "",
        "doc": "Writes the FEM nodes",
        "pointer": "/output/data/nodes",
        "type": "string"
    },
    {
        "default": null,
        "doc": "advanced options",
        "optional": [
            "reorder_nodes"
        ],
        "pointer": "/output/data/advanced",
        "type": "object"
    },
    {
        "default": false,
        "doc": "Reorder nodes accodring to input",
        "pointer": "/output/data/advanced/reorder_nodes",
        "type": "bool"
    },
    {
        "default": 0,
        "doc": "Starting file index offset for output files. Set automatically by restart JSON so that file numbering continues from the previous run.",
        "pointer": "/output/data/file_index_offset",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Write out the analytic/numerical ground-truth solution and or its gradient",
        "optional": [
            "solution",
            "gradient"
        ],
        "pointer": "/output/reference",
        "type": "object"
    },
    {
        "default": [],
        "doc": "reference solution used to compute errors",
        "pointer": "/output/reference/solution",
        "type": "list"
    },
    {
        "default": "",
        "doc": "value as a function of $x,y,z,t$",
        "pointer": "/output/reference/solution/*",
        "type": "string"
    },
    {
        "default": [],
        "doc": "gradient of the reference solution to compute errors",
        "pointer": "/output/reference/gradient",
        "type": "list"
    },
    {
        "default": "",
        "doc": "value as a function of $x,y,z,t$",
        "pointer": "/output/reference/gradient/*",
        "type": "string"
    },
    {
        "default": null,
        "doc": "Additional output options",
        "optional": [
            "timestep_prefix",
            "sol_on_grid",
            "compute_error",
            "sol_at_node",
            "vis_boundary_only",
            "curved_mesh_size",
            "save_solve_sequence_debug",
            "save_ccd_debug_meshes",
            "save_time_sequence",
            "save_nl_solve_sequence",
            "spectrum"
        ],
        "pointer": "/output/advanced",
        "type": "object"
    },
    {
        "default": "step_",
        "doc": "Prefix for output file names for each time step, the final file is step_i.[vtu|vtm] where i is the time index.",
        "pointer": "/output/advanced/timestep_prefix",
        "type": "string"
    },
    {
        "default": -1,
        "doc": "exports the solution sampled on a grid, specify the grid spacing",
        "pointer": "/output/advanced/sol_on_grid",
        "type": "float"
    },
    {
        "default": true,
        "doc": "Enables the computation of the error. If no reference solution is provided, return the norms of the solution",
        "pointer": "/output/advanced/compute_error",
        "type": "bool"
    },
    {
        "default": -1,
        "doc": "Write out solution values at a specific node. the values will be written in the output JSON file",
        "pointer": "/output/advanced/sol_at_node",
        "type": "int"
    },
    {
        "default": false,
        "doc": "saves only elements touching the boundaries",
        "pointer": "/output/advanced/vis_boundary_only",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "upsample curved edges to compute mesh size",
        "pointer": "/output/advanced/curved_mesh_size",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "saves AL internal steps, for debugging",
        "pointer": "/output/advanced/save_solve_sequence_debug",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "saves AL internal steps, for debugging",
        "pointer": "/output/advanced/save_ccd_debug_meshes",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "saves timesteps",
        "pointer": "/output/advanced/save_time_sequence",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "saves obj after every nonlinear iteration, for debugging",
        "pointer": "/output/advanced/save_nl_solve_sequence",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "exports the spectrum of the matrix in the output JSON. Works only if POLYSOLVE_WITH_SPECTRA is enabled",
        "pointer": "/output/advanced/spectrum",
        "type": "bool"
    },
    {
        "default": null,
        "doc": "input data",
        "optional": [
            "data"
        ],
        "pointer": "/input",
        "type": "object"
    },
    {
        "default": null,
        "doc": "input to restart time dependent sim",
        "optional": [
            "state",
            "reorder"
        ],
        "pointer": "/input/data",
        "type": "object"
    },
    {
        "default": "",
        "doc": "input state as hdf5",
        "pointer": "/input/data/state",
        "type": "file"
    },
    {
        "default": false,
        "doc": "reorder input data",
        "pointer": "/input/data/reorder",
        "type": "bool"
    },
    {
        "default": "skip",
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Linear"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Quadratic"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Cubic"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Sine"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Franke"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "FrankeOld"
    },
    {
        "doc": "TODO",
        "optional": [
            "func"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "GenericScalarExact"
    },
    {
        "default": 0,
        "doc": "TODO",
        "pointer": "/preset_problem/func",
        "type": "int"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required")JSE_JSON";
        text += R"JSE_JSON(: [
            "type"
        ],
        "type": "object",
        "type_name": "Zero_BC"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Elastic"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Walk"
    },
    {
        "doc": "TODO",
        "optional": [
            "axis_coordiante",
            "n_turns",
            "fixed_boundary",
            "turning_boundary",
            "bbox_center"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "TorsionElastic"
    },
    {
        "default": 2,
        "doc": "TODO",
        "pointer": "/preset_problem/axis_coordiante",
        "type": "int"
    },
    {
        "default": 0.5,
        "doc": "TODO",
        "pointer": "/preset_problem/n_turns",
        "type": "float"
    },
    {
        "default": 5,
        "doc": "TODO",
        "pointer": "/preset_problem/fixed_boundary",
        "type": "int"
    },
    {
        "default": 6,
        "doc": "TODO",
        "pointer": "/preset_problem/turning_boundary",
        "type": "int"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/preset_problem/bbox_center",
        "type": "list"
    },
    {
        "default": 0,
        "doc": "TODO",
        "pointer": "/preset_problem/bbox_center/*",
        "type": "float"
    },
    {
        "doc": "TODO",
        "optional": [
            "axis_coordiante0",
            "axis_coordiante1",
            "angular_v0",
            "angular_v1",
            "turning_boundary0",
            "turning_boundary1",
            "bbox_center"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DoubleTorsionElastic"
    },
    {
        "default": 2,
        "doc": "TODO",
        "pointer": "/preset_problem/axis_coordiante0",
        "type": "int"
    },
    {
        "default": 2,
        "doc": "TODO",
        "pointer": "/preset_problem/axis_coordiante1",
        "type": "int"
    },
    {
        "default": 0.5,
        "doc": "TODO",
        "pointer": "/preset_problem/angular_v0",
        "type": "float"
    },
    {
        "default": -0.5,
        "doc": "TODO",
        "pointer": "/preset_problem/angular_v1",
        "type": "float"
    },
    {
        "default": 5,
        "doc": "TODO",
        "pointer": "/preset_problem/turning_boundary0",
        "type": "int"
    },
    {
        "default": 6,
        "doc": "TODO",
        "pointer": "/preset_problem/turning_boundary1",
        "type": "int"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "ElasticZeroBC"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "ElasticExact"
    },
    {
        "doc": "TODO, add displacement, E, nu, formulation, mesh_size",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "ElasticCantileverExact"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "CompressionElasticExact"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "QuadraticElasticExact"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "LinearElasticExact"
    },
    {
        "doc": "TODO, add optionals",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "PointBasedTensor"
    },
    {
        "doc": "TODO, add optionals",
        "optional": [
            "formulation",
            "n_kernels",
            "kernel_distance",
            "kernel_weights"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Kernel"
    },
    {
        "default": "",
        "doc": "TODO",
        "pointer": "/preset_problem/formulation",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "TODO",
        "pointer": "/preset_problem/n_kernels",
        "type": "int"
    },
    {
        "default": 0,
        "doc": "TODO",
        "pointer": "/preset_problem/kernel_distance",
        "type": "float"
    },
    {
        "default": "",
        "doc": "TODO",
        "pointer": "/preset_problem/kernel_weights",
        "type": "string"
    },
    {
        "doc": "TODO, add optionals",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Node"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "TimeDependentScalar"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "MinSurf"
    },
    {
        "doc": "TODO",
        "optional": [
            "force"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Gravity"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/preset_problem/n_kernels/force",
        "type": "list"
    },
    {
        "default": 0,
        "doc": "TODO",
        "pointer": "/preset_problem/n_kernels/force/*",
        "type": "float"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "ConstantVelocity"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "TwoSpheres"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DrivenCavity"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DrivenCavityC0"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DrivenCavitySmooth"
    },
    {
        "doc": "TODO, add inflow, outflow, inflow_amout, outflow_amout, direction, obstacle",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Flow"
    },
    {
        "doc": "TODO",
        "optional": [
            "U"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "FlowWithObstacle"
    },
    {
        "default": 0,
        "doc": "TODO",
        "pointer": "/preset_problem/n_kernels/U",
        "type": "float"
    },
    {
        "default": false,
        "doc": "TODO",
        "pointer": "/preset_problem/n_kernels/time_dependent",
        "type": "bool"
    },
    {
        "doc": "TODO")JSE_JSON";
        text += R"JSE_JSON(,
        "optional": [
            "U",
            "time_dependent"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "CornerFlow"
    },
    {
        "doc": "TODO, add inflow_id, direction, no_slip",
        "optional": [
            "U"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "UnitFlowWithObstacle"
    },
    {
        "doc": "TODO, add radius",
        "optional": [
            "time_dependent",
            "viscosity"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "StokesLaw"
    },
    {
        "default": 0,
        "doc": "TODO",
        "pointer": "/preset_problem/n_kernels/viscosity",
        "type": "float"
    },
    {
        "doc": "TODO",
        "optional": [
            "viscosity"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "TaylorGreenVortex"
    },
    {
        "default": 1,
        "doc": "TODO",
        "pointer": "/preset_problem/viscosity",
        "type": "float"
    },
    {
        "doc": "TODO",
        "optional": [
            "func"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "SimpleStokeProblemExact"
    },
    {
        "doc": "TODO",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "SineStokeProblemExact"
    },
    {
        "doc": "TODO",
        "optional": [
            "func",
            "viscosity"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "TransientStokeProblemExact"
    },
    {
        "doc": "TODO",
        "optional": [
            "viscosity"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Kovnaszy"
    },
    {
        "doc": "TODO",
        "optional": [
            "time_dependent"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Airfoil"
    },
    {
        "doc": "TODO",
        "optional": [
            "U",
            "time_dependent"
        ],
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Lshape"
    },
    {
        "doc": "TODO, type, omega, is_scalar",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "TestProblem"
    },
    {
        "doc": "TODO, type, omega, is_scalar",
        "pointer": "/preset_problem",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "BilaplacianProblemWithSolution"
    },
    {
        "doc": "Type of preset problem to use.",
        "options": [
            "Linear",
            "Quadratic",
            "Cubic",
            "Sine",
            "Franke",
            "FrankeOld",
            "GenericScalarExact",
            "Zero_BC",
            "Elastic",
            "Walk",
            "TorsionElastic",
            "DoubleTorsionElastic",
            "ElasticZeroBC",
            "ElasticExact",
            "ElasticCantileverExact",
            "CompressionElasticExact",
            "QuadraticElasticExact",
            "LinearElasticExact",
            "PointBasedTensor",
            "Kernel",
            "Node",
            "TimeDependentScalar",
            "MinSurf",
            "Gravity",
            "ConstantVelocity",
            "TwoSpheres",
            "DrivenCavity",
            "DrivenCavityC0",
            "DrivenCavitySmooth",
            "Flow",
            "FlowWithObstacle",
            "CornerFlow",
            "UnitFlowWithObstacle",
            "StokesLaw",
            "TaylorGreenVortex",
            "SimpleStokeProblemExact",
            "SineStokeProblemExact",
            "TransientStokeProblemExact",
            "Kovnaszy",
            "Airfoil",
            "Lshape",
            "TestProblem",
            "BilaplacianProblemWithSolution"
        ],
        "pointer": "/preset_problem/type",
        "type": "string"
    }
]
)JSE_JSON";
        return nlohmann::json::parse(text);
    }();
    return value;
}

} // namespace polyfem
} // namespace embed
} // namespace jse
