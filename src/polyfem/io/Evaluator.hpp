#pragma once

#include <Eigen/Core>

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/MeshNodes.hpp>

#include <polyfem/utils/RefElementSampler.hpp>

namespace polyfem::io
{
	class Evaluator
	{
	private:
		Evaluator() {};

	public:
		/// compute von mises stress at quadrature points for the function fun, also compute the interpolated function
		/// @param[in] mesh mesh
		/// @param[in] is_problem_scalar if problem is scalar
		/// @param[in] bases bases
		/// @param[in] gbases geom bases
		/// @param[in] disc_orders discretization orders
		/// @param[in] assembler assembler
		/// @param[in] fun function to use
		/// @param[out] result output displacement
		/// @param[out] von_mises output von mises
		static void compute_stress_at_quadrature_points(
			const mesh::Mesh &mesh,
			const bool is_problem_scalar,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const Eigen::VectorXi &disc_orders,
			const assembler::Assembler &assembler,
			const Eigen::MatrixXd &fun,
			const double t,
			Eigen::MatrixXd &result,
			Eigen::VectorXd &von_mises);

		/// interpolate the function fun.
		/// @param[in] mesh mesh
		/// @param[in] is_problem_scalar if problem is scalar
		/// @param[in] bases bases
		/// @param[in] disc_orders discretization orders
		/// @param[in] polys polygons
		/// @param[in] polys_3d polyhedra
		/// @param[in] sampler sampler for the local element
		/// @param[in] n_points is the size of the output.
		/// @param[in] fun function to used
		/// @param[out] result output
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		static void interpolate_function(
			const mesh::Mesh &mesh,
			const bool is_problem_scalar,
			const std::vector<basis::ElementBases> &bases,
			const Eigen::VectorXi &disc_orders,
			const std::map<int, Eigen::MatrixXd> &polys,
			const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
			const utils::RefElementSampler &sampler,
			const int n_points,
			const Eigen::MatrixXd &fun,
			Eigen::MatrixXd &result,
			const bool use_sampler,
			const bool boundary_only);

		static void mark_flipped_cells(
			const mesh::Mesh &mesh,
			const std::vector<basis::ElementBases> &gbasis,
			const std::vector<basis::ElementBases> &basis,
			const Eigen::VectorXi &disc_orders,
			const std::map<int, Eigen::MatrixXd> &polys,
			const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
			const utils::RefElementSampler &sampler,
			const int n_points,
			const Eigen::MatrixXd &fun,
			Eigen::Vector<bool, -1> &result,
			const bool use_sampler,
			const bool boundary_only);

		/// interpolate the function fun.
		/// @param[in] mesh mesh
		/// @param[in] actual_dim is the size of the problem (e.g., 1 for Laplace, dim for elasticity)
		/// @param[in] bases bases
		/// @param[in] disc_orders discretization orders
		/// @param[in] polys polygons
		/// @param[in] polys_3d polyhedra
		/// @param[in] sampler sampler for the local element
		/// @param[in] n_points is the size of the output.
		/// @param[in] fun function to used
		/// @param[out] result output
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		static void interpolate_function(
			const mesh::Mesh &mesh,
			const int actual_dim,
			const std::vector<basis::ElementBases> &basis,
			const Eigen::VectorXi &disc_orders,
			const std::map<int, Eigen::MatrixXd> &polys,
			const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
			const utils::RefElementSampler &sampler,
			const int n_points,
			const Eigen::MatrixXd &fun,
			Eigen::MatrixXd &result,
			const bool use_sampler,
			const bool boundary_only);

		/// interpolate solution and gradient at element (calls interpolate_at_local_vals with sol)
		/// @param[in] mesh mesh
		/// @param[in] is_problem_scalar if problem is scalar
		/// @param[in] bases bases
		/// @param[in] gbases geom bases
		/// @param[in] el_index element index
		/// @param[in] local_pts points in the reference element
		/// @param[in] fun function to use
		/// @param[out] result output
		/// @param[out] result_grad output gradients
		static void interpolate_at_local_vals(
			const mesh::Mesh &mesh,
			const bool is_problem_scalar,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const int el_index,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &fun,
			Eigen::MatrixXd &result,
			Eigen::MatrixXd &result_grad);

		/// interpolate the function fun and its gradient at in element el_index for the local_pts in the reference element using bases bases
		/// interpolate solution and gradient at element (calls interpolate_at_local_vals with sol)
		/// @param[in] mesh mesh
		/// @param[in] actual_dim is the size of the problem (e.g., 1 for Laplace, dim for elasticity)
		/// @param[in] bases bases
		/// @param[in] gbases geom bases
		/// @param[in] el_index element index
		/// @param[in] local_pts points in the reference element
		/// @param[in] fun function to used
		/// @param[out] result output
		/// @param[out] result_grad output gradients
		static void interpolate_at_local_vals(
			const mesh::Mesh &mesh,
			const int actual_dim,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const int el_index,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &fun,
			Eigen::MatrixXd &result,
			Eigen::MatrixXd &result_grad);

		static void interpolate_at_local_vals(
			const int el_index,
			const int dim,
			const int actual_dim,
			const assembler::ElementAssemblyValues &vals,
			const Eigen::MatrixXd &fun,
			Eigen::MatrixXd &result,
			Eigen::MatrixXd &result_grad);

		/// checks if mises are not nan
		/// @param[in] mesh mesh
		/// @param[in] is_problem_scalar if problem is scalar
		/// @param[in] bases bases
		/// @param[in] gbases geom bases
		/// @param[in] disc_orders discretization orders
		/// @param[in] polys polygons
		/// @param[in] polys_3d polyhedra
		/// @param[in] assembler assembler
		/// @param[in] sampler sampler for the local element
		/// @param[in] fun function to used
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		/// @return if mises are nan
		bool check_scalar_value(
			const mesh::Mesh &mesh,
			const bool is_problem_scalar,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const Eigen::VectorXi &disc_orders,
			const std::map<int, Eigen::MatrixXd> &polys,
			const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
			const assembler::Assembler &assembler,
			const utils::RefElementSampler &sampler,
			const Eigen::MatrixXd &fun,
			const double t,
			const bool use_sampler,
			const bool boundary_only);

		/// computes scalar quantity of funtion (ie von mises for elasticity and norm of velocity for fluid)
		/// @param[in] mesh mesh
		/// @param[in] is_problem_scalar if problem is scalar
		/// @param[in] bases bases
		/// @param[in] gbases geom bases
		/// @param[in] disc_orders discretization orders
		/// @param[in] polys polygons
		/// @param[in] polys_3d polyhedra
		/// @param[in] assembler assembler
		/// @param[in] sampler sampler for the local element
		/// @param[in] n_points is the size of the output.
		/// @param[in] fun function to used
		/// @param[out] result scalar value
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		static void compute_scalar_value(
			const mesh::Mesh &mesh,
			const bool is_problem_scalar,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const Eigen::VectorXi &disc_orders,
			const std::map<int, Eigen::MatrixXd> &polys,
			const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
			const assembler::Assembler &assembler,
			const utils::RefElementSampler &sampler,
			const int n_points,
			const Eigen::MatrixXd &fun,
			const double t,
			std::vector<assembler::Assembler::NamedMatrix> &result,
			const bool use_sampler,
			const bool boundary_only);

		/// computes scalar quantity of funtion (ie von mises for elasticity and norm of velocity for fluid)
		/// the scalar value is averaged around every node to make it continuos
		/// @param[in] mesh mesh
		/// @param[in] is_problem_scalar if problem is scalar
		/// @param[in] n_bases number of bases
		/// @param[in] bases bases
		/// @param[in] gbases geom bases
		/// @param[in] disc_orders discretization orders
		/// @param[in] polys polygons
		/// @param[in] polys_3d polyhedra
		/// @param[in] assembler assembler
		/// @param[in] sampler sampler for the local element
		/// @param[in] n_points is the size of the output.
		/// @param[in] fun function to used
		/// @param[out] result_scalar scalar value
		/// @param[out] result_tensor tensor value
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		static void average_grad_based_function(
			const mesh::Mesh &mesh,
			const bool is_problem_scalar,
			const int n_bases,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const Eigen::VectorXi &disc_orders,
			const std::map<int, Eigen::MatrixXd> &polys,
			const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
			const assembler::Assembler &assembler,
			const utils::RefElementSampler &sampler,
			const double t,
			const int n_points,
			const Eigen::MatrixXd &fun,
			std::vector<assembler::Assembler::NamedMatrix> &result_scalar,
			std::vector<assembler::Assembler::NamedMatrix> &result_tensor,
			const bool use_sampler,
			const bool boundary_only);

		/// compute tensor quantity (ie stress tensor or velocy)
		/// @param[in] mesh mesh
		/// @param[in] is_problem_scalar if problem is scalar
		/// @param[in] bases bases
		/// @param[in] gbases geom bases
		/// @param[in] disc_orders discretization orders
		/// @param[in] polys polygons
		/// @param[in] polys_3d polyhedra
		/// @param[in] assembler assembler
		/// @param[in] sampler sampler for the local element
		/// @param[in] n_points is the size of the output.
		/// @param[in] fun function to used
		/// @param[out] result resulting tensor
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		static void compute_tensor_value(
			const mesh::Mesh &mesh,
			const bool is_problem_scalar,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const Eigen::VectorXi &disc_orders,
			const std::map<int, Eigen::MatrixXd> &polys,
			const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
			const assembler::Assembler &assembler,
			const utils::RefElementSampler &sampler,
			const int n_points,
			const Eigen::MatrixXd &fun,
			const double t,
			std::vector<assembler::Assembler::NamedMatrix> &result,
			const bool use_sampler,
			const bool boundary_only);

		/// computes integrated solution (fun) per surface face. pts and faces are the boundary are the boundary on the rest configuration
		/// @param[in] mesh mesh
		/// @param[in] is_problem_scalar if problem is scalar
		/// @param[in] bases bases
		/// @param[in] gbases geom bases
		/// @param[in] pts boundary points
		/// @param[in] faces boundary faces
		/// @param[in] fun function to used
		/// @param[in] compute_avg if compute the average across elements
		/// @param[out] result resulting value
		static void interpolate_boundary_function(
			const mesh::Mesh &mesh,
			const bool is_problem_scalar,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const Eigen::MatrixXd &pts,
			const Eigen::MatrixXi &faces,
			const Eigen::MatrixXd &fun,
			const bool compute_avg,
			Eigen::MatrixXd &result);

		static Eigen::MatrixXd generate_linear_field(
			const int n_bases,
			const std::shared_ptr<mesh::MeshNodes> mesh_nodes,
			const Eigen::MatrixXd &grad);

		static Eigen::MatrixXd get_bases_position(
			const int n_bases,
			const std::shared_ptr<mesh::MeshNodes> mesh_nodes);

		static Eigen::VectorXd integrate_function(
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const Eigen::MatrixXd &fun,
			const int dim,
			const int actual_dim);
	};
} // namespace polyfem::io