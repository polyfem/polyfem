#include "Spline2dBasis.hpp"

#include "QuadraticTPBSpline.hpp"

#include <cassert>
#include <iostream>
#include <vector>

namespace poly_fem
{

	using namespace Eigen;
	// Spline2dBasis::Spline2dBasis(const int global_index, const Eigen::MatrixXd &node, const std::vector<double> &knots_u, const std::vector<double> &knots_v)
	// : Basis(global_index, node)
	// {
	// 	spline_.init(knots_u, knots_v);
	// }



	void print_local_space(const Matrix3i &space)
	{
		for(int j=2; j >=0; --j)
		{
			for(int i=0; i < 3; ++i)
			{
				std::cout<<space(i,j)<<" ";
			}
			std::cout<<std::endl;
		}
	}

	int min_v(const int i1, const int i2, const int i3)
	{
		using std::min;
		return min(i1, min(i2, i3));
	}

	int min_v(const int i1, const int i2)
	{
		using std::min;
		return min(i1, i2);
	}


	void get_neighs(const Mesh &mesh, const int el_index,  Matrix3i &neighs, const int c)
	{
		assert(!mesh.is_volume);

		neighs.setConstant(c);

		auto el = mesh.els.row(el_index);

		const int n_x = mesh.n_x;
		const int n_y = mesh.n_y;

		const bool has_left = el(0) % (n_x + 1) != 0;
		const bool has_right = el(2) % (n_x + 1) != n_x;

		const bool has_bottom = el(0) / (n_x + 1) != 0;
		const bool has_top = el(2) / (n_x + 1) != n_y;

		neighs(1,1) = el_index;

		if(has_left)
			neighs(0, 1) = el_index - 1;
		if(has_right)
			neighs(2, 1) = el_index + 1;

		if(has_bottom)
			neighs(1, 0) = el_index - n_x;
		if(has_top)
			neighs(1, 2) = el_index + n_x;

		if(has_bottom && has_left)
			neighs(0, 0) = el_index - n_x - 1;
		if(has_bottom && has_right)
			neighs(2, 0) = el_index - n_x + 1;

		if(has_top && has_left)
			neighs(0, 2) = el_index + n_x - 1;
		if(has_top && has_right)
			neighs(2, 2) = el_index + n_x + 1;
	}


	int build_local_parameterization_space(const Mesh &mesh, std::vector<Matrix3i> &spaces)
	{
		spaces.resize(mesh.els.rows());

		int boundary_bases = 0;
		const int n_els = int(mesh.els.rows());
		const int c = n_els*10;

		for(long e = 0; e < mesh.els.rows(); ++e)
		{
			Matrix3i &space = spaces[e];

			get_neighs(mesh, int(e), space, c);

			const int top    = space(1,2);
			const int bottom = space(1,0);

			const int left  = space(0,1);
			const int right = space(2,1);

        	if(space(0,2) == c) //1
        	{
        		const int index = min_v(left, top);
        		if(index == c || e < index)
        			space(0,2) = n_els + boundary_bases++;
        		else if(index == left)
        			space(0,2) = spaces[index](1,2);
            	else //top
            		space(0,2) = spaces[index](0,1);
            }

        	if(space(1,2) == c) //2
        	{
        		const int index = min_v(left, right, top);
        		if(index == c || e < index)
        			space(1,2) = n_els + boundary_bases++;
        		else if(index == left)
        			space(1,2) = spaces[index](2,2);
        		else if(index == right)
        			space(1,2) = spaces[index](0,2);
            	else //top
            		space(1,2) = spaces[index](1,1);
            }

        	if(space(2,2) == c) //3
        	{
        		const int index = min_v(right, top);
        		if(index == c || e < index)
        			space(2,2) = n_els + boundary_bases++;
        		else if(index == right)
        			space(2,2) = spaces[index](1,2);
            	else //top
            		space(2,2) = spaces[index](2,1);
            }

        	if(space(2,1) == c) //4
        	{
        		const int index = min_v(right, top, bottom);
        		if(index == c || e < index)
        			space(2,1) = n_els + boundary_bases++;
        		else if(index == right)
        			space(2,1) = spaces[index](1,1);
        		else if(index == top)
        			space(2,1) = spaces[index](2,0);
            	else //bottom
            		space(2,1) = spaces[index](2,2);
            }

        	if(space(2,0) == c) //5
        	{
        		const int index = min_v(right, bottom);
        		if(index == c || e < index)
        			space(2,0) = n_els + boundary_bases++;
        		else if(index == right)
        			space(2,0) = spaces[index](1,0);
            	else //bottom
            		space(2,0) = spaces[index](2,1);
            }

        	if(space(1,0) == c) //6
        	{
        		const int index = min_v(left, right, bottom);
        		if(index == c || e < index)
        			space(1,0) = n_els + boundary_bases++;
        		else if(index == left)
        			space(1,0) = spaces[index](2,0);
        		else if(index == right)
        			space(1,0) = spaces[index](0,0);
            	else //bottom
            		space(1,0) = spaces[index](1,1);
            }

        	if(space(0,0) == c) //7
        	{
        		const int index = min_v(left, bottom);
        		if(index == c || e < index)
        			space(0,0) = n_els + boundary_bases++;
        		else if(index == left)
        			space(0,0) = spaces[index](1,0);
            	else //bottom
            		space(0,0) = spaces[index](0,1);
            }

        	if(space(0,1) == c) //8
        	{
        		const int index = min_v(left, top, bottom);
        		if(index == c || e < index)
        			space(0,1) = n_els + boundary_bases++;
        		else if(index == left)
        			space(0,1) = spaces[index](1,1);
        		else if(index == top)
        			space(0,1) = spaces[index](0,0);
            	else //bottom
            		space(0,1) = spaces[index](0,2);
            }
        // print_local_space(space);
        // std::cout<<"\n\n"<<std::endl;

            assert(space.maxCoeff() < c);
        }

        return boundary_bases;
    }

    template<typename T>
    void get_node(const int x, const int y, const std::vector<std::vector<double> > &h_knots, const std::vector<std::vector<double> > &v_knots, const T &p1, const T &p2, const T &p4, Eigen::Vector2d &node)
    {
    	if(h_knots[x][0] == h_knots[x][2])
    		node(0) = p1(0);
    	else if(h_knots[x][1] == h_knots[x][3])
    		node(0) = p2(0);

    	else if(h_knots[x][0] == h_knots[x][1] && h_knots[x][2] == h_knots[x][3])
    		node(0) = (p1(0)+p2(0))/2;

    	else if(h_knots[x][0] == h_knots[x][1])
    		node(0) = (p1(0)+p2(0))/2;
    	else if(h_knots[x][2] == h_knots[x][3])
    		node(0) = (p1(0)+p2(0))/2;
    	else
    		node(0) = (p1(0)+p2(0))/2;

    	if(v_knots[y][0] == v_knots[y][2])
    		node(1) = p1(1);
    	else if(v_knots[y][1] == v_knots[y][3])
    		node(1) = p4(1);

    	else if(v_knots[y][0] == v_knots[y][1] && v_knots[y][2] == v_knots[y][3])
    		node(1) = (p1(1)+p4(1))/2;

    	else if(v_knots[y][0] == v_knots[y][1])
    		node(1) = (p1(1)+p4(1))/2;
    	else if(v_knots[y][2] == v_knots[y][3])
    		node(1) = (p1(1)+p4(1))/2;
    	else
    		node(1) = (p1(1)+p4(1))/2;
    }


    int Spline2dBasis::build_bases(const Mesh &mesh, std::vector< std::vector<Basis> > &bases, std::vector< int > &bounday_nodes)
    {
    	assert(!mesh.is_volume);

    	bases.resize(mesh.els.rows());

    	const int n_els = int(mesh.els.rows());

    	bounday_nodes.clear();

    	std::vector<Matrix3i> spaces;
    	const int boundary_bases = build_local_parameterization_space(mesh, spaces);

    	for(long e = 0; e < mesh.els.rows(); ++e)
    	{
    		const Matrix3i &space = spaces[e];

    		std::vector<Basis> &b=bases[e];
    		b.resize(9);

    		std::vector<std::vector<double> > h_knots(3);
    		std::vector<std::vector<double> > v_knots(3);

        	if(space(0,1) >= n_els && space(2,1) >= n_els) //left and right neigh are absent
        	{
        		h_knots[0] = {0, 0, 0, 1};
        		h_knots[1] = {0, 0, 1, 1};
        		h_knots[2] = {0, 1, 1, 1};
        	}
        	else if(space(0,1) >= n_els) //left neigh is absent
        	{
        		h_knots[0] = {0, 0, 0, 1};
        		h_knots[1] = {0, 0, 1, 2};
        		h_knots[2] = {0, 1, 2, 3};
        	}
        	else if(space(2,1) >= n_els) //right neigh is absent
        	{
        		h_knots[0] = {-2, -1, 0, 1};
        		h_knots[1] = {-1, 0, 1, 1};
        		h_knots[2] = {0, 1, 1, 1};
        	}
        	else
        	{
        		h_knots[0] = {-2, -1, 0, 1};
        		h_knots[1] = {-1, 0, 1, 2};
        		h_knots[2] = {0, 1, 2, 3};
        	}

        	if(space(1,0) >= n_els && space(1,2) >= n_els) //top and bottom neigh are absent
        	{
        		v_knots[0] = {0, 0, 0, 1};
        		v_knots[1] = {0, 0, 1, 1};
        		v_knots[2] = {0, 1, 1, 1};
        	}
        	else if(space(1,0) >= n_els) //bottom neigh is absent
        	{
        		v_knots[0] = {0, 0, 0, 1};
        		v_knots[1] = {0, 0, 1, 2};
        		v_knots[2] = {0, 1, 2, 3};
        	}
        	else if(space(1,2) >= n_els) //top neigh is absent
        	{
        		v_knots[0] = {-2, -1, 0, 1};
        		v_knots[1] = {-1, 0, 1, 1};
        		v_knots[2] = {0, 1, 1, 1};
        	}
        	else
        	{
        		v_knots[0] = {-2, -1, 0, 1};
        		v_knots[1] = {-1, 0, 1, 2};
        		v_knots[2] = {0, 1, 2, 3};
        	}

        	auto &p1 = mesh.pts.row(mesh.els(e,0));
        	auto &p2 = mesh.pts.row(mesh.els(e,1));
        	auto &p4 = mesh.pts.row(mesh.els(e,3));

        	Vector2d node;

        	// print_local_space(space);

        	for(int y = 0; y < 3; ++y)
        	{
        		for(int x = 0; x < 3; ++x)
        		{
        			const int global_index = space(x, y);

        			if(h_knots[x][1]>=0 && h_knots[x][2]<=1 && v_knots[y][1]>=0 && v_knots[y][2]<=1)
        			{
        				get_node(x, y, h_knots, v_knots, p1, p2, p4, node);
        			}
        			else
        			{
        				//auto &p1o = mesh.pts.row(mesh.els(global_index,0));
        				//auto &p2o = mesh.pts.row(mesh.els(global_index,1));
        				//auto &p4o = mesh.pts.row(mesh.els(global_index,3));

        				//get_node(x, y, h_knots, v_knots, p1o, p2o, p4o, node);
        				node(0) = node(1) = std::numeric_limits<double>::max();
        			}

        			if(space(x, y) >= n_els)
        				bounday_nodes.push_back(global_index);

        		// b.push_back(new Spline2dBasis(space(x, y), node.transpose(), h_knots[x], v_knots[y]));
        			const int local_index = y*3 + x;
        			b[local_index].init(global_index, local_index, node.transpose());

        			// std::cout<<global_index<<" "<<v_knots[y][0]<<" "<<v_knots[y][1]<<" "<<v_knots[y][2]<<" "<<v_knots[y][3]<<" "<<std::endl;
        			const QuadraticTensorProductBSpline spline(h_knots[x], v_knots[y]);
        			b[local_index].set_basis([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.interpolate(uv, val); });
        			b[local_index].set_grad( [spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.derivative(uv, val); });
        		}
        	}
        }

        for(long e = 0; e < mesh.els.rows(); ++e)
        {
        	std::vector<Basis> &bs=bases[e];
        	const Matrix3i &space = spaces[e];

        	for(std::size_t i = 0; i < bs.size(); ++i)
        	{
        		Basis &b = bs[i];
        		if(b.node()(0) != std::numeric_limits<double>::max())
        			continue;

        		bool found = false;
        		for(int y = 0; y < 3; ++y)
        		{
        			for(int x = 0; x < 3; ++x)
        			{
        				const int global_index = space(x, y);

        				if(global_index == e || global_index >= n_els)
        					continue;

        				const std::vector<Basis> &other_bases = bases[global_index];
        				for(std::size_t j = 0; j < other_bases.size(); ++j)
        				{
        					if(other_bases[j].global_index() == b.global_index() && other_bases[j].node()(0) != std::numeric_limits<double>::max())
        					{
        						b.set_node(other_bases[j].node());
        						found = true;
        						break;
        					}
        				}

        				if(found) break;
        			}

        			if(found) break;
        		}

        		assert(found);

        	}
        }

        return n_els + boundary_bases;
    }

  //   void Spline2dBasis::basis(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
  //   {
		// // spline_.interpolate(uv, val);
  //   }

  //   void Spline2dBasis::grad(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
  //   {
		// // spline_.derivative(uv, val);
  //   }
}
