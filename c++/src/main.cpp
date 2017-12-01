#include <igl/viewer/Viewer.h>

#include "Quadrature.hpp"
#include "LineQuadrature.hpp"
#include "QuadQuadrature.hpp"
#include "HexQuadrature.hpp"

#include "Basis.hpp"
#include "QuadBasis.hpp"
#include "HexBasis.hpp"

#include "AssemblyValues.hpp"
#include "ElementAssemblyValues.hpp"

#include "Assembler.hpp"
#include "Laplacian.hpp"

#include "Linear.hpp"
#include "Quadratic.hpp"
#include "TwoDFranke.hpp"

#include <fstream>

using namespace poly_fem;
using namespace Eigen;

void save_obj(const std::string &path, const MatrixXd &pts, const MatrixXi &els)
{
    std::ofstream file;
    file.open(path.c_str());

    IOFormat format(10, DontAlignCols, " ");

    if(file.good())
    {

        for(long i = 0; i < pts.rows(); ++i)
        {
            file << "v " << pts.row(i).format(format) << "\n";
        }

        file << "\n\n";

        for(long i = 0; i < els.rows(); ++i)
        {
            file << "f " << (els.row(i).array()+1).format(format) << "\n";
        }

        file.close();
    }
}

void build_hex_mesh(const int n_x_el, const int n_y_el, const int n_z_el, MatrixXd &pts, MatrixXi &els, std::vector< int > &bounday_nodes)
{
    const int n_pts = (n_x_el+1)*(n_y_el+1)*(n_z_el+1);
    const int n_els = n_x_el*n_y_el*n_z_el;

    pts.resize(n_pts, 3);
    els.resize(n_els, 8);

    for(int k=0; k<=n_z_el;++k)
    {
        for(int j=0; j<=n_y_el;++j)
        {
            for(int i=0; i<=n_x_el;++i)
            {
                const int index = k*(n_x_el+1)*(n_y_el+1)+j*(n_x_el+1)+i;

                if( j == 0 || j == n_y_el || i == 0 || i == n_x_el || k == 0 || k == n_z_el)
                    bounday_nodes.push_back(index);

                pts.row(index)=Vector3d(i,j,k);
            }
        }
    }

    pts.col(0)/=n_x_el;
    pts.col(1)/=n_y_el;
    pts.col(2)/=n_z_el;

    Matrix<int, 1, 8> el;
    for(int k=0; k<n_z_el;++k)
    {
        for(int j=0; j<n_y_el;++j)
        {
            for(int i=0; i<n_x_el;++i)
            {
                const int i1 = k*(n_x_el+1)*(n_y_el+1)+j*(n_x_el+1)+i;
                const int i2 = k*(n_x_el+1)*(n_y_el+1)+j*(n_x_el+1)+i+1;
                const int i3 = k*(n_x_el+1)*(n_y_el+1)+(j+1)*(n_x_el+1)+i+1;
                const int i4 = k*(n_x_el+1)*(n_y_el+1)+(j+1)*(n_x_el+1)+i;

                const int i5 = (k+1)*(n_x_el+1)*(n_y_el+1)+j*(n_x_el+1)+i;
                const int i6 = (k+1)*(n_x_el+1)*(n_y_el+1)+j*(n_x_el+1)+i+1;
                const int i7 = (k+1)*(n_x_el+1)*(n_y_el+1)+(j+1)*(n_x_el+1)+i+1;
                const int i8 = (k+1)*(n_x_el+1)*(n_y_el+1)+(j+1)*(n_x_el+1)+i;

                el << i1, i2, i3, i4, i5, i6, i7, i8;
                els.row(k*n_x_el*n_y_el+j*n_x_el+i)=el;
            }
        }
    }
}

void triangulate_hex_mesh(const MatrixXi &els, Matrix<int, Dynamic, 3> &vis_faces)
{
    assert(els.cols()==8);

    const long n_els = els.rows();

    const long n_vis_faces = n_els*6*2;
    vis_faces.resize(n_vis_faces, 3);

    long index = 0;
    for (long i = 0; i < n_els; ++i)
    {
        auto el = els.row(i);

        vis_faces.row(index++)=Vector3i(el(0),el(1),el(2));
        vis_faces.row(index++)=Vector3i(el(0),el(2),el(3));

        vis_faces.row(index++)=Vector3i(el(4),el(5),el(6));
        vis_faces.row(index++)=Vector3i(el(4),el(6),el(7));

        vis_faces.row(index++)=Vector3i(el(0),el(1),el(5));
        vis_faces.row(index++)=Vector3i(el(0),el(5),el(4));

        vis_faces.row(index++)=Vector3i(el(1),el(2),el(5));
        vis_faces.row(index++)=Vector3i(el(5),el(2),el(6));

        vis_faces.row(index++)=Vector3i(el(3),el(2),el(7));
        vis_faces.row(index++)=Vector3i(el(7),el(2),el(6));

        vis_faces.row(index++)=Vector3i(el(0),el(3),el(4));
        vis_faces.row(index++)=Vector3i(el(4),el(3),el(7));
    }
}


void build_quad_mesh(const int n_x_el, const int n_y_el, MatrixXd &pts, MatrixXi &els, std::vector< int > &bounday_nodes)
{
    const int n_pts = (n_x_el+1)*(n_y_el+1);
    const int n_els = n_x_el*n_y_el;

    pts.resize(n_pts, 2);
    els.resize(n_els, 4);

    for(int j=0; j<=n_y_el;++j)
    {
        for(int i=0; i<=n_x_el;++i)
        {
            const int index = j*(n_x_el+1)+i;

            if( j == 0 || j == n_y_el || i == 0 || i == n_x_el)
                bounday_nodes.push_back(index);

            pts.row(index)=Vector2d(i, j);
        }
    }

    pts.col(0)/=n_x_el;
    pts.col(1)/=n_y_el;

    Matrix<int, 1, 4> el;

    for(int j=0; j<n_y_el;++j)
    {
        for(int i=0; i<n_x_el;++i)
        {
            const int i1 = j*(n_x_el+1)+i;
            const int i2 = j*(n_x_el+1)+i+1;
            const int i3 = (j+1)*(n_x_el+1)+i+1;
            const int i4 = (j+1)*(n_x_el+1)+i;

            el << i1, i2, i3, i4;
            els.row(j*n_x_el+i)=el;
        }
    }
}

void triangulate_quad_mesh(const MatrixXi &els, Matrix<int, Dynamic, 3> &vis_faces)
{
    assert(els.cols()==4);
    const long n_els = els.rows();

    const long n_vis_faces = n_els*2;
    vis_faces.resize(n_vis_faces, 3);

    long index = 0;
    for (long i = 0; i < n_els; ++i)
    {
        auto el = els.row(i);

        vis_faces.row(index++)=Vector3i(el(0),el(1),el(2));
        vis_faces.row(index++)=Vector3i(el(0),el(2),el(3));
    }
}


void compute_errors(const std::vector< ElementAssemblyValues > &values, const std::vector< ElementAssemblyValues > &geom_values, const Problem &problem, const Eigen::MatrixXd &sol, double &l2_err, double &linf_err)
{
    using std::max;

    const int n_el=int(values.size());

    MatrixXd v_exact, v_approx;

    l2_err = 0;
    linf_err = 0;

    for(int e = 0; e < n_el; ++e)
    {
        auto vals    = values[e];
        auto gvalues = geom_values[e];

        problem.exact(gvalues.val, v_exact);

        v_approx = MatrixXd::Zero(v_exact.rows(), v_exact.cols());

        const int n_loc_bases=int(vals.basis_values.size());

        for(int i = 0; i < n_loc_bases; ++i)
        {
            auto val=vals.basis_values[i];

            v_approx = v_approx + sol(val.global_index) * val.val;
        }

        auto err = (v_exact-v_approx).cwiseAbs();

        linf_err = max(linf_err, err.maxCoeff());
        l2_err += (err.array() * err.array() * gvalues.det.array() * vals.quadrature.weights.array()).sum();
    }

    l2_err = sqrt(l2_err);
}

void compute_assembly_values(const bool use_hex, const int quadrature_order, const std::vector< std::vector<Basis *> > &bases, std::vector< ElementAssemblyValues > &values)
{
    Quadrature quadrature;
    if(use_hex)
    {
        HexQuadrature quad_quadrature;
        quad_quadrature.get_quadrature(quadrature_order, quadrature);
    }
    else
    {
        QuadQuadrature quad_quadrature;
        quad_quadrature.get_quadrature(quadrature_order, quadrature);
    }

    for(std::size_t i = 0; i < bases.size(); ++i)
    {
        const std::vector<Basis *> &bs = bases[i];
        ElementAssemblyValues &vals = values[i];
        vals.basis_values.resize(bs.size());
        vals.quadrature = quadrature;

        Eigen::MatrixXd mval = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());

        Eigen::MatrixXd dxmv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());
        Eigen::MatrixXd dymv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());
        Eigen::MatrixXd dzmv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());

        const int n_local_bases = int(bs.size());
        for(int j = 0; j < n_local_bases; ++j)
        {
            const Basis &b=*bs[j];
            AssemblyValues &val = vals.basis_values[j];

            val.global_index = b.global_index();


            b.basis(quadrature.points, j, val.val);
            b.grad(quadrature.points, j, val.grad);

            for (long k = 0; k < val.val.rows(); ++k){
                mval.row(k) += val.val(k,0)    * b.coeff();

                dxmv.row(k) += val.grad(k,0) * b.coeff();
                dymv.row(k) += val.grad(k,1) * b.coeff();
                if(use_hex)
                    dzmv.row(k) += val.grad(k,2) * b.coeff();
            }
        }

        if(use_hex)
            vals.finalize(mval, dxmv, dymv, dzmv);
        else
            vals.finalize(mval, dxmv, dymv);
    }
}

int main(int argc, char *argv[])
{
    const int quadrature_order = 2;

    Problem *problem = NULL;

    bool use_hex = false;
    if(argc>=2 && std::string(argv[1])=="hex")
        use_hex = true;

    int n_x_el=2;
    int n_y_el=2;
    int n_z_el=2;

    if(argc>=3)
        n_x_el=atoi(argv[2]);
    if(argc>=4)
        n_y_el=atoi(argv[3]);
    if(argc>=5)
        n_z_el=atoi(argv[4]);

    if(argc>=6)
    {
        const int problem_num = atoi(argv[5]);

        switch(problem_num)
        {
            case 0: problem = new Linear(); break;
            case 1: problem = new Quadratic(); break;
            case 2: problem = new TwoDFranke(); break;
        }
    }

    if(!problem)
        problem = new TwoDFranke();

    MatrixXd                pts;
    MatrixXi                els;
    Matrix<int, Dynamic, 3> vis_faces;
    int n_bases;

    std::vector< std::vector<Basis *> >  bases;
    std::vector< ElementAssemblyValues > values;
    std::vector< int >                   bounday_nodes;

    if(use_hex)
    {
        build_hex_mesh(n_x_el, n_y_el, n_z_el, pts, els, bounday_nodes);
        triangulate_hex_mesh(els, vis_faces);

        bases.resize(els.rows());
        values.resize(els.rows());

        n_bases = int(pts.rows());

        for(long i=0; i < els.rows(); ++i)
        {
            std::vector<Basis *> &b=bases[i];
            b.reserve(8);

            for(int j = 0; j < 8; ++j)
            {
                const int global_index = els(i,j);
                b.push_back(new HexBasis(global_index, pts.row(global_index)));
            }
        }
    }
    else
    {
        build_quad_mesh(n_x_el, n_y_el, pts, els, bounday_nodes);
        triangulate_quad_mesh(els, vis_faces);

        bases.resize(els.rows());
        values.resize(els.rows());

        n_bases = int(pts.rows());

        for(long i=0; i < els.rows(); ++i)
        {
            std::vector<Basis *> &b=bases[i];
            b.reserve(4);

            for(int j = 0; j < 4; ++j)
            {
                const int global_index = els(i,j);
                b.push_back(new QuadBasis(global_index, pts.row(global_index)));
            }
        }
    }

    save_obj("mesh.obj", pts, els);


    compute_assembly_values(use_hex, quadrature_order, bases, values);

    SparseMatrix<double, Eigen::RowMajor> stiffness;
    Eigen::MatrixXd rhs;

    Assembler<Laplacian> assembler;
    assembler.assemble(n_bases, values, values, stiffness);
    assembler.set_identity(bounday_nodes, stiffness);

    assembler.rhs(n_bases, values, values, *problem, rhs);
    rhs *= -1;
    assembler.bc(pts, bounday_nodes, *problem, rhs);
    // std::cout<<rhs<<"\n\n"<<std::endl;


    BiCGSTAB<SparseMatrix<double, Eigen::RowMajor> > solver;
    // SparseLU<SparseMatrix<double, Eigen::RowMajor> > solver;
    Eigen::MatrixXd u = solver.compute(stiffness).solve(rhs);

    // std::cout<<MatrixXd(stiffness)<<"\n\n"<<std::endl;
    // std::cout<<rhs<<"\n\n"<<std::endl;
    // std::cout<<u<<std::endl;

    double l2_err, linf_err;
    compute_errors(values, values, *problem, u, l2_err, linf_err);
    std::cout<<l2_err<<" "<<linf_err<<std::endl;



    Eigen::MatrixXd u_exact;
    problem->exact(pts, u_exact);

    Eigen::MatrixXd err = (u - u_exact).cwiseAbs();
    // Eigen::MatrixXd err = u_exact;

    Matrix<double,Dynamic, 3> col;
    igl::jet(err, true, col);







// Plot the mesh
    igl::viewer::Viewer viewer;
//     viewer.data.set_points(all_val, col);

    auto vis_pts = pts;

    if(vis_pts.cols() == 2)
    {
        vis_pts = MatrixXd(vis_pts.rows(), 3);
        vis_pts.col(0) = pts.col(0);
        vis_pts.col(1) = pts.col(1);
        vis_pts.col(2) = err;
    }


    viewer.data.set_mesh(vis_pts, vis_faces);
    viewer.data.set_colors(col);

    viewer.core.set_rotation_type(igl::viewer::ViewerCore::RotationType::ROTATION_TYPE_TRACKBALL);
    viewer.launch();


    for(std::size_t i = 0; i < bases.size(); ++i)
    {
        std::vector<Basis *> &b=bases[i];

        for(std::size_t j = 0; j < b.size(); ++j)
        {
            delete b[j];
        }
    }

    delete problem;


    return EXIT_SUCCESS;
}
