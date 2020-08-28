#include "OperatorSplittingSolver.hpp"
#include <unsupported/Eigen/SparseExtra>

#ifdef POLYFEM_WITH_OPENVDB
#include <openvdb/openvdb.h>
#endif

namespace polyfem
{
void OperatorSplittingSolver::save_density()
{
#ifdef POLYFEM_WITH_OPENVDB
    openvdb::initialize();
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

    for(int i = 0; i <= grid_cell_num(0); i++)
    {
        for(int j = 0; j <= grid_cell_num(1); j++)
        {
            if(dim == 2)
            {
                const int idx = i + j * (grid_cell_num(0)+1);
                openvdb::Coord xyz(i, j, 0);
                if(density(idx) > 1e-8)
                    accessor.setValue(xyz, density(idx));
            }
            else
            {
                for(int k = 0; k <= grid_cell_num(2); k++)
                {
                    const int idx = i + (j + k * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
                    openvdb::Coord xyz(i, j, k);
                    if(density(idx) > 1e-8)
                        accessor.setValue(xyz, density(idx));
                }
            }
        }
    }
    grid->setName("density_smoke");
    grid->setGridClass(openvdb::GRID_FOG_VOLUME);

    static int num_frame = 0;
    const std::string filename = "density"+std::to_string(num_frame)+".vdb";
    openvdb::io::File file(filename.c_str());
    num_frame++;

    openvdb::GridPtrVec(grids);
    grids.push_back(grid);
    file.write(grids);
    file.close();
#else
    static int num_frame = 0;
    std::string name = "density"+std::to_string(num_frame)+".txt";
    std::ofstream file(name.c_str());
    num_frame++;
    for(int i = 0; i <= grid_cell_num(0); i++)
    {
        for(int j = 0; j <= grid_cell_num(1); j++)
        {
            if(dim == 2)
            {
                const int idx = i + j * (grid_cell_num(0)+1);
                if(density(idx) < 1e-10) continue;
                file << i << " " << j << " " << density(idx) << std::endl;
            }
            else
            {
                for(int k = 0; k <= grid_cell_num(2); k++)
                {
                    const int idx = i + (j + k * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
                    if(density(idx) < 1e-10) continue;
                    file << i << " " << j << " " << k << " " << density(idx) << std::endl;
                }
            }
        }
    }
    file.close();

    // Eigen::MatrixXd points(3, density.size());
    // Eigen::MatrixXd field(density.size(), 1);
    // Eigen::MatrixXi tets;
    // if(dim == 2) tets.resize(grid_cell_num(0)*grid_cell_num(1), 4);
    // else tets.resize(grid_cell_num(0)*grid_cell_num(1)*grid_cell_num(2), 8);
    // for(int i = 0; i < grid_cell_num(0); i++)
    // {
    //     for(int j = 0; j < grid_cell_num(1); j++)
    //     {
    //         if(dim == 2)
    //         {
    //             const int idx = i + j * grid_cell_num(0);
    //             tets(idx, 0) = i + j * (grid_cell_num(0)+1);
    //             tets(idx, 1) = (i+1) + j * (grid_cell_num(0)+1);
    //             tets(idx, 2) = (i+1) + (j+1) * (grid_cell_num(0)+1);
    //             tets(idx, 3) = i + (j+1) * (grid_cell_num(0)+1);
    //         }
    //         else
    //         {
    //             for(int k = 0; k < grid_cell_num(2); k++)
    //             {
    //                 const int idx = i + (j + k * grid_cell_num(1)) * grid_cell_num(0);
    //                 tets(idx, 0) = i + (j + k * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
    //                 tets(idx, 1) = (i+1) + (j + k * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
    //                 tets(idx, 2) = (i+1) + ((j+1) + k * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
    //                 tets(idx, 3) = i + ((j+1) + k * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
    //                 tets(idx, 4) = i + (j + (k+1) * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
    //                 tets(idx, 5) = (i+1) + (j + (k+1) * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
    //                 tets(idx, 6) = (i+1) + ((j+1) + (k+1) * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
    //                 tets(idx, 7) = i + ((j+1) + (k+1) * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
    //             }
    //         }
    //     }
    // }
    // for(int i = 0; i <= grid_cell_num(0); i++)
    // {
    //     for(int j = 0; j <= grid_cell_num(1); j++)
    //     {
    //         if(dim == 2)
    //         {
    //             const int idx = i + j * (grid_cell_num(0)+1);
    //             field(idx) = density(idx);
    //             points(0, idx) = i * resolution + min_domain(0);
    //             points(1, idx) = j * resolution + min_domain(1);
    //         }
    //         else
    //         {
    //             for(int k = 0; k <= grid_cell_num(2); k++)
    //             {
    //                 const int idx = i + (j + k * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
    //                 field(idx) = density(idx);
    //                 points(0, idx) = i * resolution + min_domain(0);
    //                 points(1, idx) = j * resolution + min_domain(1);
    //                 points(2, idx) = k * resolution + min_domain(2);
    //             }
    //         }
    //     }
    // }
    // static int num_frame = 0;
    // std::string name = "density"+std::to_string(num_frame)+".vtk";
    // std::ofstream file(name.c_str());
    // num_frame++;

    // file << "# vtk DataFile Version 2.0\nDensity\nASCII\nDATASET POLYDATA\n";
    // file << "POINTS " << points.cols() << " float\n";
    // for(int i = 0; i < points.cols(); i++)
    // {
    //     for(int d = 0; d < dim; d++) file << points(d, i) << " ";
    //     if(dim == 2) file << "0";
    //     file << "\n";
    // }
    // file << "POLYGONS " << tets.rows() << " " << tets.size()+tets.rows() << std::endl;
    // for(int e = 0; e < tets.rows(); e++)
    // {
    //     file << tets.cols() << " ";
    //     for(int k = 0; k < tets.cols(); k++) file << tets(e, k) << " ";
    //     file << "\n";
    // }
    // file << "POINT_DATA " << density.size() << "\n";
    // file << "SCALARS density float 1\n";
    // file << "LOOKUP_TABLE my_table\n";
    // for(int i = 0; i < density.size(); i++)
    // {
    //     file << density(i) << "\n";
    // }
    // file.close();
#endif
};
}