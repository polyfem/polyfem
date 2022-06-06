#include "OperatorSplittingSolver.hpp"
#include <unsupported/Eigen/SparseExtra>

#ifdef POLYFEM_WITH_OPENVDB
#include <openvdb/openvdb.h>
#endif

namespace polyfem
{
	using namespace assembler;

	namespace solver
	{
		void OperatorSplittingSolver::save_density()
		{
#ifdef POLYFEM_WITH_OPENVDB
			openvdb::initialize();
			openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
			openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

			for (int i = 0; i <= grid_cell_num(0); i++)
			{
				for (int j = 0; j <= grid_cell_num(1); j++)
				{
					if (dim == 2)
					{
						const int idx = i + j * (grid_cell_num(0) + 1);
						openvdb::Coord xyz(i, j, 0);
						if (density(idx) > 1e-8)
							accessor.setValue(xyz, density(idx));
					}
					else
					{
						for (int k = 0; k <= grid_cell_num(2); k++)
						{
							const int idx = i + (j + k * (grid_cell_num(1) + 1)) * (grid_cell_num(0) + 1);
							openvdb::Coord xyz(i, j, k);
							if (density(idx) > 1e-8)
								accessor.setValue(xyz, density(idx));
						}
					}
				}
			}
			grid->setName("density_smoke");
			grid->setGridClass(openvdb::GRID_FOG_VOLUME);

			static int num_frame = 0;
			const std::string filename = "density" + std::to_string(num_frame) + ".vdb";
			openvdb::io::File file(filename.c_str());
			num_frame++;

			openvdb::GridPtrVec(grids);
			grids.push_back(grid);
			file.write(grids);
			file.close();
#else
			static int num_frame = 0;
			std::string name = "density" + std::to_string(num_frame) + ".txt";
			std::ofstream file(name.c_str());
			num_frame++;
			for (int i = 0; i <= grid_cell_num(0); i++)
			{
				for (int j = 0; j <= grid_cell_num(1); j++)
				{
					if (dim == 2)
					{
						const int idx = i + j * (grid_cell_num(0) + 1);
						if (density(idx) < 1e-10)
							continue;
						file << i << " " << j << " " << density(idx) << std::endl;
					}
					else
					{
						for (int k = 0; k <= grid_cell_num(2); k++)
						{
							const int idx = i + (j + k * (grid_cell_num(1) + 1)) * (grid_cell_num(0) + 1);
							if (density(idx) < 1e-10)
								continue;
							file << i << " " << j << " " << k << " " << density(idx) << std::endl;
						}
					}
				}
			}
			file.close();
#endif
		};
	} // namespace solver
} // namespace polyfem