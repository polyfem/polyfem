#ifndef INTERFACE_DATA_HPP
#define INTERFACE_DATA_HPP

#include <vector>

namespace poly_fem
{
	struct InterfaceData
	{
		static const int LEFT_FLAG = 1;
		static const int TOP_FLAG = 2;
		static const int RIGHT_FLAG = 4;
		static const int BOTTOM_FLAG = 8;

		//ID of the neighoubring face, for debugging only
		int face_id = -1;

		//one of the top 6 flags, to know which parameterization boundary to sample
		int flag;
		//list of nodes of this edge/face
		std::vector<int> node_id;

		//list of local basis indices
		std::vector<int> local_indices;

		//list of local weights
		std::vector<double> vals;

		//vital! the 3 arrays MUST have the same lenght
	};

}

#endif //INTERFACE_DATA_HPP
