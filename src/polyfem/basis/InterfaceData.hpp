#pragma once

#include <vector>

namespace polyfem
{
	namespace basis
	{
		struct InterfaceData
		{
			// static const int LEFT_FLAG = 1;
			// static const int TOP_FLAG = 2;
			// static const int RIGHT_FLAG = 4;
			// static const int BOTTOM_FLAG = 8;

			// Global index of the incident element (other than the polygon)
			// int element_id = -1;

			// One of the 6 flags above, to know which boundary to sample in the parameterization domain
			// int flag;

			// The field on the interface edge/face is defined as a linear combination
			// of a certain number of basis. For regular Q1 or Q2 elements the weight
			// of the linear combination are always 1, but in the presence of irregular
			// or mixed elements, this may not always be the case.

			// list of nodes on this edge/face
			// std::vector<int> node_id;

			// list of local basis indices
			std::vector<int> local_indices;

			// list of local weights
			// std::vector<double> vals;

			// vital! the 3 arrays MUST have the same length
		};
	} // namespace basis
} // namespace polyfem
