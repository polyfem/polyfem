#pragma once

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>

#include <map>

#define POLYFEM_REMESHER_SCOPED_TIMER(name) polyfem::utils::Timer __polyfem_timer(timings.timings[name])

namespace polyfem::mesh
{
	struct RemesherTimings
	{
		double total = 0;
		std::map<std::string, double> timings;
		int total_ndofs = 0;
		int n_solves = 0;

		void reset()
		{
			total = 0;
			for (auto &[name, time] : timings)
				time = 0;
			total_ndofs = 0;
			n_solves = 0;
		}

		void log()
		{
			logger().debug("Total time: {:.3g}s", total);
			double sum = 0;
			for (const auto &[name, time] : timings)
			{
				logger().debug("{}: {:.3g}s {:.1f}%", name, time, time / total * 100);
				sum += time;
			}
			logger().debug("Miscellaneous: {:.3g}s {:.1f}%", total - sum, (total - sum) / total * 100);
			if (n_solves > 0)
				logger().debug("Avg. # DOF per solve: {}", total_ndofs / double(n_solves));
		}
	};

} // namespace polyfem::mesh
