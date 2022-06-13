#pragma once

#include <igl/Timer.h>

#define POLYFEM_SCOPED_TIMER(...) polyfem::utils::Timer __polyfem_timer(__VA_ARGS__)

namespace polyfem
{
	namespace utils
	{
		class Timer
		{
		public:
			Timer()
				: m_total_time(nullptr)
			{
				start();
			}

			Timer(const std::string &name)
				: m_name(name), m_total_time(nullptr)
			{
				start();
			}

			Timer(double &total_time)
				: m_total_time(&total_time)
			{
				start();
			}

			Timer(const std::string &name, double &total_time)
				: m_name(name), m_total_time(&total_time)
			{
				start();
			}

			virtual ~Timer()
			{
				stop();
			}

			inline void start()
			{
				m_timer.start();
			}

			inline void stop()
			{
				m_timer.stop();
				log_msg();
				if (m_total_time)
				{
					*m_total_time += getElapsedTimeInSec();
				}
			}

			inline double getElapsedTimeInSec()
			{
				return m_timer.getElapsedTimeInSec();
			}

			inline void log_msg()
			{
				if (!m_name.empty())
				{
					logger().trace("[timing] {} {:.3g}s", m_name, getElapsedTimeInSec());
				}
			}

			inline const igl::Timer &igl_timer()
			{
				return m_timer;
			}

		protected:
			std::string m_name;
			igl::Timer m_timer;
			double *m_total_time;
		};
	} // namespace utils
} // namespace polyfem
