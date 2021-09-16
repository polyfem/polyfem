#pragma once

#include <igl/Timer.h>

#define POLYFEM_SCOPED_TIMER(msg, total_time) polyfem::Timer __polyfem_timer(msg, total_time)
#define POLYFEM_SCOPED_TIMER_NO_GLOBAL(msg) polyfem::Timer __polyfem_timer(msg)

namespace polyfem
{
	class Timer
	{
	public:
		Timer(const std::string &msg)
			: m_msg(msg), m_total_time(nullptr)
		{
			start();
		}

		Timer(const std::string &msg, double &total_time)
			: m_msg(msg), m_total_time(&total_time)
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
			polyfem::logger().trace(m_msg.c_str(), getElapsedTimeInSec());
		}

		inline const igl::Timer &igl_timer()
		{
			return m_timer;
		}

	protected:
		std::string m_msg;
		igl::Timer m_timer;
		double *m_total_time;
	};
} // namespace polyfem
