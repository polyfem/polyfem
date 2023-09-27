#pragma once

// clang-format off
#include <spdlog/fmt/bundled/color.h>
#include <polyfem/utils/Logger.hpp>
// clang-format on

#include <igl/Timer.h>

#define POLYFEM_SCOPED_TIMER(...) polyfem::utils::Timer __polyfem_timer(__VA_ARGS__)

namespace polyfem
{
	namespace utils
	{
		struct Timing
		{
			operator double() const { return time; }

			void operator+=(const double t)
			{
				time += t;
				++count;
			}

			double time = 0;
			size_t count = 0;
		};

		class Timer
		{
		public:
			Timer()
			{
				start();
			}

			Timer(const std::string &name)
				: m_name(name)
			{
				start();
			}

			Timer(double &total_time)
				: m_total_time(&total_time)
			{
				start();
			}

			Timer(Timing &timing)
				: m_total_time(&timing.time), m_count(&timing.count)
			{
				start();
			}

			Timer(const std::string &name, double &total_time)
				: m_name(name), m_total_time(&total_time)
			{
				start();
			}

			Timer(const std::string &name, Timing &timing)
				: m_name(name), m_total_time(&timing.time), m_count(&timing.count)
			{
				start();
			}

			virtual ~Timer()
			{
				stop();
			}

			inline void start()
			{
				is_running = true;
				m_timer.start();
			}

			inline void stop()
			{
				if (!is_running)
					return;
				m_timer.stop();
				is_running = false;
				log_msg();
				if (m_total_time)
					*m_total_time += getElapsedTimeInSec();
				if (m_count)
					++(*m_count);
			}

			inline double getElapsedTimeInSec()
			{
				return m_timer.getElapsedTimeInSec();
			}

			inline void log_msg()
			{
				const static std::string log_fmt_text =
					fmt::format("[{}] {{}} {{:.3g}}s", fmt::format(fmt::fg(fmt::terminal_color::magenta), "timing"));

				if (!m_name.empty())
				{
					logger().trace(log_fmt_text, m_name, getElapsedTimeInSec());
				}
			}

			inline const igl::Timer &igl_timer()
			{
				return m_timer;
			}

		protected:
			std::string m_name;
			igl::Timer m_timer;
			double *m_total_time = nullptr;
			size_t *m_count = nullptr;
			bool is_running = false;
		};
	} // namespace utils
} // namespace polyfem
