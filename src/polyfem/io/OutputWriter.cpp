#include "OutputWriter.hpp"

#include <polyfem/utils/par_for.hpp>

#include <condition_variable>
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#ifdef POLYFEM_WITH_TBB
#include <tbb/task_arena.h>
#endif

namespace polyfem::io
{
	class OutputWriter::Impl
	{
	public:
		explicit Impl(const bool asynchronous)
		{
#if defined(POLYFEM_WITH_CPP_THREADS) || defined(POLYFEM_WITH_TBB)
			asynchronous_ = asynchronous;
#endif
		}

		~Impl()
		{
			try
			{
				finish();
			}
			catch (...)
			{
				// Explicit finish/flush/enqueue calls report worker errors.
			}
		}

		void enqueue(const Prepare &prepare)
		{
			std::unique_lock<std::mutex> lock(state_->mutex);
			state_->changed.wait(lock, [&] {
				return (!state_->pending && (asynchronous_ || !state_->active)) || state_->closed || state_->error;
			});
			check_error();
			if (state_->closed)
				throw std::logic_error("Cannot enqueue output after finishing the writer");

			// Preparing after the slot becomes free bounds retained frame memory.
			Job job = prepare();
			if (!job)
				throw std::invalid_argument("An output frame must have a write operation");
			if (!asynchronous_)
			{
				state_->active = true;
				lock.unlock();
				try
				{
					job();
					job = {};
				}
				catch (...)
				{
					{
						std::lock_guard<std::mutex> failed(state_->mutex);
						state_->error = std::current_exception();
						state_->closed = true;
						state_->active = false;
					}
					state_->changed.notify_all();
					throw;
				}
				lock.lock();
				state_->active = false;
				lock.unlock();
				state_->changed.notify_all();
				return;
			}
			if (!started_)
				start();
			state_->pending = std::move(job);
			lock.unlock();
			state_->changed.notify_all();
		}

		void flush()
		{
			std::unique_lock<std::mutex> lock(state_->mutex);
			state_->changed.wait(lock, [&] { return (!state_->pending && !state_->active) || state_->error; });
			check_error();
		}

		void finish()
		{
			{
				std::lock_guard<std::mutex> lock(state_->mutex);
				state_->closed = true;
			}
			state_->changed.notify_all();
			if (!asynchronous_)
			{
				std::unique_lock<std::mutex> lock(state_->mutex);
				state_->changed.wait(lock, [&] { return !state_->active; });
			}
			if (thread_.joinable())
				thread_.join();
			if (completed_.valid())
				completed_.wait();
			std::lock_guard<std::mutex> lock(state_->mutex);
			check_error();
		}

		bool asynchronous() const { return asynchronous_; }

		void execute(const Job &compute)
		{
			const size_t threads = utils::get_n_threads();
			if (threads <= 1 && !started_)
				asynchronous_ = false;
			if (!asynchronous_ || threads <= 1)
			{
				compute();
				return;
			}
#ifdef POLYFEM_WITH_CPP_THREADS
			utils::ScopedThreadLimit limit(threads - 1);
			compute();
#elif defined(POLYFEM_WITH_TBB)
			tbb::task_arena simulation_arena(static_cast<int>(threads - 1));
			simulation_arena.execute(compute);
#else
			compute();
#endif
		}

	private:
		struct State
		{
			std::mutex mutex;
			std::condition_variable changed;
			Job pending;
			std::exception_ptr error;
			bool active = false;
			bool closed = false;
		};

		void check_error() const
		{
			if (state_->error)
				std::rethrow_exception(state_->error);
		}

		static void run(const std::shared_ptr<State> &state)
		{
			utils::ScopedThreadLimit serial_output(1);
			for (;;)
			{
				Job job;
				{
					std::unique_lock<std::mutex> lock(state->mutex);
					state->changed.wait(lock, [&] { return state->pending || state->closed; });
					if (!state->pending)
						return;
					job = std::exchange(state->pending, {});
					state->active = true;
				}
				state->changed.notify_all();
				try
				{
					job();
					job = {};
				}
				catch (...)
				{
					Job discarded;
					{
						std::lock_guard<std::mutex> lock(state->mutex);
						state->error = std::current_exception();
						state->active = false;
						state->closed = true;
						discarded = std::exchange(state->pending, {});
					}
					state->changed.notify_all();
					return;
				}
				{
					std::lock_guard<std::mutex> lock(state->mutex);
					state->active = false;
				}
				state->changed.notify_all();
			}
		}

		void start()
		{
#ifdef POLYFEM_WITH_CPP_THREADS
			thread_ = std::thread([state = state_] { run(state); });
#elif defined(POLYFEM_WITH_TBB)
			auto completion = std::make_shared<std::promise<void>>();
			completed_ = completion->get_future();
			arena_.enqueue([state = state_, completion] {
				run(state);
				completion->set_value();
			});
#endif
			started_ = true;
		}

		std::shared_ptr<State> state_ = std::make_shared<State>();
		std::thread thread_;
		std::future<void> completed_;
#ifdef POLYFEM_WITH_TBB
		tbb::task_arena arena_{1, 0};
#endif
		bool asynchronous_ = false;
		bool started_ = false;
	};

	OutputWriter::OutputWriter(const bool asynchronous) : impl_(std::make_unique<Impl>(asynchronous)) {}
	OutputWriter::~OutputWriter() = default;
	void OutputWriter::enqueue(const Prepare &prepare) { impl_->enqueue(prepare); }
	void OutputWriter::execute(const Job &compute) { impl_->execute(compute); }
	void OutputWriter::flush() { impl_->flush(); }
	void OutputWriter::finish() { impl_->finish(); }
	bool OutputWriter::asynchronous() const { return impl_->asynchronous(); }
} // namespace polyfem::io
