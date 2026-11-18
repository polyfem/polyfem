#pragma once

#include <functional>
#include <memory>

namespace polyfem::io
{
	/// Ordered output with one pending frame in addition to the active frame.
	/// Uses the configured CPP, TBB, or serial execution backend.
	class OutputWriter
	{
	public:
		using Job = std::function<void()>;
		using Prepare = std::function<Job()>;

		explicit OutputWriter(bool asynchronous = true);
		~OutputWriter();
		OutputWriter(const OutputWriter &) = delete;
		OutputWriter &operator=(const OutputWriter &) = delete;

		/// Wait for the pending slot, then prepare its owned frame on the caller.
		void enqueue(const Prepare &prepare);
		/// Run simulation work with one thread reserved for asynchronous output.
		void execute(const Job &compute);
		/// Wait for all accepted frames and propagate output failures.
		void flush();
		/// Drain accepted frames, stop the worker, and propagate output failures.
		void finish();
		bool asynchronous() const;

	private:
		class Impl;
		std::unique_ptr<Impl> impl_;
	};
} // namespace polyfem::io
