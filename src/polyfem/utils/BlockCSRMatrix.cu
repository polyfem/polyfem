#include <polyfem/utils/BlockCSRMatrix.hpp>
#include <polyfem/utils/CudaUtils.hpp>

#include <cub/cub.cuh>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda_runtime_api.h>

#include <Eigen/SparseCore>

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace polyfem
{
	namespace
	{
		struct DeviceTriplet
		{
			int row;
			int col;
			double value;
		};

		// Key bit layout:
		// | col idx (32) | row idx (32) |

		__device__ uint64_t pack_key(int row, int col)
		{
			return (static_cast<uint64_t>(static_cast<uint32_t>(col)) << 32) | static_cast<uint32_t>(row);
		}

		__device__ int key_to_row(uint64_t key)
		{
			return static_cast<int>(static_cast<uint32_t>(key));
		}

		__device__ int key_to_col(uint64_t key)
		{
			return static_cast<int>(static_cast<uint32_t>(key >> 32));
		}

		__global__ void build_block_row_of_block(int block_row_num,
												 Span<const int> row_ptr,
												 Span<int> block_row_of_block)
		{
			int block_row = blockDim.x * blockIdx.x + threadIdx.x;
			if (block_row >= block_row_num)
			{
				return;
			}

			int begin = row_ptr[block_row];
			int end = row_ptr[block_row + 1];
			for (int k = begin; k < end; ++k)
			{
				block_row_of_block[k] = block_row;
			}
		}

		__global__ void build_keys_values(int bsr_scalar_nnz,
										  int triplet_nnz,
										  int block_dim,
										  BSRMatrixMutableView bsr,
										  Span<const int> block_row_of_block,
										  Span<const DeviceTriplet> triplets,
										  Span<uint64_t> keys,
										  Span<double> values)
		{
			int tid = blockDim.x * blockIdx.x + threadIdx.x;
			int nnz_total = bsr_scalar_nnz + triplet_nnz;
			if (tid >= nnz_total)
			{
				return;
			}

			// Real entry from input BSR. Compute scalar ij then pack.
			if (tid < bsr_scalar_nnz)
			{
				int block_size = block_dim * block_dim;
				int block_id = tid / block_size;
				int local = tid - block_id * block_size;
				int local_row = local / block_dim;
				int local_col = local - local_row * block_dim;

				int block_row = block_row_of_block[block_id];
				int block_col = bsr.col_idx[block_id];
				int row = block_row * block_dim + local_row;
				int col = block_col * block_dim + local_col;

				keys[tid] = pack_key(row, col);
				values[tid] = bsr.values[tid];
			}
			// Additional dynamic matrix entry.
			else
			{
				const DeviceTriplet &triplet = triplets[tid - bsr_scalar_nnz];
				keys[tid] = pack_key(triplet.row, triplet.col);
				values[tid] = triplet.value;
			}
		}

		__global__ void extract_cols(Span<const uint64_t> unique_keys,
									 Span<int> cols)
		{
			int tid = blockDim.x * blockIdx.x + threadIdx.x;
			if (tid >= unique_keys.size())
			{
				return;
			}

			cols[tid] = key_to_col(unique_keys[tid]);
		}

		__global__ void histogram(Span<const int> samples,
								  Span<int> hist,
								  int num_bins)
		{
			for (int i = blockDim.x * blockIdx.x + threadIdx.x;
				 i < samples.size();
				 i += blockDim.x * gridDim.x)
			{
				int sample = samples[i];
				assert(sample >= 0 && sample < num_bins);
				atomicAdd(&hist[sample], 1);
			}
		}

		__global__ void fill_csc_rows_vals(Span<const uint64_t> unique_keys,
										   Span<const int> value_offsets,
										   Span<const double> values,
										   Span<int> csc_rows,
										   Span<double> csc_vals)
		{
			int sid = blockDim.x * blockIdx.x + threadIdx.x;
			if (sid >= unique_keys.size())
			{
				return;
			}

			csc_rows[sid] = key_to_row(unique_keys[sid]);

			int begin = value_offsets[sid];
			int end = value_offsets[sid + 1];
			double sum = 0.0;
			for (int i = begin; i < end; ++i)
			{
				sum += values[i];
			}
			csc_vals[sid] = sum;
		}

		__global__ void add_values(Span<double> dst, Span<const double> src)
		{
			int id = blockDim.x * blockIdx.x + threadIdx.x;
			if (id >= dst.size())
			{
				return;
			}
			assert(src.size() == dst.size());
			dst[id] += src[id];
		}

		StiffnessMatrix bsr_to_stiffness_matrix_impl(
			BSRMatrixMutableView bsr,
			Span<const Eigen::Triplet<double>> triplets, // on host
			ExecutionPolicy policy)
		{
			assert(bsr.block_dim > 0);
			assert(bsr.rows % bsr.block_dim == 0);
			assert(bsr.cols % bsr.block_dim == 0);
			static_assert(std::is_same_v<StiffnessMatrix::Scalar, double>);

			int block_dim = bsr.block_dim;
			int block_size = block_dim * block_dim;
			if (bsr.col_idx.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / block_size)
				|| triplets.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
			{
				throw std::runtime_error("BSR to StiffnessMatrix input is too large. Non-zero number exceeding int32 max.");
			}

			int block_nnz = static_cast<int>(bsr.col_idx.size());
			int bsr_scalar_nnz = block_nnz * block_size;
			int triplet_nnz = static_cast<int>(triplets.size());
			if (triplet_nnz > std::numeric_limits<int>::max() - bsr_scalar_nnz)
			{
				throw std::runtime_error("BSR to StiffnessMatrix input is too large. Non-zero number exceeding int32 max.");
			}
			int nnz_total = bsr_scalar_nnz + triplet_nnz;

			if (nnz_total == 0)
			{
				StiffnessMatrix out(bsr.rows, bsr.cols);
				return out;
			}

			auto stream = policy.stream->get();

			// Upload host dynamic triplets to device POD triplets cause Eigen::Triplet is not compatible with cuda.
			std::vector<DeviceTriplet> h_triplets;
			auto d_triplets = cuda::make_buffer<DeviceTriplet>(*policy.stream, *policy.mr, triplet_nnz, cuda::no_init);
			if (triplet_nnz > 0)
			{
				h_triplets.reserve(triplet_nnz);
				for (const Eigen::Triplet<double> &triplet : triplets)
				{
					h_triplets.push_back(DeviceTriplet{
						static_cast<int>(triplet.row()),
						static_cast<int>(triplet.col()),
						triplet.value()});
				}
				cuda::copy_bytes(*policy.stream, h_triplets, d_triplets);
			}

			// ---------------------------------------------------------------------------
			// Map each BSR block index to input block row.
			// ---------------------------------------------------------------------------

			auto block_row_of_block = cuda::make_buffer<int>(*policy.stream, *policy.mr, block_nnz, cuda::no_init);
			build_block_row_of_block<<<div_round_up(bsr.block_rows(), 128), 128, 0, stream>>>(
				bsr.block_rows(),
				bsr.row_ptr,
				block_row_of_block);

			// ---------------------------------------------------------------------------
			// Convert each non-zero to [key, value] pair.
			// ---------------------------------------------------------------------------

			auto keys_in = cuda::make_buffer<uint64_t>(*policy.stream, *policy.mr, nnz_total, cuda::no_init);
			auto values_in = cuda::make_buffer<double>(*policy.stream, *policy.mr, nnz_total, cuda::no_init);
			build_keys_values<<<div_round_up(nnz_total, 128), 128, 0, stream>>>(
				bsr_scalar_nnz,
				triplet_nnz,
				block_dim,
				bsr,
				block_row_of_block,
				d_triplets,
				keys_in,
				values_in);
			d_triplets.destroy();
			block_row_of_block.destroy();

			// ---------------------------------------------------------------------------
			// Radix sort by key. Result should be in col major order.
			// ---------------------------------------------------------------------------

			auto keys_alt = cuda::make_buffer<uint64_t>(*policy.stream, *policy.mr, nnz_total, cuda::no_init);
			auto values_alt = cuda::make_buffer<double>(*policy.stream, *policy.mr, nnz_total, cuda::no_init);
			cub::DoubleBuffer<uint64_t> d_keys(keys_in.data(), keys_alt.data());
			cub::DoubleBuffer<double> d_values(values_in.data(), values_alt.data());
			auto cub_tmp = cuda::make_buffer<char>(*policy.stream, *policy.mr, 0, cuda::no_init);
			auto make_cub_tmp = [&cub_tmp, &policy](size_t required_size) {
				if (cub_tmp.size() < required_size)
				{
					cub_tmp.destroy();
					cub_tmp = cuda::make_buffer<char>(*policy.stream, *policy.mr, required_size, cuda::no_init);
				}
				return cub_tmp.data();
			};

			size_t sort_tmp_size = 0;
			cub::DeviceRadixSort::SortPairs(nullptr,
											sort_tmp_size,
											d_keys,
											d_values,
											nnz_total,
											0,
											64,
											stream);
			cub::DeviceRadixSort::SortPairs(make_cub_tmp(sort_tmp_size),
											sort_tmp_size,
											d_keys,
											d_values,
											nnz_total,
											0,
											64,
											stream);

			auto &sorted_keys_buf = (d_keys.Current() == keys_in.data()) ? keys_in : keys_alt;
			auto &sorted_values_buf = (d_values.Current() == values_in.data()) ? values_in : values_alt;
			auto &stale_keys_buf = (d_keys.Current() == keys_in.data()) ? keys_alt : keys_in;
			auto &stale_values_buf = (d_values.Current() == values_in.data()) ? values_alt : values_in;
			stale_keys_buf.destroy();
			stale_values_buf.destroy();

			// ---------------------------------------------------------------------------
			// Run-length encode.
			// This step computes non-zero scalar num + value count for each scalar.
			// ---------------------------------------------------------------------------

			auto unique_keys = cuda::make_buffer<uint64_t>(*policy.stream, *policy.mr, nnz_total, cuda::no_init);

			// We later use this buffer to do exclusive sum for value offsets.
			// Thus the size is nnz_total + 1.
			auto counts = cuda::make_buffer<int>(*policy.stream, *policy.mr, nnz_total + 1, cuda::no_init);
			auto num_runs = cuda::make_buffer<int>(*policy.stream, *policy.mr, 1, cuda::no_init);

			size_t rle_tmp_size = 0;
			cub::DeviceRunLengthEncode::Encode(nullptr,
											   rle_tmp_size,
											   d_keys.Current(),
											   unique_keys.data(),
											   counts.data(),
											   num_runs.data(),
											   nnz_total,
											   stream);
			cub::DeviceRunLengthEncode::Encode(make_cub_tmp(rle_tmp_size),
											   rle_tmp_size,
											   d_keys.Current(),
											   unique_keys.data(),
											   counts.data(),
											   num_runs.data(),
											   nnz_total,
											   stream);
			sorted_keys_buf.destroy();

			// ---------------------------------------------------------------------------
			// Histogram + ExclusiveSum
			// This step count non-zero scalars per cols to compute col_ptr.
			// ---------------------------------------------------------------------------

			int unique_nnz = 0;
			cudaMemcpyAsync(&unique_nnz, num_runs.data(), sizeof(int), cudaMemcpyDeviceToHost, stream);
			policy.stream->sync();
			num_runs.destroy();

			auto cols = cuda::make_buffer<int>(*policy.stream, *policy.mr, unique_nnz, cuda::no_init);
			// Extract col index from packed key.
			extract_cols<<<div_round_up(unique_nnz, 128), 128, 0, stream>>>(
				Span<const uint64_t>(unique_keys.data(), unique_nnz),
				cols);
			// Histogram count nnz scalar per col.
			auto hist = cuda::make_buffer<int>(*policy.stream, *policy.mr, bsr.cols + 1, 0);
			// As of 20260507, CCCL v3.0.0 histogram has a out-of-bound memory write bug.
			// If the bug is fixed in the future you shall replace the homebrew histogram.
			histogram<<<div_round_up(unique_nnz, 128), 128, 0, stream>>>(
				cols,
				hist,
				bsr.cols);
			// Exclusive scan compute CSC col ptr.
			auto csc_col_ptr = cuda::make_buffer<int>(*policy.stream, *policy.mr, bsr.cols + 1, cuda::no_init);
			size_t scan_tmp_size = 0;
			cub::DeviceScan::ExclusiveSum(nullptr,
										  scan_tmp_size,
										  hist.data(),
										  csc_col_ptr.data(),
										  bsr.cols + 1,
										  stream);
			cub::DeviceScan::ExclusiveSum(make_cub_tmp(scan_tmp_size),
										  scan_tmp_size,
										  hist.data(),
										  csc_col_ptr.data(),
										  bsr.cols + 1,
										  stream);
			cols.destroy();
			hist.destroy();

			// ---------------------------------------------------------------------------
			// Fill rows and vals non-zero scalars.
			// ---------------------------------------------------------------------------

			// Compute value offsets for each scalar.
			auto value_offsets = cuda::make_buffer<int>(*policy.stream, *policy.mr, unique_nnz + 1, cuda::no_init);
			cudaMemsetAsync(counts.data() + unique_nnz, 0, sizeof(int), stream);
			size_t off_tmp_size = 0;
			cub::DeviceScan::ExclusiveSum(nullptr,
										  off_tmp_size,
										  counts.data(),
										  value_offsets.data(),
										  unique_nnz + 1,
										  stream);
			cub::DeviceScan::ExclusiveSum(make_cub_tmp(off_tmp_size),
										  off_tmp_size,
										  counts.data(),
										  value_offsets.data(),
										  unique_nnz + 1,
										  stream);
			counts.destroy();
			cub_tmp.destroy();

			// Fill rows and vals.
			auto csc_rows = cuda::make_buffer<int>(*policy.stream, *policy.mr, unique_nnz, cuda::no_init);
			auto csc_vals = cuda::make_buffer<double>(*policy.stream, *policy.mr, unique_nnz, cuda::no_init);
			fill_csc_rows_vals<<<div_round_up(unique_nnz, 128), 128, 0, stream>>>(
				Span<const uint64_t>(unique_keys.data(), unique_nnz),
				value_offsets,
				Span<const double>(d_values.Current(), nnz_total),
				csc_rows,
				csc_vals);
			unique_keys.destroy();
			value_offsets.destroy();
			sorted_values_buf.destroy();

			// ---------------------------------------------------------------------------
			// Allocate Eigen-owned CSC storage, then download directly into it.
			// ---------------------------------------------------------------------------

			static_assert(std::is_same_v<int, StiffnessMatrix::StorageIndex>, "NG assembly path does not support large index.");
			StiffnessMatrix out(bsr.rows, bsr.cols);
			out.resizeNonZeros(unique_nnz);

			cudaMemcpyAsync(
				out.outerIndexPtr(),
				csc_col_ptr.data(),
				static_cast<size_t>(bsr.cols + 1) * sizeof(StiffnessMatrix::StorageIndex),
				cudaMemcpyDeviceToHost,
				stream);
			cudaMemcpyAsync(
				out.innerIndexPtr(),
				csc_rows.data(),
				static_cast<size_t>(unique_nnz) * sizeof(StiffnessMatrix::StorageIndex),
				cudaMemcpyDeviceToHost,
				stream);
			cudaMemcpyAsync(
				out.valuePtr(),
				csc_vals.data(),
				static_cast<size_t>(unique_nnz) * sizeof(double),
				cudaMemcpyDeviceToHost,
				stream);

			csc_col_ptr.destroy();
			csc_rows.destroy();
			csc_vals.destroy();

			policy.stream->sync();

			return out;
		}

	} // namespace

	StiffnessMatrix BSRMatrix::to_stiffness_matrix_device(ExecutionPolicy policy)
	{
		BSRMatrixMutableView device_view = this->device_static_view(policy);
		// If host static view exists, sum the value array into device value ptr.
		if (has_allocate_host_value())
		{
			BSRMatrixMutableView host_view = static_view();
			auto host_values = cuda::make_buffer<double>(
				*policy.stream,
				*policy.mr,
				host_view.values.size(),
				cuda::no_init);
			cuda::copy_bytes(*policy.stream, host_view.values, host_values);

			int grid_num = div_round_up(device_view.values.size(), 128);
			add_values<<<grid_num, 128, 0, policy.stream->get()>>>(device_view.values, host_values);
		}
		return bsr_to_stiffness_matrix_impl(device_view, dynamic_values_, policy);
	}

} // namespace polyfem
