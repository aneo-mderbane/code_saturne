/*============================================================================
 * Definitions, global variables, and base functions for accelerators.
 *============================================================================*/

/*
  This file is part of code_saturne, a general-purpose CFD tool.

  Copyright (C) 1998-2024 EDF S.A.

  This program is free software; you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
  details.

  You should have received a copy of the GNU General Public License along with
  this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
  Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

/*----------------------------------------------------------------------------*/

#include "cs_base_mempool.h"
#include "bft_mem.h"
#include "cs_base.h"

MemoryPool &
MemoryPool::instance()
{
  static std::once_flag flag;
  static MemoryPool    *instance = nullptr;
  std::call_once(flag, []() { instance = new MemoryPool(); });
  return *instance;
}

MemoryPool::MemoryPool() = default;

MemoryPool::~MemoryPool()
{
  for (auto &mode_blocks : free_blocks_) {
    cs_alloc_mode_t mode = mode_blocks.first;
    for (auto &me : mode_blocks.second) {
      free_block(me, nullptr, nullptr, 0);
    }
  }
}

void
MemoryPool::insert_block(cs_mem_block_t unmanaged_block)
{
  allocated_blocks_[unmanaged_block.host_ptr] = unmanaged_block;
}

cs_mem_block_t
MemoryPool::allocate(size_t          size,
                     cs_alloc_mode_t mode,
                     const char     *var_name,
                     const char     *file_name,
                     int             line_num)
{
  std::lock_guard<std::mutex> lock(mutex_);

  size_t adjusted_size = ((size + 63) / 64) * 64;

  const int TTL_MAX = 500;

  auto &free_blocks = free_blocks_[mode];

  auto it = free_blocks.begin();
  while (it != free_blocks.end()) {
    it->ttl += 1;
    if (it->ttl >= TTL_MAX) {
      free_block(*it, "memorypool.free", __FILE__, __LINE__);
      it = free_blocks.erase(it);
    }
    else {
      ++it;
    }
  }

  for (auto it2 = free_blocks.begin(); it2 != free_blocks.end(); ++it2) {
    if (it2->size == adjusted_size) {
      cs_mem_block_t me = *it2;
      free_blocks.erase(it2);
      allocated_blocks_[me.host_ptr] = me;
      // if (file_name != nullptr) {
      //   bft_mem_update_block_info(var_name, file_name, line_num, &me, &me);
      // }
      return me;
    }
  }

  cs_mem_block_t me =
    allocate_new_block(adjusted_size, mode, var_name, file_name, line_num);
  allocated_blocks_[me.host_ptr] = me;
  return me;
}

cs_mem_block_t
MemoryPool::get_block_info(void *ptr)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto                        it = allocated_blocks_.find(ptr);
  if (it != allocated_blocks_.end()) {
    return it->second;
  }
  else {
    cs_mem_block_t me = bft_mem_get_block_info_try(ptr);
    if (me.host_ptr != nullptr || me.device_ptr != nullptr)
      allocated_blocks_[me.host_ptr] = me;

    return me;
  }
}

void
MemoryPool::deallocate(void       *ptr,
                       const char *var_name,
                       const char *file_name,
                       int         line_num)
{
  if (ptr == nullptr)
    return;

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocated_blocks_.find(ptr);
  if (it != allocated_blocks_.end()) {
    cs_mem_block_t me = it->second;
    allocated_blocks_.erase(it);
    me.ttl = 0;
    free_blocks_[me.mode].push_back(me);
  }
  else {
    cs_mem_block_t me = bft_mem_get_block_info_try(ptr);
    me.ttl            = 0;
    free_blocks_[me.mode].push_back(me);
  }
}

cs_mem_block_t
MemoryPool::allocate_new_block(size_t          size,
                               cs_alloc_mode_t mode,
                               const char     *var_name,
                               const char     *file_name,
                               int             line_num)
{
  cs_mem_block_t me = { .host_ptr = nullptr,
#if defined(HAVE_ACCEL)
                        .device_ptr = nullptr,
#endif
                        .size = size,
                        .mode = mode,
                        .ttl  = 0 };

  if (mode < CS_ALLOC_HOST_DEVICE_PINNED) {
    me.host_ptr = bft_mem_malloc(1, size, var_name, nullptr, 0);
  }

#if defined(HAVE_CUDA)

  else if (mode == CS_ALLOC_HOST_DEVICE_PINNED) {
    me.host_ptr =
      cs_cuda_mem_malloc_host(me.size, var_name, file_name, line_num);
  }
  else if (mode == CS_ALLOC_HOST_DEVICE_SHARED) {
    me.host_ptr =
      cs_cuda_mem_malloc_managed(me.size, var_name, file_name, line_num);
    me.device_ptr = me.host_ptr;
  }
  else if (mode == CS_ALLOC_DEVICE) {
    me.device_ptr =
      cs_cuda_mem_malloc_device(me.size, var_name, file_name, line_num);
  }

#elif defined(SYCL_LANGUAGE_VERSION)

  else if (mode == CS_ALLOC_HOST_DEVICE_PINNED) {
    me.host_ptr = sycl::malloc_host(me.size, cs_glob_sycl_queue);
  }
  else if (mode == CS_ALLOC_HOST_DEVICE_SHARED) {
    me.host_ptr   = sycl::malloc_shared(me.size, cs_glob_sycl_queue);
    me.device_ptr = me.host_ptr;
  }
  else if (mode == CS_ALLOC_DEVICE) {
    me.device_ptr = sycl::malloc_device(me.size, cs_glob_sycl_queue);
  }

#elif defined(HAVE_OPENMP_TARGET)

  else if (mode == CS_ALLOC_HOST_DEVICE_PINNED) {
    me.host_ptr = omp_target_alloc_host(me.size, cs_glob_omp_target_device_id);
  }
  else if (mode == CS_ALLOC_HOST_DEVICE_SHARED) {
    me.host_ptr =
      omp_target_alloc_shared(me.size, cs_glob_omp_target_device_id);
    me.device_ptr = me.host_ptr;
  }
  else if (mode == CS_ALLOC_DEVICE) {
    me.device_ptr =
      omp_target_alloc_device(me.size, cs_glob_omp_target_device_id);
  }

#endif

  if (file_name != nullptr) {
    bft_mem_update_block_info(var_name, file_name, line_num, nullptr, &me);
  }

  return me;
}

void
MemoryPool::free_block(const cs_mem_block_t &me,
                       const char           *var_name,
                       const char           *file_name,
                       int                   line_num)
{
  if (me.mode < CS_ALLOC_HOST_DEVICE_PINNED) {
    bft_mem_free(me.host_ptr, var_name, nullptr, 0);
  }
  else if (me.host_ptr != nullptr) {
#if defined(HAVE_CUDA)

    if (me.mode == CS_ALLOC_HOST_DEVICE_SHARED) {
      cs_cuda_mem_free(me.host_ptr, var_name, file_name, line_num);
    }
    else {
      cs_cuda_mem_free_host(me.host_ptr, var_name, file_name, line_num);
    }

#elif defined(SYCL_LANGUAGE_VERSION)

    sycl::free(me.host_ptr, cs_glob_sycl_queue);

#elif defined(HAVE_OPENMP_TARGET)

    omp_target_free(me.host_ptr, cs_glob_omp_target_device_id);

#endif
  }

  if (me.device_ptr != nullptr && me.device_ptr != me.host_ptr) {
#if defined(HAVE_CUDA)

    cs_cuda_mem_free(me.device_ptr, var_name, file_name, line_num);

#elif defined(SYCL_LANGUAGE_VERSION)

    sycl::free(me.device_ptr, cs_glob_sycl_queue);

#elif defined(HAVE_OPENMP_TARGET)

    omp_target_free(me.device_ptr, cs_glob_omp_target_device_id);

#endif
  }

  if (file_name != nullptr) {
    bft_mem_update_block_info(var_name, file_name, line_num, &me, nullptr);
  }
}

cs_mem_block_t
MemoryPool::allocate_device(const cs_mem_block_t &me_old,
                            const char           *file_name,
                            int                   line_num)
{
  cs_mem_block_t me_new = me_old;

  if (me_old.mode == CS_ALLOC_HOST_DEVICE ||
      me_old.mode == CS_ALLOC_HOST_DEVICE_PINNED) {
#if defined(HAVE_CUDA)
    me_new.device_ptr =
      cs_cuda_mem_malloc_device(me_old.size, "device_ptr", file_name, line_num);
#elif defined(SYCL_LANGUAGE_VERSION)
    me_new.device_ptr =
      _sycl_mem_malloc_device(me_old.size, "device_ptr", file_name, line_num);
#elif defined(HAVE_OPENMP_TARGET)
    me_new.device_ptr = _omp_target_mem_malloc_device(me_old.size,
                                                      "device_ptr",
                                                      file_name,
                                                      line_num);
    if (omp_target_associate_ptr(me_new.host_ptr,
                                 me_new.device_ptr,
                                 me_new.size,
                                 0,
                                 cs_glob_omp_target_device_id))
      bft_error(file_name,
                line_num,
                0,
                _("%s: Can't associate host pointer %p to device pointer %p."),
                "omp_target_associate_ptr",
                me_new.host_ptr,
                me_new.device_ptr);
#endif
  }

  return me_new;
}

void
MemoryPool::update_block(void *ptr, const cs_mem_block_t &me_new)
{
  std::lock_guard<std::mutex> lock(mutex_);
  allocated_blocks_[ptr] = me_new;
}
