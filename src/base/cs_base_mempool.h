#ifndef CS_BASE_MEMPOOL_H
#define CS_BASE_MEMPOOL_H

#include <mutex>
#include <unordered_map>
#include <vector>
#include <memory>
#include "cs_base.h"  // Inclure les en-têtes nécessaires


/*----------------------------------------------------------------------------*/

#include "cs_defs.h"

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/*----------------------------------------------------------------------------
 * Standard C++ library headers
 *----------------------------------------------------------------------------*/

#include <map>

#if defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#endif

/*----------------------------------------------------------------------------
 * Local headers
 *----------------------------------------------------------------------------*/

#include "bft_error.h"
#include "bft_mem.h"

#if defined(HAVE_CUDA)
#include "cs_base_cuda.h"
#endif

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "cs_base_accel.h"


class MemoryPool {
public:
    static MemoryPool& instance();

    cs_mem_block_t allocate(size_t size,
                            cs_alloc_mode_t mode,
                            const char* var_name,
                            const char* file_name,
                            int line_num);

    cs_mem_block_t get_block_info(void* ptr);

    void deallocate(void* ptr,
                    const char* var_name,
                    const char* file_name,
                    int line_num);

    void update_block(void* ptr, const cs_mem_block_t& me_new);

    cs_mem_block_t allocate_device(const cs_mem_block_t& me_old,
                               const char* file_name,
                               int line_num);

    void insert_block(cs_mem_block_t unmanaged_block);

private:
    MemoryPool();
    ~MemoryPool();

    cs_mem_block_t allocate_new_block(size_t size,
                                      cs_alloc_mode_t mode,
                                      const char* var_name,
                                      const char* file_name,
                                      int line_num);

    void free_block(const cs_mem_block_t& me,
                    const char* var_name,
                    const char* file_name,
                    int line_num);


    std::mutex mutex_;
    std::unordered_map<void*, cs_mem_block_t> allocated_blocks_;
    std::unordered_map<cs_alloc_mode_t, std::vector<cs_mem_block_t>> free_blocks_;
};

#endif // CS_BASE_MEMPOOL_H
