#include "gpu_globals.h" // For GPU Globals
#include "gpu_api_functions.h" // For GPU API functions
#include "pagefault_handler.h"

GDF::sycl_device_t get_device_type_from_string(const std::string& device_type)
{
   if(device_type == "DEFAULT")
      return GDF::sycl_device_t::DEFAULT;
   else if(device_type == "CPU")
      return GDF::sycl_device_t::CPU;
   else if(device_type == "GPU")
      return GDF::sycl_device_t::GPU;
   else if(device_type == "ACCELERATOR")
   {
      log_msg<CDF::LogLevel::WARNING>("The compute device selected is ACCELERATOR");
      return GDF::sycl_device_t::ACCELERATOR;
   }
   else
   {
      std::string err_msg = "Unknown compute device type : " + device_type;
      log_msg<CDF::LogLevel::ERROR>(err_msg);
      return GDF::sycl_device_t::DEFAULT;
   }
}

// Allocate and initialize GPU variables
void setup_gpu_globals()
{
   #ifdef GPU_MEM_LOG
      tot_gpu_mem_used = 0;
      gpu_mem_usage_log.open("gpu_mem_usage.log", std::ios_base::trunc | std::ios_base::out);
   #endif

   // Setup GPU Manager (Has no variable which needs to be accesed on GPU)
   assert(!gpu_manager);
   gpu_manager = new GDF::GPUManager_t(GDF::sycl_device_t::GPU,
                                       {1, 1, gpu_global_range },
                                       {1, 1, gpu_local_range  } );

   setup_pagefault_handler();

   log_msg("Device type selected : DEFAULT");
}

void finalize_gpu_globals()
{
   assert(gpu_manager);
   GDF::gpu_barrier(); // Wait for all the GPU related processes to end

   // Finalize GPU Manager
   delete gpu_manager;
   gpu_manager = nullptr;

   #ifdef GPU_MEM_LOG
      gpu_mem_usage_log.close();
   #endif
}

#ifdef GPU_FULLY_OPTIMIZED
void send_vars_to_gpu()
{
   Boundary<strict_fp_t> Q_boundary_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("Q_boundary_local");
   GDF::transfer_to_gpu_noinit(Q_boundary_local);

   Cell<strict_fp_t> Q_cell_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Q_cell_local");
   GDF::transfer_to_gpu_noinit(Q_cell_local);

   VectorRead<int> number_of_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors_local");
   FaceRead<strict_fp_t> area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("area_local");
   GDF::transfer_to_gpu_readonly(number_of_neighbors_local, area_local);

   BoundaryRead<int> boundary_face_to_cell_local = m_silo.retrieve_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell_local");
   BoundaryRead<strict_fp_t> boundary_area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_area_local");
   GDF::transfer_to_gpu_readonly(boundary_face_to_cell_local, boundary_area_local);

   CellRead<strict_fp_t> volume_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("volume_local");
   GDF::transfer_to_gpu_readonly(volume_local);

   FaceRead<int> cell_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::FACE>("cell_neighbors_local");
   VectorRead<strict_fp_t> xcen_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("xcen_local");
   VectorRead<strict_fp_t> normal_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("normal_local");
   Face<strict_fp_t> rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("rdista_local");
   GDF::transfer_to_gpu_readonly(cell_neighbors_local, xcen_local, normal_local);
   GDF::transfer_to_gpu_move(rdista_local);

   VectorRead<strict_fp_t> boundary_xcen_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_xcen_local");
   VectorRead<strict_fp_t> boundary_normal_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_normal_local");
   Boundary<strict_fp_t> boundary_rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_rdista_local");
   GDF::transfer_to_gpu_readonly(boundary_xcen_local, boundary_normal_local);
   GDF::transfer_to_gpu_move(boundary_rdista_local);

   Vector<strict_fp_t> A_data_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("A_data_local");
   CellRead<int> csr_diag_idx_local = m_silo.retrieve_entry<int, CDF::StorageType::CELL>("csr_diag_idx_local");
   GDF::transfer_to_gpu_noinit(A_data_local);
   GDF::transfer_to_gpu_readonly(csr_diag_idx_local);

   FaceRead<int> csr_idx_local = m_silo.retrieve_entry<int, CDF::StorageType::FACE>("csr_idx_local");
   GDF::transfer_to_gpu_readonly(csr_idx_local);

   Cell<strict_fp_t> residual_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("residual_local");
   GDF::transfer_to_gpu_noinit(residual_local);

   Cell<strict_fp_t> rhs_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("rhs_local");
   GDF::transfer_to_gpu_noinit(rhs_local);

   VectorRead<int> ia_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ia_local");
   VectorRead<int> ja_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ja_local");
   Cell<strict_fp_t> dQ_old_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_old_local");
   Cell<strict_fp_t> dQ_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_local");
   GDF::transfer_to_gpu_readonly(ia_local, ja_local);
   GDF::transfer_to_gpu_noinit(dQ_old_local, dQ_local);
}
#endif
