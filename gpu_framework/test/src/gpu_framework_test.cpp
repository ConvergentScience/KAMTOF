#include "silo.h" // For all SILO functionalites
#include "test_funcs.h" // For test_functions
#include "gpu_api_functions.h" // For GPU API functions
#include "mpi_utils.h"
#include "pagefault_handler.h"

void test_bandwidth()
{
   if(numprocs != 2)
   {
      printf(" ERROR : This should only be run with 2 procs! Not more Not Less!\n");
      return;
   }
   size_t num_gb = 12;
   size_t num_bytes = num_gb * 1024 * 1024 * 1024;
   assert(num_bytes % sizeof(double) == 0);
   size_t num_elements = num_bytes / sizeof(double);
   double* device_ptr = GDF::malloc_gpu_var<double>(num_elements);
   
   if(rank == 0)
   {
      GDF::memset_gpu_var(device_ptr, 1, num_elements);
   }
   else
   {
      assert(rank == 1);
      GDF::memset_gpu_var(device_ptr, 0, num_elements);
   }

   MPI_Barrier(MPI_COMM_WORLD);

   auto start = std::chrono::high_resolution_clock::now();
   if(rank ==0)
   {
      MPI_Send(device_ptr, num_elements, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
   }
   else
   {
      assert(rank == 1);
      MPI_Recv(device_ptr, num_elements, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   }
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> elapsed = end - start;

   double bandwidth = num_gb / elapsed.count();
   printf("RANK %d | MPI device-to-device transfer took %.3f sec: %.2f GB/s\n", rank, elapsed.count(), bandwidth);
}

int main (int argc, char** argv)
{
   mpi_init(&argc, &argv);

   setup_pagefault_handler();

   setup_gpu_globals_test();

   setup_cdf_vars_for_gpu_framework_tests();

   if(argc == 2)
   {
      switch (atoi(argv[1]))
      {
         case 1:
         {
            porting_stage_scenario();
            break;
         }
         case 2:
         {
            demonstrate_temp_write_async();
            break;
         }
         case 3:
         {
            demonstrate_temp_write();
            break;
         }
         case 4:
         {
            backend_testing();
            break;
         }
         default:
         {
            log_error("Incorrect CLI passed for gpu_framework_test. Valid options are 1 -> porting_stage_scenario, 2 -> demonstrate_temp_write_async, "
                      "3 -> demonstrate_temp_write, 4 -> backend_testing ");
         }
      }
   }
   else
   {
      backend_testing();
      test_bandwidth();
   }

   m_silo.clear_entries();

   finalize_gpu_globals_test();

   mpi_finalize();

   return EXIT_SUCCESS;
}
