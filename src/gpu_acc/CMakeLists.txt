# Each file in its own ALONE/UNITY_GROUP so changing .mm only recompiles that file (no full rebuild).
ob_set_subtarget(ob_gpu_acc ALONE
  ob_gpu_config.cpp
)
set_source_files_properties(ob_gpu_config.cpp PROPERTIES UNITY_GROUP "ob_gpu_acc_cpp")
if(APPLE)
  ob_set_subtarget(ob_gpu_acc ALONE
    metal/count_distinct/ob_gpu_count_distinct.mm
    metal/vector/ob_metal_l2.mm
    metal/vector/ob_metal_cosine.mm
  )
  set_source_files_properties(metal/count_distinct/ob_gpu_count_distinct.mm PROPERTIES
    COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
    UNITY_GROUP "ob_gpu_acc_mm"
  )
  set_source_files_properties(metal/vector/ob_metal_l2.mm PROPERTIES
    COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
    UNITY_GROUP "ob_gpu_acc_metal_l2_mm"
  )
  set_source_files_properties(metal/vector/ob_metal_cosine.mm PROPERTIES
    COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
    UNITY_GROUP "ob_gpu_acc_metal_cosine_mm"
  )
endif()

ob_server_add_target(ob_gpu_acc)
target_link_libraries(ob_gpu_acc PUBLIC ob_base)
if(APPLE)
  target_link_libraries(ob_gpu_acc PUBLIC
    "-framework Metal"
    "-framework Foundation"
    "-framework MetalPerformanceShaders"
  )
  # add_subdirectory(test)
endif()
