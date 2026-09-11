add_executable(test-mul-mat-convrot test-mul-mat-convrot.cpp)
target_link_libraries(test-mul-mat-convrot PRIVATE ggml ggml-cpu)

add_test(NAME test-mul-mat-convrot COMMAND test-mul-mat-convrot)

if (TARGET ggml-metal)
    add_executable(test-mul-mat-convrot-metal test-mul-mat-convrot-metal.cpp)
    target_link_libraries(test-mul-mat-convrot-metal PRIVATE ggml ggml-metal)
    add_test(NAME test-mul-mat-convrot-metal COMMAND test-mul-mat-convrot-metal)
endif()

if (TARGET ggml-cuda)
    add_executable(test-mul-mat-convrot-cuda test-mul-mat-convrot-gpu.cpp)
    target_compile_definitions(test-mul-mat-convrot-cuda PRIVATE GGML_TEST_CONVROT_CUDA)
    target_link_libraries(test-mul-mat-convrot-cuda PRIVATE ggml ggml-cuda)
    add_test(NAME test-mul-mat-convrot-cuda COMMAND test-mul-mat-convrot-cuda)
endif()

if (TARGET ggml-vulkan)
    add_executable(test-mul-mat-convrot-vulkan test-mul-mat-convrot-gpu.cpp)
    target_compile_definitions(test-mul-mat-convrot-vulkan PRIVATE GGML_TEST_CONVROT_VULKAN)
    target_link_libraries(test-mul-mat-convrot-vulkan PRIVATE ggml ggml-vulkan)
    add_test(NAME test-mul-mat-convrot-vulkan COMMAND test-mul-mat-convrot-vulkan)
endif()
