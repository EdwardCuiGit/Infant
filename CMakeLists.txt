cmake_minimum_required(VERSION 3.10)

option(ENABLE_CUDA "Enable Cuda" OFF)

if (ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_VS_PLATFORM_TOOLSET_CUDA "12.8")
    project(Infant CXX CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_ARCHITECTURES 60)
else()
    project(Infant CXX)
endif()


set(CMAKE_CXX_STANDARD 20)

cmake_minimum_required(VERSION 3.10)
project(Infant)

include_directories(.)

if (ENABLE_CUDA)
    add_executable(Infant unit_tests/test_main.cc  
        src/0.tensors/predefs.cc 
        src/0.tensors/cuda_ctx.cu
        src/0.tensors/vector.cc
        src/1.functors/tensor_node.cc
        src/1.functors/functor.cc
        src/1.functors/un_func.cc
        src/1.functors/unmo_func.cc
        src/1.functors/bin_func.cc
        src/1.functors/functor_graph.cc
        src/2.operators/config_base.cc
        src/2.operators/operator_base.cc
    )

    target_link_libraries(Infant PRIVATE CUDA::cudart)
    set_target_properties(Infant PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_ARCHITECTURES 60
    )

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

    set_target_properties(Infant PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_compile_options(Infant PRIVATE
        $<$<CONFIG:Debug>:-g -G>
    )
else()
    add_executable(Infant
        src/0.tensors/predefs.cc 
        src/0.tensors/cuda_ctx.cpp
        src/0.tensors/vector.cc
        src/1.functors/tensor_node.cc
        src/1.functors/functor.cc
        src/1.functors/un_func.cc
        src/1.functors/unmo_func.cc
        src/1.functors/bin_func.cc
        src/1.functors/functor_graph.cc
        src/2.operators/config_base.cc
        src/2.operators/operator_base.cc
        unit_tests/test_main.cc  
    )

    target_compile_options(Infant PRIVATE
        $<$<CONFIG:Debug>:-g>
    )

endif()
