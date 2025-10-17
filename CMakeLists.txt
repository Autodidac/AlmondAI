cmake_minimum_required(VERSION 3.16)
project(AlmondAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ALMONDAI_HEADERS
    include/almondai/adapter.hpp
    include/almondai/buildparse.hpp
    include/almondai/eval.hpp
    include/almondai/governor.hpp
    include/almondai/ingest.hpp
    include/almondai/json.hpp
    include/almondai/mcp.hpp
    include/almondai/model.hpp
    include/almondai/retrieval.hpp
    include/almondai/serve.hpp
    include/almondai/tensor.hpp
    include/almondai/tokenizer_word.hpp
    include/almondai/train.hpp)

set(ALMONDAI_SOURCES
    src/adapter.cpp
    src/buildparse.cpp
    src/eval.cpp
    src/governor.cpp
    src/ingest.cpp
    src/json.cpp
    src/mcp.cpp
    src/model.cpp
    src/retrieval.cpp
    src/serve.cpp
    src/tensor.cpp
    src/tokenizer_word.cpp
    src/train.cpp)

add_library(almondai STATIC ${ALMONDAI_HEADERS} ${ALMONDAI_SOURCES})
target_include_directories(almondai PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(almondai_app src/main.cpp)
target_link_libraries(almondai_app PRIVATE almondai)
set_target_properties(almondai_app PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

