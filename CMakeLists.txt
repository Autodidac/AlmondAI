cmake_minimum_required(VERSION 3.16)
project(AlmondAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CURL REQUIRED)

set(ALMONDAI_HEADERS
    AlmondAI/include/almondai/adapter.hpp
    AlmondAI/include/almondai/chat/backend.hpp
    AlmondAI/include/almondai/buildparse.hpp
    AlmondAI/include/almondai/eval.hpp
    AlmondAI/include/almondai/fallback.hpp
    AlmondAI/include/almondai/governor.hpp
    AlmondAI/include/almondai/ingest.hpp
    AlmondAI/include/almondai/json.hpp
    AlmondAI/include/almondai/net/http.hpp
    AlmondAI/include/almondai/mcp.hpp
    AlmondAI/include/almondai/model.hpp
    AlmondAI/include/almondai/retrieval.hpp
    AlmondAI/include/almondai/serve.hpp
    AlmondAI/include/almondai/tensor.hpp
    AlmondAI/include/almondai/tokenizer_word.hpp
    AlmondAI/include/almondai/train.hpp)

set(ALMONDAI_SOURCES
    AlmondAI/src/adapter.cpp
    AlmondAI/src/chat/backend.cpp
    AlmondAI/src/buildparse.cpp
    AlmondAI/src/eval.cpp
    AlmondAI/src/fallback.cpp
    AlmondAI/src/governor.cpp
    AlmondAI/src/ingest.cpp
    AlmondAI/src/json.cpp
    AlmondAI/src/mcp.cpp
    AlmondAI/src/model.cpp
    AlmondAI/src/net/http.cpp
    AlmondAI/src/retrieval.cpp
    AlmondAI/src/serve.cpp
    AlmondAI/src/tensor.cpp
    AlmondAI/src/tokenizer_word.cpp
    AlmondAI/src/train.cpp)

add_library(almondai STATIC ${ALMONDAI_HEADERS} ${ALMONDAI_SOURCES})
target_include_directories(almondai PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/AlmondAI/include)
target_link_libraries(almondai PUBLIC CURL::libcurl)

