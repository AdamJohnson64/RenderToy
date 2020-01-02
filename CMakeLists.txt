cmake_minimum_required(VERSION 3.10)

project(Arcturus)

# BasicRuntimeChecks (aka /RTCxxx) will make CLR sad - remove it completely
STRING (REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# We always want Unicode for all targets
add_definitions(-DUNICODE -D_UNICODE)

# Create the Arcturus CLR library
file(GLOB ARCUTURUSSOURCES RELATIVE ${CMAKE_SOURCE_DIR} "RenderToyCLI/Arcturus/*.cpp" "RenderToyCLI/Arcturus/*.h")
add_library(ArcturusLibrary
    ${ARCUTURUSSOURCES}
    RenderToyCLI/Arcturus/DrawingContextReference.cpp)
set_property(TARGET ArcturusLibrary PROPERTY COMMON_LANGUAGE_RUNTIME "")
target_include_directories(ArcturusLibrary PUBLIC "C:/VulkanSDK/1.1.121.0/Include")

# Create the test project
add_executable(ArcturusTest
    RenderToyCLI/Arcturus/DrawingContextReference.cpp
    RenderToyCLI/Arcturus/Vector.cpp
    RenderToyTest/DrawingContextTest.cpp
    RenderToyTest/SpaceFillTest.cpp
    ThirdParty/googletest/googletest/src/gtest-all.cc
    ThirdParty/googletest/googletest/src/gtest_main.cc
)
target_include_directories(ArcturusTest PUBLIC "ThirdParty/googletest/googletest/include")
target_include_directories(ArcturusTest PUBLIC "ThirdParty/googletest/googletest")
target_include_directories(ArcturusTest PUBLIC "RenderToyCLI")
