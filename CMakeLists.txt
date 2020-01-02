cmake_minimum_required(VERSION 3.10)

project(ArcturusTest)
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
