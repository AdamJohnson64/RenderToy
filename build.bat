@ECHO OFF

ECHO Compiling Drawing Context Test
"C:\Program Files\LLVM\bin\clang-cl.exe" /EHsc -IRenderToyCLI -IThirdParty\googletest\googletest -IThirdParty\googletest\googletest\include -Wno-everything -fcxx-exceptions -o dc_test.exe ThirdParty\googletest\googletest\src\gtest_main.cc ThirdParty\googletest\googletest\src\gtest-all.cc RenderToyCLI\Arcturus\DrawingContextCPU.cpp RenderToyCLI\Arcturus\DrawingContextReference.cpp RenderToyCLI\Arcturus\Vector.cpp RenderToyTest\DrawingContextTest.cpp

ECHO Compiling Drawing Context Text Mode
"C:\Program Files\LLVM\bin\clang.exe" -Wno-everything -o dc_text.exe main_dc_text.cpp RenderToyCLI\Arcturus\DrawingContextCPU.cpp RenderToyCLI\Arcturus\DrawingContextReference.cpp RenderToyCLI\Arcturus\Vector.cpp