#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define EXCEPTION(STR) throw std::exception(__FILE__ "(" TOSTRING(__LINE__) "): EXCEPTION: " STR);

struct BuildFile
{
    const wchar_t* Name;
};

struct BuildExe
{
    const std::vector<const BuildFile*> Files;
    const std::vector<const wchar_t*> Includes;
};

BuildFile file_buildme = { L"Buildme\\Buildme.cpp" };
BuildFile file_dctest = { L"RenderToyTest\\DrawingContextTest.cpp" };
BuildFile file_dctext = { L"main_dc_text.cpp" };
BuildFile file_dccpu = { L"RenderToyCLI\\Arcturus\\DrawingContextCPU.cpp" };
BuildFile file_dcref = { L"RenderToyCLI\\Arcturus\\DrawingContextReference.cpp" };
BuildFile file_vector = { L"RenderToyCLI\\Arcturus\\Vector.cpp" };

BuildFile file_gtest = { L"ThirdParty\\googletest\\googletest\\src\\gtest_main.cc" };
BuildFile file_gtestall = { L"ThirdParty\\googletest\\googletest\\src\\gtest-all.cc" };

const wchar_t* incl_gtest = L"ThirdParty\\googletest\\googletest";
const wchar_t* incl_gtestinc = L"ThirdParty\\googletest\\googletest\\include";

const wchar_t* incl_rcli = L"RenderToyCLI";

BuildExe proj_dctext = {
    { &file_dctext, &file_dccpu, &file_dcref, &file_vector },
    {},
};

BuildExe proj_dctest = {
    { &file_gtest, &file_gtestall, &file_dctest, &file_dccpu, &file_dcref, &file_vector },
    { incl_gtest, incl_gtestinc, incl_rcli },
};

std::wstring SemiColonDelimit(const std::vector<const wchar_t*>& strings)
{
    std::wstringstream built;
    bool addsemicolon = false;
    for (const auto& string : strings)
    {
        if (addsemicolon)
        {
            built << L";";
        }
        addsemicolon = true;
        built << string;
    }
    return built.str();
}

void GenerateVisualStudioProject(std::wostream& str, const BuildExe& spec)
{
    // Emit Solution description.
    //str << std::endl;
    //str << L"Microsoft Visual Studio Solution File, Format Version 12.00" << std::endl;
    //str << L"VisualStudioVersion = 16.0.29418.71" << std::endl;

    // Emit Project description.
    str << R"LINE(<?xml version="1.0" encoding="utf-8"?>)LINE" << std::endl;
    str << R"LINE(<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">)LINE" << std::endl;
    // Project Configurations.
    str << R"LINE(  <ItemGroup Label="ProjectConfigurations">)LINE" << std::endl;
    str << R"LINE(    <ProjectConfiguration Include="Debug|x64">)LINE" << std::endl;
    str << R"LINE(      <Configuration>Debug</Configuration>)LINE" << std::endl;
    str << R"LINE(      <Platform>x64</Platform>)LINE" << std::endl;
    str << R"LINE(    </ProjectConfiguration>)LINE" << std::endl;
    str << R"LINE(  </ItemGroup>)LINE" << std::endl;
    // Global Project Settings.
    str << R"LINE(  <PropertyGroup Label="Globals">)LINE" << std::endl;
    str << R"LINE(    <VCProjectVersion>16.0</VCProjectVersion>)LINE" << std::endl;
    str << R"LINE(    <ProjectGuid>{8AD5DF22-2EB0-437B-B5A0-ABB02359130C}</ProjectGuid>)LINE" << std::endl;
    str << R"LINE(    <RootNamespace>Buildme</RootNamespace>)LINE" << std::endl;
    str << R"LINE(    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>)LINE" << std::endl;
    str << R"LINE(  </PropertyGroup>)LINE" << std::endl;
    // VISUAL STUDIO Preamble.
    str << R"LINE(  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />)LINE" << std::endl;
    // Compile/Link Global Settings.
    str << R"LINE(  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">)LINE" << std::endl;
    str << R"LINE(    <ClCompile>)LINE" << std::endl;
    str << "      <AdditionalIncludeDirectories>" << SemiColonDelimit(spec.Includes) << "</AdditionalIncludeDirectories>" << std::endl;
    str << R"LINE(    </ClCompile>)LINE" << std::endl;
    str << R"LINE(    <Link>)LINE" << std::endl;
    str << R"LINE(      <SubSystem>Console</SubSystem>)LINE" << std::endl;
    str << R"LINE(    </Link>)LINE" << std::endl;
    str << R"LINE(  </ItemDefinitionGroup>)LINE" << std::endl;
    // Project Settings.
    str << R"LINE(  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration">)LINE" << std::endl;
    str << R"LINE(    <ConfigurationType>Application</ConfigurationType>)LINE" << std::endl;
    str << R"LINE(    <PlatformToolset>v142</PlatformToolset>)LINE" << std::endl;
    str << R"LINE(    <CharacterSet>Unicode</CharacterSet>)LINE" << std::endl;
    str << R"LINE(  </PropertyGroup>)LINE" << std::endl;
    // VISUAL STUDIO Begin.
    str << R"LINE(  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />)LINE" << std::endl;
    // Project File Members.
    str << R"LINE(  <ItemGroup>)LINE" << std::endl;
    for (const auto& file : spec.Files)
    {
        str << "    <ClCompile Include=\"" << file->Name << "\" />" << std::endl;
    }
    str << R"LINE(  </ItemGroup>)LINE" << std::endl;
    // VISUAL STUDIO End.
    str << R"LINE(  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />)LINE";
    // End Of Project.
    str << R"LINE(</Project>)LINE" << std::endl;
}

int main()
{
    try
    {
        {
            std::wofstream stream("gen_dctext.vcxproj");
            GenerateVisualStudioProject(stream, proj_dctext);
        }
        {
            std::wofstream stream("gen_dctest.vcxproj");
            GenerateVisualStudioProject(stream, proj_dctest);
        }
        EXCEPTION("It's crap.");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return -1;
    }
}