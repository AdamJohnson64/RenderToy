#pragma once

#include <exception>

#include <Windows.h>

namespace Arcturus
{
    void HandleErrorD3D(const char* line, HRESULT error);
}

#define TRYD3D(FN) HandleErrorD3D(#FN, ##FN);