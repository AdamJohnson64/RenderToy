#include "ErrorD3D.h"

#include <exception>
#include <string>
#include <sstream>

#include <Windows.h>

namespace Arcturus
{

static std::string ConvertToHex(HRESULT v)
{
    std::stringstream str;
    str << "0x" << std::hex << v << " ";
    return str.str();
}

static std::string ConvertToD3DERR(HRESULT v)
{
    switch (v)
    {
    case E_OUTOFMEMORY: return "E_OUTOFMEMORY";
    case E_INVALIDARG: return "E_INVALIDARG";
    default: return ConvertToHex(v);
    }
}

static std::string ConvertToErrorString(const char* line, HRESULT error)
{
    return std::string(line) + " = " + ConvertToD3DERR(error);
}

void HandleErrorD3D(const char* line, HRESULT error)
{
    if (FAILED(error)) throw std::exception(ConvertToErrorString(line, error).c_str());
}

}