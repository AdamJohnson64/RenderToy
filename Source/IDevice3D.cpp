#include "IDevice3D.h"
#include "IDevice3D_D3D12.h"
#include "IDevice3D_Vulkan.h"

namespace Arcturus
{
    IDevice3D* CreateDevice3D_Direct3D12()
    {
        return new IDevice3D_D3D12();
    }
    IDevice3D* CreateDevice3D_Vulkan()
    {
        return new IDevice3D_Vulkan();
    }
}