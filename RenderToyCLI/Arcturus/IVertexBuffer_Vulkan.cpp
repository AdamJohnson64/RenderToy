#include "ErrorVK.h"
#include "IDevice3D_Vulkan.h"
#include "IVertexBuffer_Vulkan.h"

namespace Arcturus
{
    IVertexBuffer_Vulkan::IVertexBuffer_Vulkan(IDevice3D_Vulkan* owner, uint32_t dataSize, const void* data) : IBuffer_Vulkan(owner, dataSize, data, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
    {
    }
}