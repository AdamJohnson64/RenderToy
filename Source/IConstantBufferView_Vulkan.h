#pragma once

#include "IConstantBufferView.h"

namespace Arcturus
{
    class IConstantBuffer_Vulkan;
    class IDevice3D_Vulkan;

    class IConstantBufferView_Vulkan : public IConstantBufferView
    {
    public:
        IConstantBufferView_Vulkan(IDevice3D_Vulkan* owner, IConstantBuffer_Vulkan* constantBuffer);
        IDevice3D_Vulkan* m_owner;
    };
}