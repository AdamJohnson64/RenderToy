#pragma once

#include "IShader.h"

namespace Arcturus
{
    class IDevice3D_Vulkan;

    class IShader_Vulkan : public IShader
    {
    public:
        IShader_Vulkan(IDevice3D_Vulkan* owner);
        ~IShader_Vulkan();
        IDevice3D_Vulkan*   m_owner;
        VkShaderModule      m_vkShaderModuleVertex;
        VkShaderModule      m_vkShaderModuleFragment;
        VkPipeline          m_vkPipeline;
    };
}