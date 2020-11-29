#pragma once

#include "IRenderTarget.h"
#include <atlbase.h>
#include <Windows.h>

namespace Arcturus
{
    class IDevice3D_Vulkan;

    class IRenderTarget_Vulkan: public IRenderTarget
    {
    public:
        IRenderTarget_Vulkan(IDevice3D_Vulkan* owner, const RenderTargetDeclaration& declaration, HANDLE d3d11);
        ~IRenderTarget_Vulkan();
        IDevice3D_Vulkan*   m_owner;
        uint32_t            m_width;
        uint32_t            m_height;
        VkImage             m_vkImage;
        VkDeviceMemory      m_vkDeviceMemory;
        VkImageView         m_vkImageView;
        VkFramebuffer       m_vkFramebuffer;
    };
}