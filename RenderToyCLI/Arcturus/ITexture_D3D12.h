#pragma once

#include "ITexture.h"

#include <stdint.h>

namespace Arcturus
{
    class IDevice3D_D3D12;

    class ITexture_D3D12 : public ITexture
    {
    public:
        ITexture_D3D12(IDevice3D_D3D12* owner, uint32_t width, uint32_t height, const void* data);
        IDevice3D_D3D12*                m_owner;
        AutoRelease<ID3D12Resource1>    m_resource;
        D3D12_SHADER_RESOURCE_VIEW_DESC m_view;
    };
}