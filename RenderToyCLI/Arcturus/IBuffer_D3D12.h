#pragma once

#include "AutoRelease.h"

#include <stdint.h>

#include <d3d12.h>

namespace Arcturus
{
    class IDevice3D_D3D12;

    class IBuffer_D3D12
    {
    public:
        IBuffer_D3D12(IDevice3D_D3D12* owner, uint32_t bufferSize, uint32_t dataSize, const void* data);
        IDevice3D_D3D12* m_owner;
        uint32_t m_size;
        AutoRelease<ID3D12Resource1> m_resource;
    };
}