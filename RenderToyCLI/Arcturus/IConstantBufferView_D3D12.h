#pragma once

#include "IConstantBufferView.h"

namespace Arcturus
{
    class IConstantBuffer_D3D12;
    class IDevice3D_D3D12;

    class IConstantBufferView_D3D12 : public IConstantBufferView
    {
    public:
        IConstantBufferView_D3D12(IDevice3D_D3D12* owner, IConstantBuffer_D3D12* constantBuffer);
        IDevice3D_D3D12* m_owner;
    };
}