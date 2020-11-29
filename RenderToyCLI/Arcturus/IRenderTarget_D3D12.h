#pragma once

#include "IDevice3D_D3D12.h"
#include "IRenderTarget.h"

namespace Arcturus
{
    class IRenderTarget_D3D12 : public IRenderTarget
    {
    public:
        IRenderTarget_D3D12(IDevice3D_D3D12* owner, const RenderTargetDeclaration& declaration);
        IRenderTarget_D3D12(IDevice3D_D3D12* owner, const RenderTargetDeclaration& declaration, HANDLE handle);
        IDevice3D_D3D12* m_owner;
        CComPtr<ID3D12Resource1> m_resource;
        HANDLE m_handleNT;
    };
}