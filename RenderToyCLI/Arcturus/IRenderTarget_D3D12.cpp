#include "ErrorD3D.h"
#include "IDevice3D_D3D12.h"
#include "IRenderTarget_D3D12.h"

#include <dxgi1_6.h>

namespace Arcturus
{
    IRenderTarget_D3D12::IRenderTarget_D3D12(IDevice3D_D3D12* owner, const RenderTargetDeclaration& declaration) : m_owner(owner)
    {
        D3D12_CLEAR_VALUE descClear = {};
        descClear.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        descClear.DepthStencil.Depth = 1.0f;
        {
            D3D12_HEAP_PROPERTIES descHeapProperties = {};
            descHeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC descResource = {};
            descResource.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            descResource.Width = declaration.width;
            descResource.Height = declaration.height;
            descResource.DepthOrArraySize = 1;
            descResource.MipLevels = 1;
            descResource.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            descResource.SampleDesc.Count = 1;
            descResource.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            TRYD3D(owner->m_device->CreateCommittedResource1(&descHeapProperties, D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES | D3D12_HEAP_FLAG_SHARED, &descResource, D3D12_RESOURCE_STATE_COMMON, &descClear, nullptr, __uuidof(ID3D12Resource1), (void**)&m_resource.p));
        }
        TRYD3D(m_owner->m_device->CreateSharedHandle(m_resource, nullptr, GENERIC_ALL, nullptr, &m_handleNT));
    }

    IRenderTarget_D3D12::IRenderTarget_D3D12(IDevice3D_D3D12* owner, const RenderTargetDeclaration& declaration, HANDLE handle) : m_owner(owner)
    {
        TRYD3D(m_owner->m_device->OpenSharedHandle(handle, __uuidof(ID3D12Resource1), (void**)&m_resource.p));
    }
}