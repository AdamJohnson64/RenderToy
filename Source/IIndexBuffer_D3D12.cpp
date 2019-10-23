#include "IIndexBuffer_D3D12.h"

namespace Arcturus
{
    IIndexBuffer_D3D12::IIndexBuffer_D3D12(IDevice3D_D3D12* owner, uint32_t dataSize, const void* data) : IBuffer_D3D12(owner, dataSize, dataSize, data)
    {
        m_descIndexBufferView.BufferLocation = m_resource->GetGPUVirtualAddress();
        m_descIndexBufferView.SizeInBytes = dataSize;
        m_descIndexBufferView.Format = DXGI_FORMAT_R32_UINT;
    }
}