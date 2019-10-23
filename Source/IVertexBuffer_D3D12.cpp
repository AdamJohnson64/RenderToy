#include "D3D12Utility.h"
#include "ErrorD3D.h"
#include "IVertexBuffer_D3D12.h"
#include "Vector.h"

namespace Arcturus
{
    IVertexBuffer_D3D12::IVertexBuffer_D3D12(IDevice3D_D3D12* owner, uint32_t dataSize, uint32_t strideSize, const void* data) : IBuffer_D3D12(owner, dataSize, dataSize, data), m_strideSize(strideSize)
    {
        m_descVertexBufferView.BufferLocation = m_resource->GetGPUVirtualAddress();
        m_descVertexBufferView.SizeInBytes = dataSize;
        m_descVertexBufferView.StrideInBytes = strideSize;
    }
}