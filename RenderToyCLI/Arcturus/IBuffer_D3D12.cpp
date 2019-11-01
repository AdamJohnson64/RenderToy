#include "D3D12Utility.h"
#include "ErrorD3D.h"
#include "IBuffer_D3D12.h"
#include "IDevice3D_D3D12.h"

namespace Arcturus
{
    IBuffer_D3D12::IBuffer_D3D12(IDevice3D_D3D12* owner, uint32_t bufferSize, uint32_t dataSize, const void* data) : m_owner(owner), m_size(dataSize)
    {
        m_resource.p = D3D12CreateBuffer(m_owner->m_device, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, bufferSize, dataSize, data, m_owner->m_commandQueue, m_owner->m_commandAllocator);
    }
}