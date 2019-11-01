#include "D3D12Utility.h"
#include "IConstantBuffer_D3D12.h"

namespace Arcturus
{
    IConstantBuffer_D3D12::IConstantBuffer_D3D12(IDevice3D_D3D12* owner, uint32_t dataSize, const void* data) : IBuffer_D3D12(owner, D3D12Align(dataSize, 256), dataSize, data)
    {
    }
}