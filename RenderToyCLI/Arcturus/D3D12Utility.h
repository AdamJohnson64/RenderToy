#pragma once

#include <stdint.h>
#include <d3d12.h>

namespace Arcturus
{
    uint32_t D3D12Align(uint32_t size, uint32_t alignSize);

    ID3D12Resource1* D3D12CreateBuffer(ID3D12Device* device, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, uint32_t bufferSize);

    ID3D12Resource1* D3D12CreateBuffer(ID3D12Device* device, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, uint32_t bufferSize, uint32_t dataSize, const void* data, ID3D12CommandQueue* commandQueue, ID3D12CommandAllocator *commandAllocator);

    void D3D12WaitForGPUIdle(ID3D12Device* device, ID3D12CommandQueue* queue);
}