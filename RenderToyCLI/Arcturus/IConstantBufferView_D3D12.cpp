#include "IConstantBuffer_D3D12.h"
#include "IConstantBufferView_D3D12.h"
#include "IDevice3D_D3D12.h"

#include <d3d12.h>

namespace Arcturus
{
    IConstantBufferView_D3D12::IConstantBufferView_D3D12(IDevice3D_D3D12* owner, IConstantBuffer_D3D12* constantBuffer) : m_owner(owner)
    {
    	D3D12_CPU_DESCRIPTOR_HANDLE descriptorBase = m_owner->m_descriptorHeapGPU->GetCPUDescriptorHandleForHeapStart();
	    UINT descriptorElementSize = m_owner->m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        // Create the CBV for the vertex transform.
        {
            D3D12_CONSTANT_BUFFER_VIEW_DESC descCBV = {};
            descCBV.BufferLocation = constantBuffer->m_resource->GetGPUVirtualAddress();
            descCBV.SizeInBytes = 256;
            m_owner->m_device->CreateConstantBufferView(&descCBV, descriptorBase);
            descriptorBase.ptr += descriptorElementSize;
        }
    }
}