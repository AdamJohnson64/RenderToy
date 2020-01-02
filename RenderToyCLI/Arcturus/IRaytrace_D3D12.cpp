#include "D3D12Utility.h"
#include "ErrorD3D.h"
#include "IRaytrace_D3D12.h"
#include "Vector.h"
#include "generated.dxr.h"

#include <memory>

namespace Arcturus
{
    void TestRaytracer(IDevice3D_D3D12* m_owner, IRenderTarget_D3D12* renderTarget, IVertexBuffer_D3D12* vertexBuffer, IIndexBuffer_D3D12* indexBuffer)
    {
        ////////////////////////////////////////////////////////////////////////////////
        // PIPELINE - Build the pipeline with all ray shaders.
        ////////////////////////////////////////////////////////////////////////////////
        AutoRelease<ID3D12RootSignature> RootSignature;
        {
            uint32_t setupRange = 0;
            uint32_t setupOffset = 0;

	        D3D12_DESCRIPTOR_RANGE descDescriptorRange[32];
            
            /*
            // NOTE: We don't use any constant buffers yet.
            descDescriptorRange[setupRange].BaseShaderRegister = 0;
	        descDescriptorRange[setupRange].NumDescriptors = 2;
	        descDescriptorRange[setupRange].RegisterSpace = 0;
	        descDescriptorRange[setupRange].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
	        descDescriptorRange[setupRange].OffsetInDescriptorsFromTableStart = setupOffset;
            setupOffset += descDescriptorRange[setupRange].NumDescriptors;
            ++setupRange;
            */

	        descDescriptorRange[setupRange].BaseShaderRegister = 0;
	        descDescriptorRange[setupRange].NumDescriptors = 1;
	        descDescriptorRange[setupRange].RegisterSpace = 0;
	        descDescriptorRange[setupRange].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	        descDescriptorRange[setupRange].OffsetInDescriptorsFromTableStart = setupOffset;
            setupOffset += descDescriptorRange[setupRange].NumDescriptors;
            ++setupRange;

	        descDescriptorRange[setupRange].BaseShaderRegister = 0;
	        descDescriptorRange[setupRange].NumDescriptors = 1;
	        descDescriptorRange[setupRange].RegisterSpace = 0;
	        descDescriptorRange[setupRange].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	        descDescriptorRange[setupRange].OffsetInDescriptorsFromTableStart = setupOffset;
            setupOffset += descDescriptorRange[setupRange].NumDescriptors;
            ++setupRange;

	        D3D12_ROOT_PARAMETER descRootParameter = {};
	        descRootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	        descRootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	        descRootParameter.DescriptorTable.NumDescriptorRanges = setupRange;
	        descRootParameter.DescriptorTable.pDescriptorRanges = descDescriptorRange;

            D3D12_ROOT_SIGNATURE_DESC descSignature = {};
	        descSignature.NumParameters = 1;
	        descSignature.pParameters = &descRootParameter;
	        descSignature.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;

            AutoRelease<ID3DBlob> m_blob;
            AutoRelease<ID3DBlob> m_blobError;
            TRYD3D(D3D12SerializeRootSignature(&descSignature, D3D_ROOT_SIGNATURE_VERSION_1_0, &m_blob, &m_blobError));
            TRYD3D(m_owner->m_device->CreateRootSignature(0, m_blob->GetBufferPointer(), m_blob->GetBufferSize(), __uuidof(ID3D12RootSignature), (void**)&RootSignature));
            RootSignature->SetName(L"DXR Root Signature");
        }
        ////////////////////////////////////////////////////////////////////////////////
        // PIPELINE - Build the pipeline with all ray shaders.
        ////////////////////////////////////////////////////////////////////////////////
        AutoRelease<ID3D12StateObject> PipelineStateObject;
        {
            uint32_t setupSubobject = 0;

            D3D12_STATE_SUBOBJECT descSubobject[32] = {};
            
            D3D12_DXIL_LIBRARY_DESC descLibrary = {};
            descLibrary.DXILLibrary.pShaderBytecode = g_dxr_shader;
            descLibrary.DXILLibrary.BytecodeLength = sizeof(g_dxr_shader);
            descSubobject[setupSubobject].Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
            descSubobject[setupSubobject].pDesc = &descLibrary;
            ++setupSubobject;

            D3D12_RAYTRACING_SHADER_CONFIG descShaderConfig = {};
            descShaderConfig.MaxPayloadSizeInBytes = sizeof(float[3]) + sizeof(float); // RGB + Distance
            descShaderConfig.MaxAttributeSizeInBytes = D3D12_RAYTRACING_MAX_ATTRIBUTE_SIZE_IN_BYTES;
            descSubobject[setupSubobject].Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
            descSubobject[setupSubobject].pDesc = &descShaderConfig;
            ++setupSubobject;

            const WCHAR* shaderExports[] = { L"raygeneration", L"miss", L"HitGroup" };
	        D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION descSubobjectExports = {};
	        descSubobjectExports.NumExports = _countof(shaderExports);
	        descSubobjectExports.pExports = shaderExports;
	        descSubobjectExports.pSubobjectToAssociate = &descSubobject[setupSubobject - 1];
	        descSubobject[setupSubobject].Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
	        descSubobject[setupSubobject].pDesc = &descSubobjectExports;
            ++setupSubobject;

	        D3D12_STATE_SUBOBJECT descRootSignature = {};
	        descSubobject[setupSubobject].Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
	        descSubobject[setupSubobject].pDesc = &RootSignature;
            ++setupSubobject;

	        D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION descShaderRootSignature = {};
	        descShaderRootSignature.NumExports = _countof(shaderExports);
	        descShaderRootSignature.pExports = shaderExports;
	        descShaderRootSignature.pSubobjectToAssociate = &descSubobject[setupSubobject - 1];

	        D3D12_STATE_SUBOBJECT descShaderRootSignatureAssociation = {};
	        descSubobject[setupSubobject].Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
	        descSubobject[setupSubobject].pDesc = &descShaderRootSignature;
            ++setupSubobject;

            D3D12_RAYTRACING_PIPELINE_CONFIG descPipelineConfig = {};
            descPipelineConfig.MaxTraceRecursionDepth = 1;
            descSubobject[setupSubobject].Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
            descSubobject[setupSubobject].pDesc = &descPipelineConfig;
            ++setupSubobject;

            D3D12_HIT_GROUP_DESC descHitGroup = {};
            descHitGroup.HitGroupExport = L"HitGroup";
            descHitGroup.ClosestHitShaderImport = L"closesthit";
            descSubobject[setupSubobject].Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
            descSubobject[setupSubobject].pDesc = &descHitGroup;
            ++setupSubobject;

            D3D12_STATE_OBJECT_DESC descStateObject = {};
            descStateObject.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
            descStateObject.NumSubobjects = setupSubobject;
            descStateObject.pSubobjects = descSubobject;
            TRYD3D(m_owner->m_device->CreateStateObject(&descStateObject, __uuidof(ID3D12StateObject), (void**)&PipelineStateObject));
            PipelineStateObject->SetName(L"DXR Pipeline State");
        }
        ////////////////////////////////////////////////////////////////////////////////
        // BLAS - Build the bottom level acceleration structure.
        ////////////////////////////////////////////////////////////////////////////////
        // Create and initialize the BLAS.
        AutoRelease<ID3D12Resource1> ResourceBLAS;
        {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO descRaytracingPrebuild = {};
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS descRaytracingInputs = {};
            descRaytracingInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            descRaytracingInputs.NumDescs = 1;
            descRaytracingInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            D3D12_RAYTRACING_GEOMETRY_DESC descGeometry = {};
            descGeometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            descGeometry.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
            descGeometry.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
            descGeometry.Triangles.VertexFormat = DXGI_FORMAT_R32G32_FLOAT;
            descGeometry.Triangles.IndexCount = indexBuffer->m_size / sizeof(uint32_t);
            descGeometry.Triangles.VertexCount = vertexBuffer->m_size / vertexBuffer->m_strideSize;
            descGeometry.Triangles.IndexBuffer = indexBuffer->m_resource->GetGPUVirtualAddress();
            descGeometry.Triangles.VertexBuffer.StartAddress = vertexBuffer->m_resource->GetGPUVirtualAddress();
            descGeometry.Triangles.VertexBuffer.StrideInBytes = vertexBuffer->m_strideSize;
            descRaytracingInputs.pGeometryDescs = &descGeometry;
            m_owner->m_device->GetRaytracingAccelerationStructurePrebuildInfo(&descRaytracingInputs, &descRaytracingPrebuild);
            // Create the output and scratch buffers.
            ResourceBLAS.p = D3D12CreateBuffer(m_owner->m_device, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, descRaytracingPrebuild.ResultDataMaxSizeInBytes);
            AutoRelease<ID3D12Resource1> ResourceASScratch;
            ResourceASScratch.p = D3D12CreateBuffer(m_owner->m_device, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, descRaytracingPrebuild.ResultDataMaxSizeInBytes);
            // Build the acceleration structure.
            AutoRelease<ID3D12GraphicsCommandList5> UploadBLASCommandList;
            TRYD3D(m_owner->m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_owner->m_commandAllocator, nullptr, __uuidof(ID3D12GraphicsCommandList5), (void**)&UploadBLASCommandList));
            UploadBLASCommandList->SetName(L"BLAS Upload Command List");
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC descBuild = {};
            descBuild.DestAccelerationStructureData = ResourceBLAS->GetGPUVirtualAddress();
            descBuild.Inputs = descRaytracingInputs;
            descBuild.ScratchAccelerationStructureData = ResourceASScratch->GetGPUVirtualAddress();
            UploadBLASCommandList->BuildRaytracingAccelerationStructure(&descBuild, 0, nullptr);
            TRYD3D(UploadBLASCommandList->Close());
            m_owner->m_commandQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&UploadBLASCommandList);
            D3D12WaitForGPUIdle(m_owner->m_device, m_owner->m_commandQueue);
        }
        ////////////////////////////////////////////////////////////////////////////////
        // INSTANCE - Create the instancing table.
        ////////////////////////////////////////////////////////////////////////////////
        AutoRelease<ID3D12Resource1> ResourceInstance;
        {
            D3D12_RAYTRACING_INSTANCE_DESC DxrInstance = {};
            DxrInstance.Transform[0][0] = 1;
            DxrInstance.Transform[1][1] = 1;
            DxrInstance.Transform[2][2] = 1;
            DxrInstance.InstanceMask = 0xFF;
            DxrInstance.AccelerationStructure = ResourceBLAS->GetGPUVirtualAddress();
            ResourceInstance.p = D3D12CreateBuffer(m_owner->m_device, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, sizeof(DxrInstance), sizeof(DxrInstance), &DxrInstance, m_owner->m_commandQueue, m_owner->m_commandAllocator);
        }
        ////////////////////////////////////////////////////////////////////////////////
        // TLAS - Build the top level acceleration structure.
        ////////////////////////////////////////////////////////////////////////////////
        AutoRelease<ID3D12Resource1> ResourceTLAS;
        {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO descRaytracingPrebuild = {};
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS descRaytracingInputs = {};
            descRaytracingInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
            descRaytracingInputs.NumDescs = 1;
            descRaytracingInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            descRaytracingInputs.InstanceDescs = ResourceInstance->GetGPUVirtualAddress();
            m_owner->m_device->GetRaytracingAccelerationStructurePrebuildInfo(&descRaytracingInputs, &descRaytracingPrebuild);
            // Create the output and scratch buffers.
            ResourceTLAS.p = D3D12CreateBuffer(m_owner->m_device, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, descRaytracingPrebuild.ResultDataMaxSizeInBytes);
            AutoRelease<ID3D12Resource1> ResourceASScratch;
            ResourceASScratch.p = D3D12CreateBuffer(m_owner->m_device, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, descRaytracingPrebuild.ResultDataMaxSizeInBytes);
            // Build the acceleration structure.
            AutoRelease<ID3D12GraphicsCommandList5> UploadTLASCommandList;
            TRYD3D(m_owner->m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_owner->m_commandAllocator, nullptr, __uuidof(ID3D12GraphicsCommandList5), (void**)&UploadTLASCommandList));
            UploadTLASCommandList->SetName(L"TLAS Upload Command List");
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC descBuild = {};
            descBuild.DestAccelerationStructureData = ResourceTLAS->GetGPUVirtualAddress();
            descBuild.Inputs = descRaytracingInputs;
            descBuild.ScratchAccelerationStructureData = ResourceASScratch->GetGPUVirtualAddress();
            UploadTLASCommandList->BuildRaytracingAccelerationStructure(&descBuild, 0, nullptr);
            TRYD3D(UploadTLASCommandList->Close());
            m_owner->m_commandQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&UploadTLASCommandList);
            D3D12WaitForGPUIdle(m_owner->m_device, m_owner->m_commandQueue);
        }
        ////////////////////////////////////////////////////////////////////////////////
        // DESCRIPTOR HEAP - Shader resources (SRVs, UAVs, etc).
        ////////////////////////////////////////////////////////////////////////////////
        // Create descriptor heap with all SRVs and UAVs bound for render.
        AutoRelease<ID3D12DescriptorHeap> DescriptorHeap;
        {
            D3D12_DESCRIPTOR_HEAP_DESC descDescriptorHeap = {};
            descDescriptorHeap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            descDescriptorHeap.NumDescriptors = 2;
            descDescriptorHeap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            TRYD3D(m_owner->m_device->CreateDescriptorHeap(&descDescriptorHeap, __uuidof(ID3D12DescriptorHeap), (void**)&DescriptorHeap));
            DescriptorHeap->SetName(L"DXR Descriptor Heap");
        }
        // Create the unordered access buffer for raytracing output.
        AutoRelease<ID3D12Resource1> ResourceUAV;
        {
            D3D12_HEAP_PROPERTIES descHeapProperties = {};
            descHeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC descResource = {};
            descResource.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            descResource.Width = 256;
            descResource.Height = 256;
            descResource.DepthOrArraySize = 1;
            descResource.MipLevels = 1;
            descResource.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            descResource.SampleDesc.Count = 1;
            descResource.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            TRYD3D(m_owner->m_device->CreateCommittedResource1(&descHeapProperties, D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES, &descResource, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr, __uuidof(ID3D12Resource1), (void**)&ResourceUAV));
            ResourceUAV->SetName(L"DXR Output Texture2D UAV");
        }
        {
    	    D3D12_CPU_DESCRIPTOR_HANDLE descriptorBase = DescriptorHeap->GetCPUDescriptorHandleForHeapStart();
	        UINT descriptorElementSize = m_owner->m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            // Create the UAV for the raytracer output.
            {
                D3D12_UNORDERED_ACCESS_VIEW_DESC descUAV = {};
                descUAV.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                m_owner->m_device->CreateUnorderedAccessView(ResourceUAV, nullptr, &descUAV, descriptorBase);
                descriptorBase.ptr += descriptorElementSize;
            }
            // Create the SRV for the acceleration structure.
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC descSRV = {};
    	        descSRV.Format = DXGI_FORMAT_UNKNOWN;
    	        descSRV.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    	        descSRV.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    	        descSRV.RaytracingAccelerationStructure.Location = ResourceTLAS->GetGPUVirtualAddress();
    	        m_owner->m_device->CreateShaderResourceView(nullptr, &descSRV, descriptorBase);
                descriptorBase.ptr += descriptorElementSize;
            }
        }
        ////////////////////////////////////////////////////////////////////////////////
        // SHADER TABLE - Create a table of all shaders for the raytracer.
        ////////////////////////////////////////////////////////////////////////////////
        AutoRelease<ID3D12Resource1> ResourceShaderTable;
        {
            AutoRelease<ID3D12StateObjectProperties> stateObjectProperties;
            TRYD3D(PipelineStateObject->QueryInterface<ID3D12StateObjectProperties>(&stateObjectProperties));
            uint32_t shaderEntrySize = 64;
            uint32_t shaderTableSize = shaderEntrySize * 3;
            std::unique_ptr<uint8_t[]> shaderTableCPU(new uint8_t[shaderTableSize]);
            memset(&shaderTableCPU[0], 0, shaderTableSize);
            // Shader Index 0 - Ray Generation Shader
            memcpy(&shaderTableCPU[shaderEntrySize * 0], stateObjectProperties->GetShaderIdentifier(L"raygeneration"), D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            *reinterpret_cast<D3D12_GPU_DESCRIPTOR_HANDLE*>(&shaderTableCPU[shaderEntrySize * 0] + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES) = DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
            // Shader Index 1 - Miss Shader
            memcpy(&shaderTableCPU[shaderEntrySize * 1], stateObjectProperties->GetShaderIdentifier(L"miss"), D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            *reinterpret_cast<D3D12_GPU_DESCRIPTOR_HANDLE*>(&shaderTableCPU[shaderEntrySize * 1] + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES) = DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
            // Shader Index 2 - Hit Shader
            memcpy(&shaderTableCPU[shaderEntrySize * 2], stateObjectProperties->GetShaderIdentifier(L"HitGroup"), D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            *reinterpret_cast<D3D12_GPU_DESCRIPTOR_HANDLE*>(&shaderTableCPU[shaderEntrySize * 2] + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES) = DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
            ResourceShaderTable.p = D3D12CreateBuffer(m_owner->m_device, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, shaderTableSize, shaderTableSize, &shaderTableCPU[0], m_owner->m_commandQueue, m_owner->m_commandAllocator);
            ResourceShaderTable->SetName(L"DXR Shader Table");
        }
        ////////////////////////////////////////////////////////////////////////////////
        // RAYTRACE - Finally call the raytracer and generate the frame.
        ////////////////////////////////////////////////////////////////////////////////
        {
            AutoRelease<ID3D12GraphicsCommandList5> RaytraceCommandList;
            TRYD3D(m_owner->m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_owner->m_commandAllocator, nullptr, __uuidof(ID3D12GraphicsCommandList5), (void**)&RaytraceCommandList));
            RaytraceCommandList->SetName(L"DXR Raytrace Command List");
            RaytraceCommandList->SetDescriptorHeaps(1, &DescriptorHeap);
            RaytraceCommandList->SetPipelineState1(PipelineStateObject);
            {
                D3D12_DISPATCH_RAYS_DESC descDispatchRays = {};
                descDispatchRays.RayGenerationShaderRecord.StartAddress = ResourceShaderTable->GetGPUVirtualAddress() + 64 * 0;
                descDispatchRays.RayGenerationShaderRecord.SizeInBytes = 64;
                descDispatchRays.MissShaderTable.StartAddress = ResourceShaderTable->GetGPUVirtualAddress() + 64 * 1;
                descDispatchRays.MissShaderTable.SizeInBytes = 64;
                descDispatchRays.MissShaderTable.StrideInBytes = 64;
                descDispatchRays.HitGroupTable.StartAddress = ResourceShaderTable->GetGPUVirtualAddress() + 64 * 2;
                descDispatchRays.HitGroupTable.SizeInBytes = 64;
                descDispatchRays.HitGroupTable.StrideInBytes = 64;
                descDispatchRays.Width = 256;
                descDispatchRays.Height = 256;
                descDispatchRays.Depth = 1;
                RaytraceCommandList->DispatchRays(&descDispatchRays);
            }
            {
                D3D12_RESOURCE_BARRIER descResourceBarrier = {};
                descResourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                descResourceBarrier.Transition.pResource = ResourceUAV;
                descResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                descResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                descResourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                RaytraceCommandList->ResourceBarrier(1, &descResourceBarrier);
            }
            {
                D3D12_RESOURCE_BARRIER descResourceBarrier = {};
                descResourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                descResourceBarrier.Transition.pResource = renderTarget->m_resource;
                descResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                descResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                descResourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                RaytraceCommandList->ResourceBarrier(1, &descResourceBarrier);
            }
            RaytraceCommandList->CopyResource(renderTarget->m_resource, ResourceUAV);
            {
                D3D12_RESOURCE_BARRIER descResourceBarrier = {};
                descResourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                descResourceBarrier.Transition.pResource = ResourceUAV;
                descResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
                descResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
                descResourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                RaytraceCommandList->ResourceBarrier(1, &descResourceBarrier);
            }
            {
                D3D12_RESOURCE_BARRIER descResourceBarrier = {};
                descResourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                descResourceBarrier.Transition.pResource = renderTarget->m_resource;
                descResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                descResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
                descResourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                RaytraceCommandList->ResourceBarrier(1, &descResourceBarrier);
            }
            TRYD3D(RaytraceCommandList->Close());
            m_owner->m_commandQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&RaytraceCommandList);
            D3D12WaitForGPUIdle(m_owner->m_device, m_owner->m_commandQueue);
        }
    }
}