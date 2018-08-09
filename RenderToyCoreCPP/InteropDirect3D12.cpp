////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

#include <memory>
#include <msclr\marshal_cppstd.h>
#include <d3d9.h>
#include <d3d12.h>
#include "InteropCommon.h"

#define TRY_D3D(FUNCTION) if (FUNCTION != S_OK) throw gcnew System::Exception(#FUNCTION);

using namespace System::Runtime::InteropServices;

namespace RenderToy
{
	// Shim access to revised serialization.
	public ref class D3D12Shim
	{
	public:
		static void DescriptorHeap_GetCPUDescriptorHandleForHeapStart(RenderToyCOM::ID3D12DescriptorHeap ^descriptorHeap, RenderToyCOM::D3D12_CPU_DESCRIPTOR_HANDLE %cpudesc)
		{
			auto unmanaged = Marshal::GetComInterfaceForObject(descriptorHeap, RenderToyCOM::ID3D12DescriptorHeap::typeid);
			Marshal::AddRef(unmanaged);
			auto result = ((ID3D12DescriptorHeap*)unmanaged.ToPointer())->GetCPUDescriptorHandleForHeapStart();
			cpudesc.ptr = result.ptr;
		}
	};
	#pragma region - Direct3D12 Structures -
	public value struct D3D12ResourceTransitionBarrier
	{
		RenderToyCOM::ID3D12Resource^		pResource;
		UINT								Subresource;
		RenderToyCOM::D3D12_RESOURCE_STATES	StateBefore;
		RenderToyCOM::D3D12_RESOURCE_STATES	StateAfter;
	};
	public value struct D3D12ResourceBarrier
	{
		RenderToyCOM::D3D12_RESOURCE_BARRIER_TYPE		Type;
		RenderToyCOM::D3D12_RESOURCE_BARRIER_FLAGS		Flags;
		D3D12ResourceTransitionBarrier					Transition;
	};
	public value struct D3D12CachedPipelineState
	{
		cli::array<byte>^	pCachedBlob;
	};
	public value struct D3D12ClearValue
	{
		RenderToyCOM::DXGI_FORMAT Format;
		float R, G, B, A;
	};
	public value struct D3D12CPUDescriptorHandle
	{
		System::IntPtr ptr;
	};
	public value struct D3D12InputElementDesc
	{
		System::String^								SemanticName;
		UINT										SemanticIndex;
		RenderToyCOM::DXGI_FORMAT					Format;
		UINT										InputSlot;
		UINT										AlignedByteOffset;
		RenderToyCOM::D3D12_INPUT_CLASSIFICATION	InputSlotClass;
		UINT										InstanceDataStepRate;
	};
	public value struct D3D12InputLayoutDesc
	{
		cli::array<D3D12InputElementDesc>^	pInputElementDescs;
	};
	public value struct D3D12SODeclarationEntry
	{
		UINT			Stream;
		System::String^ SemanticName;
		UINT			SemanticIndex;
		BYTE			StartComponent;
		BYTE			ComponentCount;
		BYTE			OutputSlot;
	};
	public value struct D3D12StreamOutputDesc
	{
		cli::array<D3D12SODeclarationEntry>^	pSODeclaration;
		cli::array<UINT>^						pBufferStrides;
		UINT									RasterizedStream;
	};
	public value struct D3D12GraphicsPipelineStateDesc
	{
		RenderToyCOM::ID3D12RootSignature^					pRootSignature;
		cli::array<byte>^									VS;
		cli::array<byte>^									PS;
		cli::array<byte>^									DS;
		cli::array<byte>^									HS;
		cli::array<byte>^									GS;
		RenderToyCOM::D3D12_STREAM_OUTPUT_DESC				StreamOutput;
		RenderToyCOM::D3D12_BLEND_DESC						BlendState;
		UINT												SampleMask;
		RenderToyCOM::D3D12_RASTERIZER_DESC					RasterizerState;
		RenderToyCOM::D3D12_DEPTH_STENCIL_DESC				DepthStencilState;
		D3D12InputLayoutDesc								InputLayout;
		RenderToyCOM::D3D12_INDEX_BUFFER_STRIP_CUT_VALUE	IBStripCutValue;
		RenderToyCOM::D3D12_PRIMITIVE_TOPOLOGY_TYPE			PrimitiveTopologyType;
		UINT												NumRenderTargets;
		RenderToyCOM::DXGI_FORMAT							RTVFormats0;
		RenderToyCOM::DXGI_FORMAT							RTVFormats1;
		RenderToyCOM::DXGI_FORMAT							RTVFormats2;
		RenderToyCOM::DXGI_FORMAT							RTVFormats3;
		RenderToyCOM::DXGI_FORMAT							RTVFormats4;
		RenderToyCOM::DXGI_FORMAT							RTVFormats5;
		RenderToyCOM::DXGI_FORMAT							RTVFormats6;
		RenderToyCOM::DXGI_FORMAT							RTVFormats7;
		RenderToyCOM::DXGI_FORMAT							DSVFormat;
		RenderToyCOM::DXGI_SAMPLE_DESC						SampleDesc;
		UINT												NodeMask;
		RenderToyCOM::D3D12_CACHED_PIPELINE_STATE			CachedPSO;
		RenderToyCOM::D3D12_PIPELINE_STATE_FLAGS			Flags;
	};
	#pragma endregion
	#pragma region - D3D12GraphicsCommandList1 -
	public ref class D3D12GraphicsCommandList1 : public COMWrapper<ID3D12GraphicsCommandList1>
	{
	public:
		D3D12GraphicsCommandList1(ID3D12GraphicsCommandList1 *pObj) : COMWrapper(pObj)
		{
		}
		void ClearRenderTargetView(D3D12CPUDescriptorHandle RenderTargetView, float R, float G, float B, float A)
		{
			float rgba[4] = { R, G, B, A };
			D3D12_CPU_DESCRIPTOR_HANDLE desc;
			desc.ptr = (SIZE_T)RenderTargetView.ptr.ToPointer();
			WrappedInterface()->ClearRenderTargetView(desc, rgba, 0U, nullptr);
		};
		void Close()
		{
			TRY_D3D(WrappedInterface()->Close());
		}
		void CopyResource(RenderToyCOM::ID3D12Resource ^pDstResource, RenderToyCOM::ID3D12Resource ^pSrcResource)
		{
			WrappedInterface()->CopyResource((ID3D12Resource*)Marshal::GetComInterfaceForObject(pDstResource, RenderToyCOM::ID3D12Resource::typeid).ToPointer(), (ID3D12Resource*)Marshal::GetComInterfaceForObject(pSrcResource, RenderToyCOM::ID3D12Resource::typeid).ToPointer());
		}
		void DrawInstanced(UINT VertexCountPerInstance, UINT InstanceCount, UINT StartVertexLocation, UINT StartInstanceLocation)
		{
			WrappedInterface()->DrawInstanced(VertexCountPerInstance, InstanceCount, StartVertexLocation, StartInstanceLocation);
		}
		void IASetPrimitiveTopology(RenderToyCOM::D3D_PRIMITIVE_TOPOLOGY PrimitiveTopology)
		{
			WrappedInterface()->IASetPrimitiveTopology((D3D12_PRIMITIVE_TOPOLOGY)PrimitiveTopology);
		}
		void IASetVertexBuffers(UINT StartSlot, cli::array<RenderToyCOM::D3D12_VERTEX_BUFFER_VIEW> ^pViews)
		{
			pin_ptr<RenderToyCOM::D3D12_VERTEX_BUFFER_VIEW> pViewsM = &pViews[0];
			WrappedInterface()->IASetVertexBuffers(StartSlot, pViews->Length, (D3D12_VERTEX_BUFFER_VIEW*)&pViewsM[0]);
		}
		void OMSetRenderTargets(cli::array<D3D12CPUDescriptorHandle> ^pRenderTargetDescriptors, BOOL RTsSingleHandleToDescriptorRange, System::Nullable<D3D12CPUDescriptorHandle> pDepthStencilDescriptor)
		{
			std::unique_ptr<D3D12_CPU_DESCRIPTOR_HANDLE[]> pRenderTargetDescriptorsM(new D3D12_CPU_DESCRIPTOR_HANDLE[pRenderTargetDescriptors->Length]);
			for (int i = 0; i < pRenderTargetDescriptors->Length; ++i)
			{
				pRenderTargetDescriptorsM[i].ptr = (SIZE_T)pRenderTargetDescriptors[i].ptr.ToPointer();
			}
			D3D12_CPU_DESCRIPTOR_HANDLE pDepthStencilDescriptorM;
			pDepthStencilDescriptorM.ptr = (SIZE_T)(pDepthStencilDescriptor.HasValue ? pDepthStencilDescriptor.Value.ptr.ToPointer() : nullptr);
			WrappedInterface()->OMSetRenderTargets(pRenderTargetDescriptors->Length, pRenderTargetDescriptorsM.get(), RTsSingleHandleToDescriptorRange, pDepthStencilDescriptor.HasValue ? &pDepthStencilDescriptorM : nullptr);
		}
		void RSSetScissorRects(cli::array<RenderToyCOM::tagRECT> ^pRects)
		{
			pin_ptr<RenderToyCOM::tagRECT> pRectsM = &pRects[0];
			WrappedInterface()->RSSetScissorRects(pRects->Length, reinterpret_cast<D3D12_RECT*>(&pRectsM[0]));
		}
		void RSSetViewports(cli::array<RenderToyCOM::D3D12_VIEWPORT> ^pViewports)
		{
			pin_ptr<RenderToyCOM::D3D12_VIEWPORT> pViewportsM = &pViewports[0];
			WrappedInterface()->RSSetViewports(pViewports->Length, reinterpret_cast<D3D12_VIEWPORT*>(&pViewportsM[0]));
		}
		void SetDescriptorHeaps(cli::array<RenderToyCOM::ID3D12DescriptorHeap^> ^ppDescriptorHeaps)
		{
			std::unique_ptr<ID3D12DescriptorHeap*[]> ppDescriptorHeapsM(new ID3D12DescriptorHeap*[ppDescriptorHeaps->Length]);
			for (int i = 0; i < ppDescriptorHeaps->Length; ++i)
			{
				ppDescriptorHeapsM[i] = (ID3D12DescriptorHeap*)Marshal::GetComInterfaceForObject(ppDescriptorHeaps[i], RenderToyCOM::ID3D12DescriptorHeap::typeid).ToPointer();
			}
			WrappedInterface()->SetDescriptorHeaps(ppDescriptorHeaps->Length, ppDescriptorHeapsM.get());
		}
		void SetGraphicsRoot32BitConstants(UINT RootParameterIndex, UINT Num32BitValuesToSet, cli::array<float> ^pSrcData, UINT DestOffsetIn32BitValues)
		{
			pin_ptr<float> pSrcDataM = &pSrcData[0];
			WrappedInterface()->SetGraphicsRoot32BitConstants(RootParameterIndex, Num32BitValuesToSet, &pSrcDataM[0], DestOffsetIn32BitValues);
		}
		void SetGraphicsRootSignature(RenderToyCOM::ID3D12RootSignature ^pRootSignature)
		{
			WrappedInterface()->SetGraphicsRootSignature((ID3D12RootSignature*)Marshal::GetComInterfaceForObject(pRootSignature, RenderToyCOM::ID3D12RootSignature::typeid).ToPointer());
		}
		void SetPipelineState(RenderToyCOM::ID3D12PipelineState ^pPipelineState)
		{
			WrappedInterface()->SetPipelineState((ID3D12PipelineState*)Marshal::GetComInterfaceForObject(pPipelineState, RenderToyCOM::ID3D12PipelineState::typeid).ToPointer());
		}
		void Reset(RenderToyCOM::ID3D12CommandAllocator ^pAllocator, RenderToyCOM::ID3D12PipelineState ^pInitialState)
		{
			TRY_D3D(WrappedInterface()->Reset((ID3D12CommandAllocator*)Marshal::GetComInterfaceForObject(pAllocator, RenderToyCOM::ID3D12CommandAllocator::typeid).ToPointer(), (ID3D12PipelineState*)Marshal::GetComInterfaceForObject(pInitialState, RenderToyCOM::ID3D12PipelineState::typeid).ToPointer()));
		}
		void ResourceBarrier(cli::array<D3D12ResourceBarrier> ^pBarriers)
		{
			std::unique_ptr<D3D12_RESOURCE_BARRIER[]> pBarriersM(new D3D12_RESOURCE_BARRIER[pBarriers->Length]);
			for (int i = 0; i < pBarriers->Length; ++i)
			{
				pBarriersM[i].Type = (D3D12_RESOURCE_BARRIER_TYPE)pBarriers[i].Type;
				pBarriersM[i].Flags = (D3D12_RESOURCE_BARRIER_FLAGS)pBarriers[i].Flags;
				pBarriersM[i].Transition.pResource = (ID3D12Resource*)Marshal::GetComInterfaceForObject(pBarriers[i].Transition.pResource, RenderToyCOM::ID3D12Resource::typeid).ToPointer();
				pBarriersM[i].Transition.Subresource = pBarriers[i].Transition.Subresource;
				pBarriersM[i].Transition.StateBefore = (D3D12_RESOURCE_STATES)pBarriers[i].Transition.StateBefore;
				pBarriersM[i].Transition.StateAfter = (D3D12_RESOURCE_STATES)pBarriers[i].Transition.StateAfter;
			}
			WrappedInterface()->ResourceBarrier(pBarriers->Length, pBarriersM.get());
		}
	};
	#pragma endregion
	#pragma region - D3D12Device -
	public ref class D3D12Device : public COMWrapper<ID3D12Device3>
	{
	public:
		D3D12Device(ID3D12Device3 *obj) : COMWrapper(obj)
		{
		}
		RenderToyCOM::ID3D12CommandAllocator^ CreateCommandAllocator(RenderToyCOM::D3D12_COMMAND_LIST_TYPE type)
		{
			void *ppCommandAllocator = nullptr;
			TRY_D3D(WrappedInterface()->CreateCommandAllocator((D3D12_COMMAND_LIST_TYPE)type, __uuidof(ID3D12CommandAllocator), &ppCommandAllocator));
			return (RenderToyCOM::ID3D12CommandAllocator^)Marshal::GetTypedObjectForIUnknown(System::IntPtr(ppCommandAllocator), RenderToyCOM::ID3D12CommandAllocator::typeid);
		}
		RenderToyCOM::ID3D12GraphicsCommandList1^ CreateCommandList(UINT nodeMask, RenderToyCOM::D3D12_COMMAND_LIST_TYPE type, RenderToyCOM::ID3D12CommandAllocator ^pCommandAllocator, RenderToyCOM::ID3D12PipelineState ^pInitialState)
		{
			void *ppCommandList = nullptr;
			TRY_D3D(WrappedInterface()->CreateCommandList(nodeMask, (D3D12_COMMAND_LIST_TYPE)type, (ID3D12CommandAllocator*)Marshal::GetComInterfaceForObject(pCommandAllocator, RenderToyCOM::ID3D12CommandAllocator::typeid).ToPointer(), (ID3D12PipelineState*)Marshal::GetComInterfaceForObject(pInitialState, RenderToyCOM::ID3D12PipelineState::typeid).ToPointer(), __uuidof(ID3D12GraphicsCommandList1), &ppCommandList));
			return (RenderToyCOM::ID3D12GraphicsCommandList1^)Marshal::GetTypedObjectForIUnknown(System::IntPtr(ppCommandList), RenderToyCOM::ID3D12GraphicsCommandList1::typeid);
		}
		RenderToyCOM::ID3D12CommandQueue^ CreateCommandQueue(RenderToyCOM::D3D12_COMMAND_QUEUE_DESC pDesc)
		{
			void *ppCommandQueue = nullptr;
			TRY_D3D(WrappedInterface()->CreateCommandQueue((D3D12_COMMAND_QUEUE_DESC*)&pDesc, __uuidof(ID3D12CommandQueue), &ppCommandQueue));
			return (RenderToyCOM::ID3D12CommandQueue^)Marshal::GetTypedObjectForIUnknown(System::IntPtr(ppCommandQueue), RenderToyCOM::ID3D12CommandQueue::typeid);
		}
		RenderToyCOM::ID3D12Resource^ CreateCommittedResource(RenderToyCOM::D3D12_HEAP_PROPERTIES pHeapProperties, RenderToyCOM::D3D12_HEAP_FLAGS HeapFlags, RenderToyCOM::D3D12_RESOURCE_DESC pDesc, RenderToyCOM::D3D12_RESOURCE_STATES InitialResourceState, System::Nullable<D3D12ClearValue> pOptimizedClearValue)
		{
			void *ppvResource = nullptr;
			TRY_D3D(WrappedInterface()->CreateCommittedResource(reinterpret_cast<D3D12_HEAP_PROPERTIES*>(&pHeapProperties), (D3D12_HEAP_FLAGS)HeapFlags, reinterpret_cast<D3D12_RESOURCE_DESC*>(&pDesc), (D3D12_RESOURCE_STATES)InitialResourceState, pOptimizedClearValue.HasValue ? reinterpret_cast<D3D12_CLEAR_VALUE*>(&pOptimizedClearValue.Value) : nullptr, __uuidof(ID3D12Resource), &ppvResource));
			return (RenderToyCOM::ID3D12Resource^)Marshal::GetTypedObjectForIUnknown(System::IntPtr(ppvResource), RenderToyCOM::ID3D12Resource::typeid);
		}
		RenderToyCOM::ID3D12DescriptorHeap^ CreateDescriptorHeap(RenderToyCOM::D3D12_DESCRIPTOR_HEAP_DESC pDescriptorHeapDesc)
		{
			void *ppvHeap = nullptr;
			TRY_D3D(WrappedInterface()->CreateDescriptorHeap((D3D12_DESCRIPTOR_HEAP_DESC*)&pDescriptorHeapDesc, __uuidof(ID3D12DescriptorHeap), &ppvHeap));
			return (RenderToyCOM::ID3D12DescriptorHeap^)Marshal::GetTypedObjectForIUnknown(System::IntPtr(ppvHeap), RenderToyCOM::ID3D12DescriptorHeap::typeid);
		}
		RenderToyCOM::ID3D12Fence^ CreateFence(UINT64 InitialValue, RenderToyCOM::D3D12_FENCE_FLAGS Flags)
		{
			void *ppFence = nullptr;
			TRY_D3D(WrappedInterface()->CreateFence(InitialValue, (D3D12_FENCE_FLAGS)Flags, __uuidof(ID3D12Fence), &ppFence));
			return (RenderToyCOM::ID3D12Fence^)Marshal::GetTypedObjectForIUnknown(System::IntPtr(ppFence), RenderToyCOM::ID3D12Fence::typeid);
		}
		RenderToyCOM::ID3D12PipelineState^ CreateGraphicsPipelineState(D3D12GraphicsPipelineStateDesc pDesc)
		{
			void *ppPipelineState = nullptr;
			D3D12_GRAPHICS_PIPELINE_STATE_DESC pDesc2 = { 0 };
			pDesc2.pRootSignature = (ID3D12RootSignature*)Marshal::GetComInterfaceForObject(pDesc.pRootSignature, RenderToyCOM::ID3D12RootSignature::typeid).ToPointer();
			pin_ptr<byte> vs(&pDesc.VS[0]);
			pDesc2.VS.pShaderBytecode = vs;
			pDesc2.VS.BytecodeLength = pDesc.VS == nullptr ? 0 : pDesc.VS->Length;
			pin_ptr<byte> ps(&pDesc.PS[0]);
			pDesc2.PS.pShaderBytecode = ps;
			pDesc2.PS.BytecodeLength = pDesc.PS == nullptr ? 0 : pDesc.PS->Length;
			pDesc2.DS.BytecodeLength = pDesc.DS == nullptr ? 0 : pDesc.DS->Length;
			pDesc2.HS.BytecodeLength = pDesc.HS == nullptr ? 0 : pDesc.HS->Length;
			pDesc2.GS.BytecodeLength = pDesc.GS == nullptr ? 0 : pDesc.GS->Length;
			pDesc2.BlendState.AlphaToCoverageEnable = pDesc.BlendState.AlphaToCoverageEnable;
			pDesc2.BlendState.IndependentBlendEnable = pDesc.BlendState.IndependentBlendEnable;
			pDesc2.BlendState.RenderTarget[0].BlendEnable = pDesc.BlendState.RenderTarget[0].BlendEnable;
			pDesc2.BlendState.RenderTarget[0].LogicOpEnable = pDesc.BlendState.RenderTarget[0].LogicOpEnable;
			pDesc2.BlendState.RenderTarget[0].SrcBlend = (D3D12_BLEND)pDesc.BlendState.RenderTarget[0].SrcBlend;
			pDesc2.BlendState.RenderTarget[0].DestBlend = (D3D12_BLEND)pDesc.BlendState.RenderTarget[0].DestBlend;
			pDesc2.BlendState.RenderTarget[0].BlendOp = (D3D12_BLEND_OP)pDesc.BlendState.RenderTarget[0].BlendOp;
			pDesc2.BlendState.RenderTarget[0].SrcBlendAlpha = (D3D12_BLEND)pDesc.BlendState.RenderTarget[0].SrcBlendAlpha;
			pDesc2.BlendState.RenderTarget[0].DestBlendAlpha = (D3D12_BLEND)pDesc.BlendState.RenderTarget[0].DestBlendAlpha;
			pDesc2.BlendState.RenderTarget[0].BlendOpAlpha = (D3D12_BLEND_OP)pDesc.BlendState.RenderTarget[0].BlendOpAlpha;
			pDesc2.BlendState.RenderTarget[0].LogicOp = (D3D12_LOGIC_OP)pDesc.BlendState.RenderTarget[0].LogicOp;
			pDesc2.BlendState.RenderTarget[0].RenderTargetWriteMask = pDesc.BlendState.RenderTarget[0].RenderTargetWriteMask;
			pDesc2.SampleMask = pDesc.SampleMask;
			pDesc2.RasterizerState.CullMode = (D3D12_CULL_MODE)pDesc.RasterizerState.CullMode;
			pDesc2.RasterizerState.FillMode = (D3D12_FILL_MODE)pDesc.RasterizerState.FillMode;
			pDesc2.SampleDesc.Count = pDesc.SampleDesc.Count;
			pDesc2.SampleDesc.Quality = pDesc.SampleDesc.Quality;
			std::unique_ptr<D3D12_INPUT_ELEMENT_DESC[]> pInputElementDescsM(new D3D12_INPUT_ELEMENT_DESC[pDesc.InputLayout.pInputElementDescs->Length]);
			msclr::interop::marshal_context ctx;
			for (int i = 0; i < pDesc.InputLayout.pInputElementDescs->Length; ++i)
			{
				pInputElementDescsM[i].SemanticName = ctx.marshal_as<LPCSTR>(pDesc.InputLayout.pInputElementDescs[i].SemanticName);
				pInputElementDescsM[i].SemanticIndex = pDesc.InputLayout.pInputElementDescs[i].SemanticIndex;
				pInputElementDescsM[i].Format = (DXGI_FORMAT)pDesc.InputLayout.pInputElementDescs[i].Format;
				pInputElementDescsM[i].InputSlot = pDesc.InputLayout.pInputElementDescs[i].InputSlot;
				pInputElementDescsM[i].AlignedByteOffset = pDesc.InputLayout.pInputElementDescs[i].AlignedByteOffset;
				pInputElementDescsM[i].InputSlotClass = (D3D12_INPUT_CLASSIFICATION)pDesc.InputLayout.pInputElementDescs[i].InputSlotClass;
				pInputElementDescsM[i].InstanceDataStepRate = pDesc.InputLayout.pInputElementDescs[i].InstanceDataStepRate;
			}
			pDesc2.InputLayout.pInputElementDescs = &pInputElementDescsM[0];
			pDesc2.InputLayout.NumElements = pDesc.InputLayout.pInputElementDescs->Length;
			pDesc2.PrimitiveTopologyType = (D3D12_PRIMITIVE_TOPOLOGY_TYPE)pDesc.PrimitiveTopologyType;
			pDesc2.NumRenderTargets = pDesc.NumRenderTargets;
			pDesc2.RTVFormats[0] = (DXGI_FORMAT)pDesc.RTVFormats0;
			TRY_D3D(WrappedInterface()->CreateGraphicsPipelineState(&pDesc2, __uuidof(ID3D12PipelineState), &ppPipelineState));
			return (RenderToyCOM::ID3D12PipelineState^)Marshal::GetTypedObjectForIUnknown(System::IntPtr(ppPipelineState), RenderToyCOM::ID3D12PipelineState::typeid);
		}
		void CreateRenderTargetView(RenderToyCOM::ID3D12Resource ^pResource, RenderToyCOM::D3D12_RENDER_TARGET_VIEW_DESC pDesc, RenderToyCOM::D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor)
		{
			D3D12_RENDER_TARGET_VIEW_DESC pDesc2;
			pDesc2.Format = (DXGI_FORMAT)pDesc.Format;
			pDesc2.ViewDimension = (D3D12_RTV_DIMENSION)pDesc.ViewDimension;
			pDesc2.Texture2D.MipSlice = pDesc.__MIDL____MIDL_itf_RenderToy_0001_00180005.Texture2D.MipSlice;
			pDesc2.Texture2D.PlaneSlice = pDesc.__MIDL____MIDL_itf_RenderToy_0001_00180005.Texture2D.PlaneSlice;
			D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor2;
			DestDescriptor2.ptr = (SIZE_T)DestDescriptor.ptr;
			WrappedInterface()->CreateRenderTargetView((ID3D12Resource*)Marshal::GetComInterfaceForObject(pResource, RenderToyCOM::ID3D12Resource::typeid).ToPointer(), &pDesc2, DestDescriptor2);
		}
		RenderToyCOM::ID3D12RootSignature^ CreateRootSignature(UINT nodeMask, cli::array<byte> ^pBlobWithRootSignature)
		{
			void* ppvRootSignature = nullptr;
			pin_ptr<byte> pBlobWithRootSignatureM = &pBlobWithRootSignature[0];
			TRY_D3D(WrappedInterface()->CreateRootSignature(nodeMask, pBlobWithRootSignatureM, pBlobWithRootSignature->Length, __uuidof(ID3D12RootSignature), &ppvRootSignature));
			return (RenderToyCOM::ID3D12RootSignature^)Marshal::GetTypedObjectForIUnknown(System::IntPtr(ppvRootSignature), RenderToyCOM::ID3D12RootSignature::typeid);
		}
	};
	#pragma endregion
	#pragma region - D3D12Debug -
	public ref class D3D12Debug : public COMWrapper<ID3D12Debug>
	{
	public:
		D3D12Debug(ID3D12Debug *obj) : COMWrapper(obj)
		{
		}
		void EnableDebugLayer()
		{
			WrappedInterface()->EnableDebugLayer();
		}
	};
	#pragma endregion
	#pragma region - Direct3D12 Global Functions -
	public ref class Direct3D12
	{
	public:
		static D3D12Debug^ D3D12GetDebugInterface()
		{
			void *ppvDebug = nullptr;
			TRY_D3D(::D3D12GetDebugInterface(__uuidof(ID3D12Debug), &ppvDebug));
			return gcnew D3D12Debug(reinterpret_cast<ID3D12Debug*>(ppvDebug));
		}
		static D3D12Device^ D3D12CreateDevice()
		{
			void *ppDevice = nullptr;
			TRY_D3D(::D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device3), &ppDevice));
			return gcnew D3D12Device(reinterpret_cast<ID3D12Device3*>(ppDevice));
		}
	};
	#pragma endregion
}