#include <memory>
#include <msclr\marshal_cppstd.h>
#include <d3d11_4.h>
#include "InteropCommon.h"

#define TRY_D3D(FUNCTION) if (FUNCTION != S_OK) throw gcnew System::Exception(#FUNCTION);

namespace RenderToy
{
	public enum struct D3D11BindFlag
	{
		VertexBuffer = D3D11_BIND_VERTEX_BUFFER,
		IndexBuffer = D3D11_BIND_INDEX_BUFFER,
		ConstantBuffer = D3D11_BIND_CONSTANT_BUFFER,
		ShaderResource = D3D11_BIND_SHADER_RESOURCE,
		StreamOutput = D3D11_BIND_STREAM_OUTPUT,
		RenderTarget = D3D11_BIND_RENDER_TARGET,
		DepthStencil = D3D11_BIND_DEPTH_STENCIL,
		UnorderedAccess = D3D11_BIND_UNORDERED_ACCESS,
		Decoder = D3D11_BIND_DECODER,
		VideoEncoder = D3D11_BIND_VIDEO_ENCODER,
	};
	public enum struct D3D11ComparisonFunc
	{
		Never = D3D11_COMPARISON_NEVER,
		Less = D3D11_COMPARISON_LESS,
		Equal = D3D11_COMPARISON_EQUAL,
		LessEqual = D3D11_COMPARISON_LESS_EQUAL,
		Greate = D3D11_COMPARISON_GREATER,
		NotEqual = D3D11_COMPARISON_NOT_EQUAL,
		GreaterEqual = D3D11_COMPARISON_GREATER_EQUAL,
		Always = D3D11_COMPARISON_ALWAYS,
	};
	public enum struct D3D11ClearFlag
	{
		Depth = D3D11_CLEAR_DEPTH,
		Stencil = D3D11_CLEAR_STENCIL,
	};
	public enum struct D3D11CpuAccessFlag
	{
		Write = D3D11_CPU_ACCESS_WRITE,
		Read = D3D11_CPU_ACCESS_READ,
	};
	public enum struct D3D11CullMode
	{
		None = D3D11_CULL_NONE,
		Front = D3D11_CULL_FRONT,
		Back = D3D11_CULL_BACK,
	};
	public enum struct D3D11DsvDimension
	{
		Unknown = D3D11_DSV_DIMENSION_UNKNOWN,
		Texture1D = D3D11_DSV_DIMENSION_TEXTURE1D,
		Texture1DArray = D3D11_DSV_DIMENSION_TEXTURE1DARRAY,
		Texture2D = D3D11_DSV_DIMENSION_TEXTURE2D,
		Texture2DArray = D3D11_DSV_DIMENSION_TEXTURE2DARRAY,
		Texture2DMs = D3D11_DSV_DIMENSION_TEXTURE2DMS,
		Texture2DMsArray = D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY,
	};
	public enum struct D3D11FillMode
	{
		Wireframe = D3D11_FILL_WIREFRAME,
		Solid = D3D11_FILL_SOLID,
	};
	public enum struct D3D11Filter
	{
		MinMagMipPoint = D3D11_FILTER_MIN_MAG_MIP_POINT,
		MinMagPointMipLinear = D3D11_FILTER_MIN_MAG_POINT_MIP_LINEAR,
		MinPointMagLinearMipPoint = D3D11_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT,
		MinPointMagMipLinear = D3D11_FILTER_MIN_POINT_MAG_MIP_LINEAR,
		MinLinearMagMipPoint = D3D11_FILTER_MIN_LINEAR_MAG_MIP_POINT,
		MinLinearMagPointMipLinear = D3D11_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR,
		MinMagLinearMipPoint = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT,
		MinMagMipLinear = D3D11_FILTER_MIN_MAG_MIP_LINEAR,
		Anisotropic = D3D11_FILTER_ANISOTROPIC,
		ComparisonMinMagMipPoint = D3D11_FILTER_COMPARISON_MIN_MAG_MIP_POINT,
		ComparisonMinMagPointMipLinear = D3D11_FILTER_COMPARISON_MIN_MAG_POINT_MIP_LINEAR,
		ComparisonMinPointMagLinearMipPoint = D3D11_FILTER_COMPARISON_MIN_POINT_MAG_LINEAR_MIP_POINT,
		ComparisonMinPointMagMipLinear = D3D11_FILTER_COMPARISON_MIN_POINT_MAG_MIP_LINEAR,
		ComparisonMinLinearMagMipPoint = D3D11_FILTER_COMPARISON_MIN_LINEAR_MAG_MIP_POINT,
		ComparisonMinLinearMagPointMipLinear = D3D11_FILTER_COMPARISON_MIN_LINEAR_MAG_POINT_MIP_LINEAR,
		ComparisonMinMagLinearMipPoint = D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
		ComparisonMinMagMipLinear = D3D11_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR,
		ComparisonAnisotropic = D3D11_FILTER_COMPARISON_ANISOTROPIC,
		MinimumMinMagMipPoint = D3D11_FILTER_MINIMUM_MIN_MAG_MIP_POINT,
		MinimumMinMagPointMipLinear = D3D11_FILTER_MINIMUM_MIN_MAG_POINT_MIP_LINEAR,
		MinimumMinPointMagLinearMipPoint = D3D11_FILTER_MINIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT,
		MinimumMinPointMagMipLinear = D3D11_FILTER_MINIMUM_MIN_POINT_MAG_MIP_LINEAR,
		MinimumMinLinearMagMipPoint = D3D11_FILTER_MINIMUM_MIN_LINEAR_MAG_MIP_POINT,
		MinimumMinLinearMagPointMipLinear = D3D11_FILTER_MINIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR,
		MinimumMinMagLinearMipPoint = D3D11_FILTER_MINIMUM_MIN_MAG_LINEAR_MIP_POINT,
		MinimumMinMagMipLinear = D3D11_FILTER_MINIMUM_MIN_MAG_MIP_LINEAR,
		MinimumAnisotropic = D3D11_FILTER_MINIMUM_ANISOTROPIC,
		MaximumMinMagMipPoint = D3D11_FILTER_MAXIMUM_MIN_MAG_MIP_POINT,
		MaximumMinMagPointMipLinear = D3D11_FILTER_MAXIMUM_MIN_MAG_POINT_MIP_LINEAR,
		MaximumMinPointMagLinearMipPoint = D3D11_FILTER_MAXIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT,
		MaximumMinPointMagMipLinear = D3D11_FILTER_MAXIMUM_MIN_POINT_MAG_MIP_LINEAR,
		MaxiumuMinLinearMagMipPoint = D3D11_FILTER_MAXIMUM_MIN_LINEAR_MAG_MIP_POINT,
		MaximumMinLinearMagPointMipLinear = D3D11_FILTER_MAXIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR,
		MaximumMinMagLinearMipPoint = D3D11_FILTER_MAXIMUM_MIN_MAG_LINEAR_MIP_POINT,
		MaximumMinMagMipLinear = D3D11_FILTER_MAXIMUM_MIN_MAG_MIP_LINEAR,
		MaximumAnisotropic = D3D11_FILTER_MAXIMUM_ANISOTROPIC,
	};
	public enum struct D3D11InputClassification
	{
		PerVertexData = D3D11_INPUT_PER_VERTEX_DATA,
		PerInstanceData = D3D11_INPUT_PER_INSTANCE_DATA,
	};
	public enum struct D3D11Map
	{
		Read = D3D11_MAP_READ,
		Write = D3D11_MAP_WRITE,
		ReadWrite = D3D11_MAP_READ_WRITE,
		Discard = D3D11_MAP_WRITE_DISCARD,
		NoOverwrite = D3D11_MAP_WRITE_NO_OVERWRITE,
	};
	public enum struct D3D11MapFlag
	{
		DoNotWait = D3D11_MAP_FLAG_DO_NOT_WAIT,
	};
	public enum struct D3D11ResourceMiscFlag
	{
		GenerateMips = D3D11_RESOURCE_MISC_GENERATE_MIPS,
		Shared = D3D11_RESOURCE_MISC_SHARED,
		TextureCube = D3D11_RESOURCE_MISC_TEXTURECUBE,
		DrawIndirectArgs = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS,
		BufferAllowRawViews = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS,
		BufferStructured = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
		ResourceClamp = D3D11_RESOURCE_MISC_RESOURCE_CLAMP,
		SharedKeyedMuted = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX,
		GdiCompatible = D3D11_RESOURCE_MISC_GDI_COMPATIBLE,
		SharedNtHandle = D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
		RestrictedContent = D3D11_RESOURCE_MISC_RESTRICTED_CONTENT,
		RestrictSharedResource = D3D11_RESOURCE_MISC_RESTRICT_SHARED_RESOURCE,
		RestrictSharedResourceDriver = D3D11_RESOURCE_MISC_RESTRICT_SHARED_RESOURCE_DRIVER,
		Guarded = D3D11_RESOURCE_MISC_GUARDED,
		TilePool = D3D11_RESOURCE_MISC_TILE_POOL,
		Tiled = D3D11_RESOURCE_MISC_TILED,
		HwProtected = D3D11_RESOURCE_MISC_HW_PROTECTED,
	};
	public enum struct D3D11RtvDimension
	{
		Unknown = D3D11_RTV_DIMENSION_UNKNOWN,
		Buffer = D3D11_RTV_DIMENSION_BUFFER,
		Texture1D = D3D11_RTV_DIMENSION_TEXTURE1D,
		Texture1DArray = D3D11_RTV_DIMENSION_TEXTURE1DARRAY,
		Texture2D = D3D11_RTV_DIMENSION_TEXTURE2D,
		Texture2DArray = D3D11_RTV_DIMENSION_TEXTURE2DARRAY,
		Texture2DMs = D3D11_RTV_DIMENSION_TEXTURE2DMS,
		Texture2DMsArray = D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY,
		Texture3D = D3D11_RTV_DIMENSION_TEXTURE3D,
	};
	public enum struct D3D11TextureAddressMode
	{
		Wrap = D3D11_TEXTURE_ADDRESS_WRAP,
		Mirror = D3D11_TEXTURE_ADDRESS_MIRROR,
		Clamp = D3D11_TEXTURE_ADDRESS_CLAMP,
		Border = D3D11_TEXTURE_ADDRESS_BORDER,
		MirrorOnce = D3D11_TEXTURE_ADDRESS_MIRROR_ONCE,
	};
	public enum struct D3D11Usage
	{
		Default = D3D11_USAGE_DEFAULT,
		Immutable = D3D11_USAGE_IMMUTABLE,
		Dynamic = D3D11_USAGE_DYNAMIC,
		Staging = D3D11_USAGE_STAGING,
	};
	public value struct D3D11BufferDesc
	{
		UINT					ByteWidth;
		D3D11Usage				Usage;
		D3D11BindFlag			BindFlags;
		D3D11CpuAccessFlag		CPUAccessFlags;
		D3D11ResourceMiscFlag	MiscFlags;
		UINT					StructureByteStride;
	};	
	public value struct D3D11InputElementDesc
	{
		System::String^             SemanticName;
		UINT						SemanticIndex;
		DXGIFormat					Format;
		UINT						InputSlot;
		UINT						AlignedByteOffset;
		D3D11InputClassification	InputSlotClass;
		UINT						InstanceDataStepRate;
	};
	public value struct D3D11MappedSubresource
	{
		System::IntPtr pData;
		UINT RowPitch;
		UINT DepthPitch;
	};
	public value struct D3D11RasterizerDesc
	{
		D3D11FillMode FillMode;
		D3D11CullMode CullMode;
		BOOL FrontCounterClockwise;
		INT DepthBias;
		FLOAT DepthBiasClamp;
		FLOAT SlopeScaledDepthBias;
		BOOL DepthClipEnable;
		BOOL ScissorEnable;
		BOOL MultisampleEnable;
		BOOL AntialiasedLineEnable;
	};
	public value struct D3D11Rect
	{
		LONG left;
		LONG top;
		LONG right;
		LONG bottom;
	};
	public value struct D3D11SamplerDesc
	{
		D3D11Filter Filter;
		D3D11TextureAddressMode AddressU;
		D3D11TextureAddressMode AddressV;
		D3D11TextureAddressMode AddressW;
		FLOAT MipLODBias;
		UINT MaxAnisotropy;
		D3D11ComparisonFunc ComparisonFunc;
		FLOAT BorderColor0;
		FLOAT BorderColor1;
		FLOAT BorderColor2;
		FLOAT BorderColor3;
		FLOAT MinLOD;
		FLOAT MaxLOD;
	};
	public value struct D3D11Tex2DDsv
	{
		UINT MipSlice;
	};
	public value struct D3D11DepthStencilViewDesc
	{
		DXGIFormat Format;
		D3D11DsvDimension ViewDimension;
		UINT Flags;
		D3D11Tex2DDsv Texture2D;
	};
	public value struct D3D11Tex2DRtv
	{
		UINT MipSlice;
	};
	public value struct D3D11Tex2DSrv
	{
		UINT MostDetailedMip;
		UINT MipLevels;
	};
	public value struct D3D11RenderTargetViewDesc
	{
		DXGIFormat			Format;
		D3D11RtvDimension	ViewDimension;
		D3D11Tex2DRtv       Texture2D;
	};
	public value struct D3D11ShaderResourceViewDesc
	{
		DXGIFormat		Format;
		D3DSrvDimension	ViewDimension;
		D3D11Tex2DSrv	Texture2D;
	};
	public value struct D3D11SubresourceData
	{
		System::Array^	pSysMem;
		UINT			SysMemPitch;
		UINT			SysMemSlicePitch;
	};
	public value struct D3D11Texture2DDesc
	{
		UINT Width;
		UINT Height;
		UINT MipLevels;
		UINT ArraySize;
		DXGIFormat Format;
		DXGISampleDesc SampleDesc;
		D3D11Usage Usage;
		D3D11BindFlag BindFlags;
		D3D11CpuAccessFlag CPUAccessFlags;
		D3D11ResourceMiscFlag MiscFlags;
	};
	public value struct D3D11Viewport
	{
		FLOAT TopLeftX;
		FLOAT TopLeftY;
		FLOAT Width;
		FLOAT Height;
		FLOAT MinDepth;
		FLOAT MaxDepth;
	};
	public interface class D3D11Resource
	{
	public:
		virtual ID3D11Resource* GetResource() = 0;
	};
	public ref class D3D11Buffer : COMWrapper<ID3D11Buffer>
	{
	public:
		D3D11Buffer(ID3D11Buffer *pObj) : COMWrapper(pObj)
		{
		}
	};
	public ref class D3D11DepthStencilView : COMWrapper<ID3D11DepthStencilView>
	{
	public:
		D3D11DepthStencilView(ID3D11DepthStencilView *pObj) : COMWrapper(pObj)
		{
		}
	};
	public ref class D3D11InputLayout : COMWrapper<ID3D11InputLayout>
	{
	public:
		D3D11InputLayout(ID3D11InputLayout *pObj) : COMWrapper(pObj)
		{
		}
	};
	public ref class D3D11PixelShader : COMWrapper<ID3D11PixelShader>
	{
	public:
		D3D11PixelShader(ID3D11PixelShader *pObj) : COMWrapper(pObj)
		{
		}
	};
	public ref class D3D11RasterizerState2 : COMWrapper<ID3D11RasterizerState2>
	{
	public:
		D3D11RasterizerState2(ID3D11RasterizerState2 *pObj) : COMWrapper(pObj)
		{
		}
	};
	public ref class D3D11RenderTargetView : COMWrapper<ID3D11RenderTargetView>
	{
	public:
		D3D11RenderTargetView(ID3D11RenderTargetView *pObj) : COMWrapper(pObj)
		{
		}
	};
	public ref class D3D11SamplerState : COMWrapper<ID3D11SamplerState>
	{
	public:
		D3D11SamplerState(ID3D11SamplerState *pObj) : COMWrapper(pObj)
		{
		}
	};
	public ref class D3D11ShaderResourceView1 : COMWrapper<ID3D11ShaderResourceView1>
	{
	public:
		D3D11ShaderResourceView1(ID3D11ShaderResourceView1 *pObj) : COMWrapper(pObj)
		{
		}
	};
	public ref class D3D11Texture2D : COMWrapper<ID3D11Texture2D>, D3D11Resource
	{
	public:
		D3D11Texture2D(ID3D11Texture2D *pObj) : COMWrapper(pObj)
		{
		}
		virtual ID3D11Resource* GetResource()
		{
			return Wrapped;
		}
	};
	public ref class D3D11VertexShader : COMWrapper<ID3D11VertexShader>
	{
	public:
		D3D11VertexShader(ID3D11VertexShader *pObj) : COMWrapper(pObj)
		{
		}
	};
	public ref class D3D11DeviceContext4 : COMWrapper<ID3D11DeviceContext4>
	{
	public:
		D3D11DeviceContext4(ID3D11DeviceContext4 *pObj) : COMWrapper(pObj)
		{
		}
		void Begin()
		{
			pWrapped->Begin(nullptr);
		}
		void ClearDepthStencilView(D3D11DepthStencilView ^pDepthStencilView, D3D11ClearFlag ClearFlags, FLOAT Depth, UINT8 Stencil)
		{
			pWrapped->ClearDepthStencilView(pDepthStencilView->Wrapped, (D3D11_CLEAR_FLAG)ClearFlags, Depth, Stencil);
		}
		void ClearRenderTargetView(D3D11RenderTargetView ^pRenderTargetView, float R, float G, float B, float A)
		{
			float ColorRGBA[4] = { R,G,B,A };
			pWrapped->ClearRenderTargetView(pRenderTargetView == nullptr ? nullptr : pRenderTargetView->Wrapped, ColorRGBA);
		}
		void CopyResource(D3D11Resource ^pDstResource, D3D11Resource ^pSrcResource)
		{
			pWrapped->CopyResource(pDstResource->GetResource(), pSrcResource->GetResource());
		}
		void Draw(UINT VertexCount, UINT StartVertexLocation)
		{
			pWrapped->Draw(VertexCount, StartVertexLocation);
		}
		void End()
		{
			pWrapped->End(nullptr);
		}
		void Flush()
		{
			pWrapped->Flush();
		}
		void IASetInputLayout(D3D11InputLayout ^pInputLayout)
		{
			pWrapped->IASetInputLayout(pInputLayout == nullptr ? nullptr : pInputLayout->Wrapped);
		}
		void IASetPrimitiveTopology(D3DPrimitiveTopology Topology)
		{
			pWrapped->IASetPrimitiveTopology((D3D11_PRIMITIVE_TOPOLOGY)Topology);
		}
		void IASetVertexBuffers(UINT StartSlot, cli::array<D3D11Buffer^> ^ppVertexBuffers, cli::array<UINT> ^pStrides, cli::array<UINT> ^pOffsets)
		{
			std::unique_ptr<ID3D11Buffer*[]> ppVertexBuffersM(new ID3D11Buffer*[ppVertexBuffers->Length]);
			for (int i = 0; i < ppVertexBuffers->Length; ++i)
			{
				ppVertexBuffersM[i] = ppVertexBuffers[i]->Wrapped;
			}
			pin_ptr<UINT> pStridesM = &pStrides[0];
			pin_ptr<UINT> pOffsetsM = &pOffsets[0];
			pWrapped->IASetVertexBuffers(StartSlot, ppVertexBuffers->Length, &ppVertexBuffersM[0], &pStridesM[0], &pOffsetsM[0]);
		}
		D3D11MappedSubresource Map(D3D11Resource ^pResource, UINT Subresource, D3D11Map MapType, D3D11MapFlag MapFlags)
		{
			D3D11_MAPPED_SUBRESOURCE pMappedResourceM = { 0 };
			TRY_D3D(pWrapped->Map(pResource->GetResource(), Subresource, (D3D11_MAP)MapType, (UINT)MapFlags, &pMappedResourceM));
			D3D11MappedSubresource pMappedResource;
			pMappedResource.pData = System::IntPtr(pMappedResourceM.pData);
			pMappedResource.RowPitch = pMappedResourceM.RowPitch;
			pMappedResource.DepthPitch = pMappedResourceM.DepthPitch;
			return pMappedResource;
		};
		void OMSetRenderTargets(cli::array<D3D11RenderTargetView^> ^ppRenderTargetViews, D3D11DepthStencilView ^pDepthStencilView)
		{
			std::unique_ptr<ID3D11RenderTargetView*[]> ppRenderTargetViewsM(new ID3D11RenderTargetView*[ppRenderTargetViews->Length]);
			for (int i = 0; i < ppRenderTargetViews->Length; ++i)
			{
				ppRenderTargetViewsM[i] = ppRenderTargetViews[i] == nullptr ? nullptr : ppRenderTargetViews[i]->Wrapped;
			}
			pWrapped->OMSetRenderTargets(ppRenderTargetViews->Length, &ppRenderTargetViewsM[0], pDepthStencilView == nullptr ? nullptr : pDepthStencilView->Wrapped);
		}
		void PSSetSamplers(UINT StartSlot, cli::array<D3D11SamplerState^> ^ppSamplers)
		{
			std::unique_ptr<ID3D11SamplerState*[]> ppSamplersM(new ID3D11SamplerState*[ppSamplers->Length]);
			for (int i = 0; i < ppSamplers->Length; ++i)
			{
				ppSamplersM[i] = ppSamplers[i] == nullptr ? nullptr : ppSamplers[i]->Wrapped;
			}
			pWrapped->PSSetSamplers(StartSlot, ppSamplers->Length, &ppSamplersM[0]);
		}
		void PSSetShaderResources(UINT StartSlot, cli::array<D3D11ShaderResourceView1^> ^ppShaderResourceViews)
		{
			std::unique_ptr<ID3D11ShaderResourceView*[]> ppShaderResourceViewsM(new ID3D11ShaderResourceView*[ppShaderResourceViews->Length]);
			for (int i = 0; i < ppShaderResourceViews->Length; ++i)
			{
				ppShaderResourceViewsM[i] = ppShaderResourceViews[i] == nullptr ? nullptr : ppShaderResourceViews[i]->Wrapped;
			}
			pWrapped->PSSetShaderResources(StartSlot, ppShaderResourceViews->Length, &ppShaderResourceViewsM[0]);
		}
		void PSSetShader(D3D11PixelShader ^pPixelShader)
		{
			pWrapped->PSSetShader(pPixelShader == nullptr ? nullptr : pPixelShader->Wrapped, nullptr, 0);
		}
		void RSSetState(D3D11RasterizerState2 ^ppRasterizerState)
		{
			pWrapped->RSSetState(ppRasterizerState == nullptr ? nullptr : ppRasterizerState->Wrapped);
		}
		void RSSetScissorRects(cli::array<D3D11Rect> ^pRects)
		{
			pin_ptr<D3D11Rect> pRectsM = &pRects[0];
			pWrapped->RSSetScissorRects(pRects->Length, (D3D11_RECT*)&pRectsM[0]);
		}
		void RSSetViewports(cli::array<D3D11Viewport> ^pViewports)
		{
			pin_ptr<D3D11Viewport> pViewportsM = &pViewports[0];
			pWrapped->RSSetViewports(pViewports->Length, (D3D11_VIEWPORT*)&pViewportsM[0]);
		}
		void Unmap(D3D11Resource ^pResource, UINT Subresource)
		{
			pWrapped->Unmap(pResource == nullptr ? nullptr : pResource->GetResource(), Subresource);
		}
		void VSSetConstantBuffers(UINT StartSlot, cli::array<D3D11Buffer^> ^ppConstantBuffers)
		{
			std::unique_ptr<ID3D11Buffer*[]> ppConstantBuffersM(new ID3D11Buffer*[ppConstantBuffers->Length]);
			for (int i = 0; i < ppConstantBuffers->Length; ++i)
			{
				ppConstantBuffersM[i] = ppConstantBuffers[i]->Wrapped;
			}
			pWrapped->VSSetConstantBuffers(StartSlot, ppConstantBuffers->Length, &ppConstantBuffersM[0]);
		}
		void VSSetShader(D3D11VertexShader ^pVertexShader)
		{
			pWrapped->VSSetShader(pVertexShader == nullptr ? nullptr : pVertexShader->Wrapped, nullptr, 0);
		}
	};
	public ref class D3D11Device5 : COMWrapper<ID3D11Device5>
	{
	public:
		D3D11Device5(ID3D11Device5 *pObj) : COMWrapper(pObj)
		{
		}
		D3D11Buffer^ CreateBuffer(D3D11BufferDesc pDesc, System::Nullable<D3D11SubresourceData> pInitialData)
		{
			ID3D11Buffer *ppBuffer = nullptr;
			if (pInitialData.HasValue)
			{
				D3D11_SUBRESOURCE_DATA pInitialDataM = { 0 };
				auto gchandle = System::Runtime::InteropServices::GCHandle::Alloc(pInitialData.Value.pSysMem, System::Runtime::InteropServices::GCHandleType::Pinned);
				try
				{
					pInitialDataM.pSysMem = gchandle.AddrOfPinnedObject().ToPointer();
					pInitialDataM.SysMemPitch = pInitialData.Value.SysMemPitch;
					pInitialDataM.SysMemSlicePitch = pInitialData.Value.SysMemSlicePitch;
					TRY_D3D(pWrapped->CreateBuffer((D3D11_BUFFER_DESC*)&pDesc, &pInitialDataM, &ppBuffer));
				}
				finally
				{
					gchandle.Free();
				}
			}
			else
			{
				TRY_D3D(pWrapped->CreateBuffer((D3D11_BUFFER_DESC*)&pDesc, nullptr, &ppBuffer));
			}
			return gcnew D3D11Buffer(ppBuffer);
		}
		D3D11DepthStencilView^ CreateDepthStencilView(D3D11Resource ^pResource, D3D11DepthStencilViewDesc pDesc)
		{
			ID3D11DepthStencilView *ppDepthStencilView = nullptr;
			D3D11_DEPTH_STENCIL_VIEW_DESC pDescM;
			pDescM.Format = (DXGI_FORMAT)pDesc.Format;
			pDescM.ViewDimension = (D3D11_DSV_DIMENSION)pDesc.ViewDimension;
			pDescM.Texture2D.MipSlice = pDesc.Texture2D.MipSlice;
			TRY_D3D(pWrapped->CreateDepthStencilView(pResource->GetResource(), &pDescM, &ppDepthStencilView));
			return gcnew D3D11DepthStencilView(ppDepthStencilView);
		}
		D3D11InputLayout^ CreateInputLayout(cli::array<D3D11InputElementDesc> ^pInputElementDescs, cli::array<byte> ^pShaderBytecodeWithInputSignature)
		{
			ID3D11InputLayout *ppInputLayout = nullptr;
			msclr::interop::marshal_context ctx;
			std::unique_ptr<D3D11_INPUT_ELEMENT_DESC[]> pInputElementDescsM(new D3D11_INPUT_ELEMENT_DESC[pInputElementDescs->Length]);
			pin_ptr<byte> pShaderBytecodeWithInputSignatureM = &pShaderBytecodeWithInputSignature[0];
			for (int i = 0; i < pInputElementDescs->Length; ++i)
			{
				pInputElementDescsM[i].SemanticName = ctx.marshal_as<LPCSTR>(pInputElementDescs[i].SemanticName);
				pInputElementDescsM[i].SemanticIndex = pInputElementDescs[i].SemanticIndex;
				pInputElementDescsM[i].Format = (DXGI_FORMAT)pInputElementDescs[i].Format;
				pInputElementDescsM[i].InputSlot = pInputElementDescs[i].InputSlot;
				pInputElementDescsM[i].AlignedByteOffset = pInputElementDescs[i].AlignedByteOffset;
				pInputElementDescsM[i].InputSlotClass = (D3D11_INPUT_CLASSIFICATION)pInputElementDescs[i].InputSlotClass;
				pInputElementDescsM[i].InstanceDataStepRate = pInputElementDescs[i].InstanceDataStepRate;
			}
			TRY_D3D(pWrapped->CreateInputLayout(&pInputElementDescsM[0], pInputElementDescs->Length, pShaderBytecodeWithInputSignatureM, pShaderBytecodeWithInputSignature->Length, &ppInputLayout));
			return gcnew D3D11InputLayout(ppInputLayout);
		}
		D3D11PixelShader^ CreatePixelShader(cli::array<byte> ^pShaderBytecode)
		{
			ID3D11PixelShader *ppPixelShader = nullptr;
			pin_ptr<byte> pShaderBytecodeM = &pShaderBytecode[0];
			TRY_D3D(pWrapped->CreatePixelShader(pShaderBytecodeM, pShaderBytecode->Length, nullptr, &ppPixelShader));
			return gcnew D3D11PixelShader(ppPixelShader);
		}
		D3D11RasterizerState2^ CreateRasterizerState(D3D11RasterizerDesc pDesc)
		{
			ID3D11RasterizerState *ppRasterizerState = nullptr;
			TRY_D3D(pWrapped->CreateRasterizerState((D3D11_RASTERIZER_DESC*)&pDesc, &ppRasterizerState));
			ID3D11RasterizerState2 *ppRasterizerState2 = nullptr;
			TRY_D3D(ppRasterizerState->QueryInterface<ID3D11RasterizerState2>(&ppRasterizerState2));
			return gcnew D3D11RasterizerState2(ppRasterizerState2);
		}
		D3D11RenderTargetView^ CreateRenderTargetView(D3D11Resource ^pResource, D3D11RenderTargetViewDesc pDesc)
		{
			ID3D11RenderTargetView *ppRTView = nullptr;
			D3D11_RENDER_TARGET_VIEW_DESC pDescM;
			pDescM.Format = (DXGI_FORMAT)pDesc.Format;
			pDescM.ViewDimension = (D3D11_RTV_DIMENSION)pDesc.ViewDimension;
			pDescM.Texture2D.MipSlice = pDesc.Texture2D.MipSlice;
			TRY_D3D(pWrapped->CreateRenderTargetView(pResource->GetResource(), &pDescM, &ppRTView));
			return gcnew D3D11RenderTargetView(ppRTView);
		}
		D3D11SamplerState^ CreateSamplerState(D3D11SamplerDesc pSamplerDesc)
		{
			ID3D11SamplerState *ppSamplerState = nullptr;
			TRY_D3D(pWrapped->CreateSamplerState((D3D11_SAMPLER_DESC*)&pSamplerDesc, &ppSamplerState));
			return gcnew D3D11SamplerState(ppSamplerState);
		}
		D3D11ShaderResourceView1^ CreateShaderResourceView(D3D11Resource ^pResource, D3D11ShaderResourceViewDesc pDesc)
		{
			ID3D11ShaderResourceView *ppSRView = nullptr;
			D3D11_SHADER_RESOURCE_VIEW_DESC pDescM;
			pDescM.Format = (DXGI_FORMAT)pDesc.Format;
			pDescM.ViewDimension = (D3D11_SRV_DIMENSION)pDesc.ViewDimension;
			pDescM.Texture2D.MipLevels = pDesc.Texture2D.MipLevels;
			pDescM.Texture2D.MostDetailedMip = pDesc.Texture2D.MostDetailedMip;
			TRY_D3D(pWrapped->CreateShaderResourceView(pResource->GetResource(), &pDescM, &ppSRView));
			ID3D11ShaderResourceView1 *ppSRView1 = nullptr;
			TRY_D3D(ppSRView->QueryInterface<ID3D11ShaderResourceView1>(&ppSRView1));
			return gcnew D3D11ShaderResourceView1(ppSRView1);
		}
		D3D11Texture2D^ CreateTexture2D(D3D11Texture2DDesc pDesc, System::Nullable<D3D11SubresourceData> pInitialData)
		{
			ID3D11Texture2D *ppTexture2D = nullptr;
			if (pInitialData.HasValue)
			{
				D3D11_SUBRESOURCE_DATA pInitialDataM = { 0 };
				auto gchandle = System::Runtime::InteropServices::GCHandle::Alloc(pInitialData.Value.pSysMem, System::Runtime::InteropServices::GCHandleType::Pinned);
				try
				{
					pInitialDataM.pSysMem = gchandle.AddrOfPinnedObject().ToPointer();
					pInitialDataM.SysMemPitch = pInitialData.Value.SysMemPitch;
					pInitialDataM.SysMemSlicePitch = pInitialData.Value.SysMemSlicePitch;
					TRY_D3D(pWrapped->CreateTexture2D((D3D11_TEXTURE2D_DESC*)&pDesc, &pInitialDataM, &ppTexture2D));
				}
				finally
				{
					gchandle.Free();
				}
			}
			else
			{
				TRY_D3D(pWrapped->CreateTexture2D((D3D11_TEXTURE2D_DESC*)&pDesc, nullptr, &ppTexture2D));
			}
			return gcnew D3D11Texture2D(ppTexture2D);
		}
		D3D11VertexShader^ CreateVertexShader(cli::array<byte> ^pShaderBytecode)
		{
			ID3D11VertexShader *ppVertexShader = nullptr;
			pin_ptr<byte> pShaderBytecodeM = &pShaderBytecode[0];
			TRY_D3D(pWrapped->CreateVertexShader(pShaderBytecodeM, pShaderBytecode->Length, nullptr, &ppVertexShader));
			return gcnew D3D11VertexShader(ppVertexShader);
		}
		D3D11DeviceContext4^ GetImmediateContext()
		{
			ID3D11DeviceContext *ppImmediateContext = nullptr;
			pWrapped->GetImmediateContext(&ppImmediateContext);
			ID3D11DeviceContext4 *ppImmediateContext4 = nullptr;
			TRY_D3D(ppImmediateContext->QueryInterface<ID3D11DeviceContext4>(&ppImmediateContext4));
			ppImmediateContext->Release();
			return gcnew D3D11DeviceContext4(ppImmediateContext4);
		}
	};
	public ref class Direct3D11
	{
	public:
		static D3D11Device5^ D3D11CreateDevice()
		{
			ID3D11Device *ppDevice = nullptr;
			TRY_D3D(::D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_DEBUG, nullptr, 0, D3D11_SDK_VERSION, &ppDevice, nullptr, nullptr));
			ID3D11Device5 *ppDevice5 = nullptr;
			TRY_D3D(ppDevice->QueryInterface<ID3D11Device5>(&ppDevice5));
			ppDevice->Release();
			return gcnew D3D11Device5(ppDevice5);
		}
	};
}