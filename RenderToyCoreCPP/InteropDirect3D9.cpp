////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

#include <d3d9.h>
#include <msclr\marshal_cppstd.h>
#include "InteropCommon.h"

System::String^ DXErrorString(HRESULT error)
{
	switch (error)
	{
	case D3D_OK: return "D3D_OK";
	case D3DERR_DEVICELOST: return "D3DERR_DEVICELOST";
	case D3DERR_DEVICEREMOVED: return "D3DERR_DEVICEREMOVED";
	case D3DERR_DRIVERINTERNALERROR: return "D3DERR_DRIVERINTERNALERROR";
	case D3DERR_OUTOFVIDEOMEMORY: return "D3DERR_OUTOFVIDEOMEMORY";
	case D3DERR_INVALIDCALL: return "D3DERR_INVALIDCALL";
	default: return "0x" + error.ToString("X8");
	}
}

#define TRY_D3D(FUNCTION) { HRESULT result = FUNCTION; if (result != D3D_OK) throw gcnew System::Exception("Direct3D 9 Error: " + DXErrorString(result) + "\n" + #FUNCTION); }

namespace RenderToy
{
	#pragma region - Direct3D9 Enumerations -
	public enum class D3DClear
	{
		Target = D3DCLEAR_TARGET,
		ZBuffer = D3DCLEAR_ZBUFFER,
		Stencil = D3DCLEAR_STENCIL,
	};
	public enum class D3DCullMode
	{
		None = D3DCULL_NONE,
		CW = D3DCULL_CW,
		CCW = D3DCULL_CCW,
	};
	public enum class D3DDeclMethod : byte {
		Default = D3DDECLMETHOD_DEFAULT,
		PartialU = D3DDECLMETHOD_PARTIALU,
		PartialV = D3DDECLMETHOD_PARTIALV,
		CrossUV = D3DDECLMETHOD_CROSSUV,
		UV = D3DDECLMETHOD_UV,
		Lookup = D3DDECLMETHOD_LOOKUP,
		LookupPresampled = D3DDECLMETHOD_LOOKUPPRESAMPLED,
	};
	public enum class D3DDeclType : byte {
		Float1 = D3DDECLTYPE_FLOAT1,
		Float2 = D3DDECLTYPE_FLOAT2,
		Float3 = D3DDECLTYPE_FLOAT3,
		Float4 = D3DDECLTYPE_FLOAT4,
		D3DColor = D3DDECLTYPE_D3DCOLOR,
		UByte4 = D3DDECLTYPE_UBYTE4,
		Short2 = D3DDECLTYPE_SHORT2,
		Short4 = D3DDECLTYPE_SHORT4,
		UByte4N = D3DDECLTYPE_UBYTE4N,
		Short2N = D3DDECLTYPE_SHORT2N,
		Short4N = D3DDECLTYPE_SHORT4N,
		UShort2N = D3DDECLTYPE_USHORT2N,
		UShort4N = D3DDECLTYPE_USHORT4N,
		UDec3 = D3DDECLTYPE_UDEC3,
		Dec3N = D3DDECLTYPE_DEC3N,
		Float16_2 = D3DDECLTYPE_FLOAT16_2,
		Float16_4 = D3DDECLTYPE_FLOAT16_4,
		Unused = D3DDECLTYPE_UNUSED,
	};
	public enum class D3DDeclUsage : byte {
		Position = D3DDECLUSAGE_POSITION,
		BlendWeight = D3DDECLUSAGE_BLENDWEIGHT,
		BlendIndices = D3DDECLUSAGE_BLENDINDICES,
		Normal = D3DDECLUSAGE_NORMAL,
		PSize = D3DDECLUSAGE_PSIZE,
		TexCoord = D3DDECLUSAGE_TEXCOORD,
		Tangent = D3DDECLUSAGE_TANGENT,
		Binormal = D3DDECLUSAGE_BINORMAL,
		TessFactor = D3DDECLUSAGE_TESSFACTOR,
		PositionT = D3DDECLUSAGE_POSITIONT,
		Color = D3DDECLUSAGE_COLOR,
		Fog = D3DDECLUSAGE_FOG,
		Depth = D3DDECLUSAGE_DEPTH,
		Sample = D3DDECLUSAGE_SAMPLE,
	};
	public enum class D3DFormat
	{
		A8R8G8B8 = D3DFMT_A8R8G8B8,
		D24X8 = D3DFMT_D24X8,
	};
	public enum class D3DFvf
	{
		XYZ = D3DFVF_XYZ,
		Normal = D3DFVF_NORMAL,
		Diffuse = D3DFVF_DIFFUSE,
		Tex1 = D3DFVF_TEX1,
		XYZW = D3DFVF_XYZW,
	};
	public enum class D3DMultisample
	{
		None = D3DMULTISAMPLE_NONE,
	};
	public enum class D3DPool
	{
		Default = D3DPOOL_DEFAULT,
		Managed = D3DPOOL_MANAGED,
		SystemMemory = D3DPOOL_SYSTEMMEM,
		Scratch = D3DPOOL_SCRATCH,
	};
	public enum class D3DPrimitiveType
	{
		TriangleList = D3DPT_TRIANGLELIST,
	};
	public enum class D3DRenderState
	{
		ZEnable = D3DRS_ZENABLE,
		CullMode = D3DRS_CULLMODE,
		Lighting = D3DRS_LIGHTING,
	};
	public enum class D3DSamplerState
	{
		MagFilter = D3DSAMP_MAGFILTER,
		MinFilter = D3DSAMP_MINFILTER,
		MipFilter = D3DSAMP_MIPFILTER,
		MaxAnisotropy = D3DSAMP_MAXANISOTROPY,
	};
	public enum class D3DTextureFilter
	{
		None = D3DTEXF_NONE,
		Point = D3DTEXF_POINT,
		Linear = D3DTEXF_LINEAR,
		Anisotropic = D3DTEXF_ANISOTROPIC,
	};
	public enum class D3DTransformState
	{
		View = D3DTS_VIEW,
		Projection = D3DTS_PROJECTION,
	};
	public enum class D3DUsage
	{
		WriteOnly = D3DUSAGE_WRITEONLY,
	};
	#pragma endregion
	#pragma region - Direct3D9 Structures -
	public ref class D3DLockedRect
	{
	public:
		INT Pitch;
		System::IntPtr Bits;
	};
	public value struct D3DVertexElement9
	{
		WORD			Stream;     // Stream index
		WORD			Offset;     // Offset in the stream in bytes
		D3DDeclType		Type;       // Data type
		D3DDeclMethod	Method;     // Processing method
		D3DDeclUsage	Usage;      // Semantics
		BYTE			UsageIndex; // Semantic index
	};
	#pragma endregion
	#pragma region - Direct3D9 Global Services -
	ref class Direct3D9Globals
	{
	private:
		Direct3D9Globals()
		{
			hHostWindow = CreateWindow("STATIC", "D3D9HostWindow", WS_OVERLAPPEDWINDOW, 0, 0, 16, 16, nullptr, nullptr, nullptr, nullptr);
			if (hHostWindow == nullptr) {
				throw gcnew System::Exception("CreateWindow() failed.");
			}
		}
		!Direct3D9Globals()
		{
			Destroy();
		}
		~Direct3D9Globals()
		{
			Destroy();
		}
		void Destroy()
		{
			if (hHostWindow != nullptr) {
				DestroyWindow(hHostWindow);
				hHostWindow = nullptr;
			}
		}
	public:
		HWND hHostWindow = nullptr;
		static Direct3D9Globals^ Instance = gcnew Direct3D9Globals();
	};
	#pragma endregion
	#pragma region - Direct3DPixelShader9 -
	public ref class Direct3DPixelShader9 : public COMWrapper<IDirect3DPixelShader9>
	{
	public:
		Direct3DPixelShader9(IDirect3DPixelShader9 *pObject) : COMWrapper(pObject)
		{
		}
	};
	#pragma endregion
	#pragma region - Direct3DSurface9 -
	public ref class Direct3DSurface9 : public COMWrapper<IDirect3DSurface9>
	{
	public:
		Direct3DSurface9(IDirect3DSurface9* pObject) : COMWrapper(pObject)
		{
		}
		D3DLockedRect^ LockRect()
		{
			D3DLOCKED_RECT LockedRect = { 0 };
			TRY_D3D(pWrapped->LockRect(&LockedRect, nullptr, D3DLOCK_READONLY));
			auto result = gcnew D3DLockedRect();
			result->Pitch = LockedRect.Pitch;
			result->Bits = System::IntPtr(LockedRect.pBits);
			return result;
		}
		void UnlockRect()
		{
			TRY_D3D(pWrapped->UnlockRect());
		}
	};
	#pragma endregion
	#pragma region - Direct3DTexture9 -
	public ref class Direct3DTexture9 : public COMWrapper<IDirect3DTexture9>
	{
	public:
		Direct3DTexture9(IDirect3DTexture9 *pObject)
		{
			pWrapped = pObject;
		}
		D3DLockedRect^ LockRect(unsigned int Level)
		{
			D3DLOCKED_RECT LockedRect = { 0 };
			TRY_D3D(pWrapped->LockRect(Level, &LockedRect, nullptr, D3DLOCK_DISCARD));
			auto result = gcnew D3DLockedRect();
			result->Pitch = LockedRect.Pitch;
			result->Bits = System::IntPtr(LockedRect.pBits);
			return result;
		}
		void UnlockRect(unsigned int Level)
		{
			TRY_D3D(pWrapped->UnlockRect(Level));
		}
	};
	#pragma endregion
	#pragma region - Direct3DVertexBuffer9 -
	public ref class Direct3DVertexBuffer9 : public COMWrapper<IDirect3DVertexBuffer9>
	{
	public:
		Direct3DVertexBuffer9(IDirect3DVertexBuffer9 *pObject)
		{
			pWrapped = pObject;
		}
		System::IntPtr Lock(UINT OffsetToLock, UINT SizeToLock, DWORD Flags)
		{
			void* ppbData = nullptr;
			TRY_D3D(pWrapped->Lock(OffsetToLock, SizeToLock, &ppbData, Flags));
			return System::IntPtr(ppbData);
		}
		void Unlock()
		{
			TRY_D3D(pWrapped->Unlock());
		}
	};
	#pragma endregion
	#pragma region - Direct3DVertexDeclaration9 -
	public ref class Direct3DVertexDeclaration9 : COMWrapper<IDirect3DVertexDeclaration9>
	{
	public:
		Direct3DVertexDeclaration9(IDirect3DVertexDeclaration9 *pWrapped) : COMWrapper(pWrapped)
		{
		}
	};
	#pragma endregion
	#pragma region - Direct3DVertexShader9 -
	public ref class Direct3DVertexShader9 : COMWrapper<IDirect3DVertexShader9>
	{
	public:
		Direct3DVertexShader9(IDirect3DVertexShader9 *pWrapped) : COMWrapper(pWrapped)
		{
		}
	};
	#pragma endregion
	#pragma region - Direct3DDevice9Ex -
	public ref class Direct3DDevice9Ex : public COMWrapper<IDirect3DDevice9Ex>
	{
	public:
		Direct3DDevice9Ex(IDirect3DDevice9Ex *pObject)
		{
			pWrapped = pObject;
		}
		Direct3DTexture9^ CreateTexture(UINT Width, UINT Height, UINT Levels, DWORD Usage, D3DFormat Format, D3DPool Pool)
		{
			IDirect3DTexture9 *ppTexture = nullptr;
			TRY_D3D(pWrapped->CreateTexture(Width, Height, Levels, Usage, (D3DFORMAT)Format, (D3DPOOL)Pool, &ppTexture, nullptr));
			return gcnew Direct3DTexture9(ppTexture);
		}
		Direct3DVertexBuffer9^ CreateVertexBuffer(UINT Length, DWORD Usage, DWORD FVF, D3DPool Pool)
		{
			IDirect3DVertexBuffer9 *ppVertexBuffer = nullptr;
			TRY_D3D(pWrapped->CreateVertexBuffer(Length, Usage, FVF, (D3DPOOL)Pool, &ppVertexBuffer, nullptr));
			return gcnew Direct3DVertexBuffer9(ppVertexBuffer);
		}
		Direct3DSurface9^ CreateRenderTarget(UINT Width, UINT Height, D3DFormat Format, D3DMultisample MultiSample, DWORD MultisampleQuality, BOOL Lockable)
		{
			IDirect3DSurface9 *ppSurface = nullptr;
			TRY_D3D(pWrapped->CreateRenderTarget(Width, Height, (D3DFORMAT)Format, (D3DMULTISAMPLE_TYPE)MultiSample, MultisampleQuality, Lockable, &ppSurface, nullptr));
			return gcnew Direct3DSurface9(ppSurface);
		}
		Direct3DSurface9^ CreateDepthStencilSurface(UINT Width, UINT Height, D3DFormat Format, D3DMultisample MultiSample, DWORD MultisampleQuality, BOOL Discard)
		{
			IDirect3DSurface9 *ppSurface = nullptr;
			TRY_D3D(pWrapped->CreateDepthStencilSurface(Width, Height, (D3DFORMAT)Format, (D3DMULTISAMPLE_TYPE)MultiSample, MultisampleQuality, Discard, &ppSurface, nullptr));
			return gcnew Direct3DSurface9(ppSurface);
		}
		void UpdateTexture(Direct3DTexture9 ^pSourceTexture, Direct3DTexture9 ^pDestinationTexture)
		{
			TRY_D3D(pWrapped->UpdateTexture(pSourceTexture == nullptr ? nullptr : pSourceTexture->Wrapped, pDestinationTexture == nullptr ? nullptr : pDestinationTexture->Wrapped))
		}
		void SetRenderTarget(DWORD RenderTargetIndex, Direct3DSurface9^ pRenderTarget)
		{
			TRY_D3D(pWrapped->SetRenderTarget(RenderTargetIndex, pRenderTarget->Wrapped));
		}
		void SetDepthStencilSurface(Direct3DSurface9^ pNewZStencil)
		{
			TRY_D3D(pWrapped->SetDepthStencilSurface(pNewZStencil->Wrapped));
		}
		void BeginScene()
		{
			TRY_D3D(pWrapped->BeginScene());
		}
		void EndScene()
		{
			TRY_D3D(pWrapped->EndScene());
		}
		void SetTexture(DWORD Stage, Direct3DTexture9 ^pTexture)
		{
			TRY_D3D(pWrapped->SetTexture(Stage, pTexture == nullptr ? nullptr : pTexture->Wrapped));
		}
		void SetSamplerState(DWORD Sampler, D3DSamplerState Type, DWORD Value)
		{
			TRY_D3D(pWrapped->SetSamplerState(Sampler, (D3DSAMPLERSTATETYPE)Type, Value));
		}
		void Clear(D3DClear Flags, D3DCOLOR Color, float Z, DWORD Stencil)
		{
			TRY_D3D(pWrapped->Clear(0, nullptr, (DWORD)Flags, Color, Z, Stencil));
		}
		void SetTransform(D3DTransformState State, System::IntPtr ^pMatrix)
		{
			TRY_D3D(pWrapped->SetTransform((D3DTRANSFORMSTATETYPE)State, (const D3DMATRIX*)pMatrix->ToPointer()));
		}
		void SetRenderState(D3DRenderState State, DWORD Value)
		{
			TRY_D3D(pWrapped->SetRenderState((D3DRENDERSTATETYPE)State, Value));
		}
		Direct3DVertexDeclaration9^ CreateVertexDeclaration(array<D3DVertexElement9> ^pVertexElements)
		{
			std::unique_ptr<D3DVERTEXELEMENT9[]> declout(new D3DVERTEXELEMENT9[pVertexElements->Length + 1]);
			pin_ptr<D3DVertexElement9> declin(&pVertexElements[0]);
			memcpy(&declout[0], declin, sizeof(D3DVERTEXELEMENT9) * pVertexElements->Length);
			declout[pVertexElements->Length] = D3DDECL_END();
			IDirect3DVertexDeclaration9 *pTemp = nullptr;
			TRY_D3D(pWrapped->CreateVertexDeclaration(&declout[0], &pTemp));
			return gcnew Direct3DVertexDeclaration9(pTemp);
		}
		void SetVertexDeclaration(Direct3DVertexDeclaration9 ^pDecl)
		{
			TRY_D3D(pWrapped->SetVertexDeclaration(pDecl->Wrapped));
		}
		void SetFVF(D3DFvf FVF)
		{
			TRY_D3D(pWrapped->SetFVF((DWORD)FVF));
		}
		Direct3DVertexShader9^ CreateVertexShader(array<byte> ^pFunction)
		{
			pin_ptr<byte> marshal_pFunction(&pFunction[0]);
			IDirect3DVertexShader9 *pTemp = nullptr;
			TRY_D3D(pWrapped->CreateVertexShader(reinterpret_cast<const DWORD*>(marshal_pFunction), &pTemp));
			return gcnew Direct3DVertexShader9(pTemp);
		}
		void SetVertexShader(Direct3DVertexShader9 ^pShader)
		{
			TRY_D3D(pWrapped->SetVertexShader(pShader == nullptr ? nullptr : pShader->Wrapped));
		}
		void SetVertexShaderConstantF(UINT StartRegister, System::IntPtr pConstantData, UINT Vector4fCount)
		{
			TRY_D3D(pWrapped->SetVertexShaderConstantF(StartRegister, (const float*)pConstantData.ToPointer(), Vector4fCount));
		}
		void SetStreamSource(UINT StreamNumber, Direct3DVertexBuffer9 ^pStreamData, UINT OffsetInBytes, UINT Stride)
		{
			TRY_D3D(pWrapped->SetStreamSource(StreamNumber, pStreamData == nullptr ? nullptr : pStreamData->Wrapped, OffsetInBytes, Stride));
		}
		Direct3DPixelShader9^ CreatePixelShader(array<byte> ^pFunction)
		{
			pin_ptr<byte> marshal_pFunction(&pFunction[0]);
			IDirect3DPixelShader9 *pTemp = nullptr;
			TRY_D3D(pWrapped->CreatePixelShader(reinterpret_cast<const DWORD*>(marshal_pFunction), &pTemp));
			return gcnew Direct3DPixelShader9(pTemp);
		}
		void SetPixelShader(Direct3DPixelShader9 ^pShader)
		{
			TRY_D3D(pWrapped->SetPixelShader(pShader == nullptr ? nullptr : pShader->Wrapped));
		}
		void DrawPrimitive(D3DPrimitiveType PrimitiveType, UINT StartVertex, UINT PrimitiveCount)
		{
			TRY_D3D(pWrapped->DrawPrimitive((D3DPRIMITIVETYPE)PrimitiveType, StartVertex, PrimitiveCount));
		}
		void DrawPrimitiveUP(D3DPrimitiveType PrimitiveType, UINT PrimitiveCount, System::IntPtr ^pVertexStreamZeroData, UINT VertexStreamZeroStride)
		{
			TRY_D3D(pWrapped->DrawPrimitiveUP((D3DPRIMITIVETYPE)PrimitiveType, PrimitiveCount, pVertexStreamZeroData->ToPointer(), VertexStreamZeroStride));
		}
	};
	#pragma endregion
	#pragma region - Direct3D9Ex -
	public ref class Direct3D9Ex : public COMWrapper<IDirect3D9Ex>
	{
	public:
		Direct3D9Ex()
		{
			IDirect3D9Ex *pD3D9Ex = nullptr;
			TRY_D3D(Direct3DCreate9Ex(D3D_SDK_VERSION, &pD3D9Ex));
			pWrapped = pD3D9Ex;
		}
		Direct3DDevice9Ex^ CreateDevice()
		{
			D3DPRESENT_PARAMETERS d3dpp = { 0 };
			d3dpp.BackBufferWidth = 1;
			d3dpp.BackBufferHeight = 1;
			d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
			d3dpp.Windowed = TRUE;
			IDirect3DDevice9Ex* pReturnedDeviceInterface = nullptr;
			TRY_D3D(pWrapped->CreateDeviceEx(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, Direct3D9Globals::Instance->hHostWindow, D3DCREATE_FPU_PRESERVE | D3DCREATE_MULTITHREADED | D3DCREATE_HARDWARE_VERTEXPROCESSING, &d3dpp, nullptr, &pReturnedDeviceInterface));
			return gcnew Direct3DDevice9Ex(pReturnedDeviceInterface);
		}
	};
	#pragma endregion
}