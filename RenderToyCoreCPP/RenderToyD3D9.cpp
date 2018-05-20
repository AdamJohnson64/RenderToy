////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// A simple Fixed Function Direct3D 9 interface presented to the Common Language
// Runtime for consumption by .NET languages.
//
// This interface is highly-simplistic and does not attempt to model persistent
// GPU resources or framebuffers. Loss of device is trivially handled by never
// attempting to reuse devices beyond a single frame.
////////////////////////////////////////////////////////////////////////////////

#include <d3d9.h>
#include <d3dcompiler.h>
#include "msclr\marshal_cppstd.h"

#define TRY_D3D(D3D9FUNC) if ((D3D9FUNC) != D3D_OK) throw gcnew System::Exception(#D3D9FUNC)

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
	template <typename T>
	public ref class Direct3DWrap
	{
	public:
		Direct3DWrap()
		{
			this->pWrapped = nullptr;
		}
		Direct3DWrap(T* pWrapped)
		{
			this->pWrapped = pWrapped;
		}
		!Direct3DWrap()
		{
			Destroy();
		}
		~Direct3DWrap()
		{
			Destroy();
		}
		void Destroy()
		{
			if (pWrapped != nullptr)
			{
				pWrapped->Release();
				pWrapped = nullptr;
			}
		}
		property T* Wrapped
		{
			T* get()
			{
				return pWrapped;
			}
		}
		property System::IntPtr ManagedPtr
		{
			System::IntPtr get()
			{
				return System::IntPtr(pWrapped);
			}
		}
	protected:
		T* pWrapped;
	};
	#pragma endregion
	#pragma region - Direct3DPixelShader9 -
	public ref class Direct3DPixelShader9 : public Direct3DWrap<IDirect3DPixelShader9>
	{
	public:
		Direct3DPixelShader9(IDirect3DPixelShader9 *pObject) : Direct3DWrap(pObject)
		{
		}
	};
	#pragma endregion
	#pragma region - Direct3DSurface9 -
	public ref class Direct3DSurface9 : public Direct3DWrap<IDirect3DSurface9>
	{
	public:
		Direct3DSurface9(IDirect3DSurface9* pObject) : Direct3DWrap(pObject)
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
	public ref class Direct3DTexture9 : public Direct3DWrap<IDirect3DTexture9>
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
	public ref class Direct3DVertexBuffer9 : public Direct3DWrap<IDirect3DVertexBuffer9>
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
	#pragma region - Direct3DVertexShader9 -
	public ref class Direct3DVertexShader9 : Direct3DWrap<IDirect3DVertexShader9>
	{
	public:
		Direct3DVertexShader9(IDirect3DVertexShader9 *pWrapped) : Direct3DWrap(pWrapped)
		{
		}
	};
	#pragma endregion
	#pragma region - Direct3DDevice9 -
	public ref class Direct3DDevice9 : public Direct3DWrap<IDirect3DDevice9>
	{
	public:
		Direct3DDevice9(IDirect3DDevice9 *pObject)
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
			TRY_D3D(pWrapped->SetTexture(Stage, pTexture->Wrapped));
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
			TRY_D3D(pWrapped->SetVertexShader(pShader->Wrapped));
		}
		void SetVertexShaderConstantF(UINT StartRegister, System::IntPtr pConstantData, UINT Vector4fCount)
		{
			TRY_D3D(pWrapped->SetVertexShaderConstantF(StartRegister, (const float*)pConstantData.ToPointer(), Vector4fCount));
		}
		void SetStreamSource(UINT StreamNumber, Direct3DVertexBuffer9 ^pStreamData, UINT OffsetInBytes, UINT Stride)
		{
			TRY_D3D(pWrapped->SetStreamSource(StreamNumber, pStreamData->Wrapped, OffsetInBytes, Stride));
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
			TRY_D3D(pWrapped->SetPixelShader(pShader->Wrapped));
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
	#pragma region - Direct3D9 -
	public ref class Direct3D9 : public Direct3DWrap<IDirect3D9>
	{
	public:
		Direct3D9()
		{
			pWrapped = Direct3DCreate9(D3D_SDK_VERSION);
			if (pWrapped == nullptr) {
				throw gcnew System::Exception("Direct3DCreate9() failed.");
			}
		}
		Direct3DDevice9^ CreateDevice()
		{
			D3DPRESENT_PARAMETERS d3dpp = { 0 };
			d3dpp.BackBufferWidth = 1;
			d3dpp.BackBufferHeight = 1;
			d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
			d3dpp.Windowed = TRUE;
			IDirect3DDevice9* pReturnedDeviceInterface = nullptr;
			TRY_D3D(pWrapped->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, Direct3D9Globals::Instance->hHostWindow, D3DCREATE_FPU_PRESERVE | D3DCREATE_MULTITHREADED | D3DCREATE_HARDWARE_VERTEXPROCESSING, &d3dpp, &pReturnedDeviceInterface));
			return gcnew Direct3DDevice9(pReturnedDeviceInterface);
		}
	};
	#pragma endregion
	#pragma region - Direct3D Compiler -
	public ref class D3DBlob : Direct3DWrap<ID3DBlob>
	{
	public:
		D3DBlob() : Direct3DWrap(nullptr)
		{
		}
		D3DBlob(ID3DBlob *pWrapped) : Direct3DWrap(pWrapped)
		{
		}
		System::IntPtr GetBufferPointer()
		{
			if (pWrapped == nullptr) return System::IntPtr(nullptr);
			return System::IntPtr(pWrapped->GetBufferPointer());
		}
		size_t GetBufferSize()
		{
			if (pWrapped == nullptr) return 0;
			return pWrapped->GetBufferSize();
		}
	internal:
		void SetWrappedPointer(ID3DBlob *pNewWrapped)
		{
			if (pWrapped != nullptr)
			{
				pWrapped->Release();
				pWrapped = nullptr;
			}
			pWrapped = pNewWrapped;
		}
	};
	public ref class Direct3DCompiler
	{
	public:
		//static void D3DCompile(LPCVOID pSrcData, SIZE_T SrcDataSize, LPCSTR pSourceName, const D3D_SHADER_MACRO *pDefines, ID3DInclude *pInclude, LPCSTR pEntrypoint, LPCSTR pTarget, UINT Flags1, UINT Flags2, ID3DBlob **ppCode, ID3DBlob **ppErrorMsgs)
		static void D3DCompile(System::String ^pSrcData, System::String ^pSourceName, System::String ^pEntrypoint, System::String ^pTarget, UINT Flags1, UINT Flags2, D3DBlob ^ppCode, D3DBlob ^ppErrorMsgs)
		{
			msclr::interop::marshal_context marshalling;
			auto marshal_pSrcData = marshalling.marshal_as<const char*>(pSrcData);
			auto marshal_pSourceName = marshalling.marshal_as<const char*>(pSourceName);
			auto marshal_pEntrypoint = marshalling.marshal_as<const char*>(pEntrypoint);
			auto marshal_pTarget = marshalling.marshal_as<const char*>(pTarget);
			auto marshal_pCode = (ID3DBlob*)nullptr;
			auto marshal_pErrorMsgs = (ID3DBlob*)nullptr;
			auto marshal_ppCode = ppCode == nullptr ? nullptr : &marshal_pCode;
			auto marshal_ppErrorMsgs = ppErrorMsgs == nullptr ? nullptr : &marshal_pErrorMsgs;
			::D3DCompile(marshal_pSrcData, pSrcData->Length, marshal_pSourceName, nullptr, nullptr, marshal_pEntrypoint, marshal_pTarget, Flags1, Flags2, marshal_ppCode, marshal_ppErrorMsgs);
			if (marshal_ppCode != nullptr) ppCode->SetWrappedPointer(marshal_pCode);
			if (marshal_ppErrorMsgs != nullptr) ppErrorMsgs->SetWrappedPointer(marshal_pErrorMsgs);
			if (marshal_pCode == nullptr) return;
		}
	};
	#pragma endregion
}