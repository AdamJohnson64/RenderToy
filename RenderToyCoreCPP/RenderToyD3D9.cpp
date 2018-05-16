////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
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

#define TRY_D3D(D3D9FUNC) if ((D3D9FUNC) != D3D_OK) throw gcnew System::Exception(#D3D9FUNC)

namespace RenderToy
{
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
	public enum class D3DTransformState
	{
		View = D3DTS_VIEW,
		Projection = D3DTS_PROJECTION,
	};
	public enum class D3DUsage
	{
		WriteOnly = D3DUSAGE_WRITEONLY,
	};
	public ref class D3DLockedRect
	{
	public:
		INT Pitch;
		void* Bits;
	};
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
	protected:
		T* pWrapped;
	};
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
			result->Bits = LockedRect.pBits;
			return result;
		}
		void UnlockRect()
		{
			TRY_D3D(pWrapped->UnlockRect());
		}
	};
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
			result->Bits = LockedRect.pBits;
			return result;
		}
		void UnlockRect(unsigned int Level)
		{
			TRY_D3D(pWrapped->UnlockRect(Level));
		}
	};
	public ref class Direct3DDevice9 : public Direct3DWrap<IDirect3DDevice9>
	{
	public:
		Direct3DDevice9(IDirect3DDevice9 *pObject)
		{
			pWrapped = pObject;
		}
		Direct3DTexture9^ CreateTexture(unsigned int Width, unsigned int Height, unsigned int Levels, DWORD Usage, D3DFormat Format, D3DPool Pool)
		{
			IDirect3DTexture9 *ppTexture = nullptr;
			TRY_D3D(pWrapped->CreateTexture(Width, Height, Levels, Usage, (D3DFORMAT)Format, (D3DPOOL)Pool, &ppTexture, nullptr));
			return gcnew Direct3DTexture9(ppTexture);
		}
		Direct3DSurface9^ CreateRenderTarget(unsigned int Width, unsigned int Height, D3DFormat Format, D3DMultisample MultiSample, DWORD MultisampleQuality, BOOL Lockable)
		{
			IDirect3DSurface9 *ppSurface = nullptr;
			TRY_D3D(pWrapped->CreateRenderTarget(Width, Height, (D3DFORMAT)Format, (D3DMULTISAMPLE_TYPE)MultiSample, MultisampleQuality, Lockable, &ppSurface, nullptr));
			return gcnew Direct3DSurface9(ppSurface);
		}
		Direct3DSurface9^ CreateDepthStencilSurface(unsigned int Width, unsigned int Height, D3DFormat Format, D3DMultisample MultiSample, DWORD MultisampleQuality, BOOL Discard)
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
		void DrawPrimitiveUP(D3DPrimitiveType PrimitiveType, unsigned int PrimitiveCount, System::IntPtr ^pVertexStreamZeroData, unsigned int VertexStreamZeroStride)
		{
			TRY_D3D(pWrapped->DrawPrimitiveUP((D3DPRIMITIVETYPE)PrimitiveType, PrimitiveCount, pVertexStreamZeroData->ToPointer(), VertexStreamZeroStride));
		}
	};
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
}