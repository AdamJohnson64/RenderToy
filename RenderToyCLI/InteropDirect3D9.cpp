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
	public enum class D3DFormat
	{
		A8R8G8B8 = D3DFMT_A8R8G8B8,
		D24X8 = D3DFMT_D24X8,
	};
	public enum class D3DMultisample
	{
		None = D3DMULTISAMPLE_NONE,
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
			TRY_D3D(WrappedInterface()->LockRect(&LockedRect, nullptr, D3DLOCK_READONLY));
			auto result = gcnew D3DLockedRect();
			result->Pitch = LockedRect.Pitch;
			result->Bits = System::IntPtr(LockedRect.pBits);
			return result;
		}
		void UnlockRect()
		{
			TRY_D3D(WrappedInterface()->UnlockRect());
		}
	};
	#pragma endregion
	#pragma region - Direct3DDevice9Ex -
	generic <typename T>
	public ref class NullablePtr
	{
	public:
		NullablePtr(T value)
		{
			Value = value;
		}
		T Value;
	};
	public ref class Direct3DDevice9Ex : public COMWrapper<IDirect3DDevice9Ex>
	{
	public:
		Direct3DDevice9Ex(IDirect3DDevice9Ex *pObject)
		{
			pWrapped = pObject;
		}
		Direct3DSurface9^ CreateRenderTarget(UINT Width, UINT Height, D3DFormat Format, D3DMultisample MultiSample, DWORD MultisampleQuality, BOOL Lockable, NullablePtr<System::IntPtr> ^ppSharedHandle)
		{
			IDirect3DSurface9 *ppSurface = nullptr;
			HANDLE hSharedHandle = ppSharedHandle == nullptr ? nullptr : ppSharedHandle->Value.ToPointer();
			TRY_D3D(WrappedInterface()->CreateRenderTarget(Width, Height, (D3DFORMAT)Format, (D3DMULTISAMPLE_TYPE)MultiSample, MultisampleQuality, Lockable, &ppSurface, ppSharedHandle == nullptr ? nullptr : &hSharedHandle));
			if (ppSharedHandle != nullptr) ppSharedHandle->Value = System::IntPtr(hSharedHandle);
			return gcnew Direct3DSurface9(ppSurface);
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
		Direct3DDevice9Ex^ CreateDevice(System::IntPtr hFocusWindow)
		{
			D3DPRESENT_PARAMETERS d3dpp = { 0 };
			d3dpp.BackBufferWidth = 1;
			d3dpp.BackBufferHeight = 1;
			d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
			d3dpp.Windowed = TRUE;
			IDirect3DDevice9Ex* pReturnedDeviceInterface = nullptr;
			TRY_D3D(WrappedInterface()->CreateDeviceEx(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, reinterpret_cast<HWND>(hFocusWindow.ToPointer()), D3DCREATE_FPU_PRESERVE | D3DCREATE_MULTITHREADED | D3DCREATE_HARDWARE_VERTEXPROCESSING, &d3dpp, nullptr, &pReturnedDeviceInterface));
			return gcnew Direct3DDevice9Ex(pReturnedDeviceInterface);
		}
	};
	#pragma endregion
}