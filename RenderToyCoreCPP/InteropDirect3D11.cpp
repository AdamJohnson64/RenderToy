////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

#include <memory>
#include <msclr\marshal_cppstd.h>
#include <d3d9.h>
#include <d3d11_4.h>
#include "InteropCommon.h"

#define TRY_D3D(FUNCTION) if (FUNCTION != S_OK) throw gcnew System::Exception(#FUNCTION);

using namespace System::Runtime::InteropServices;

namespace RenderToy
{
	public value struct MIDL_D3D11_INPUT_ELEMENT_DESC
	{
	public:
		typedef RenderToyCOM::D3D11_INPUT_ELEMENT_DESC	ManagedType;
		typedef D3D11_INPUT_ELEMENT_DESC				UnmanagedType;
	public:
		System::String^								SemanticName;
		UINT										SemanticIndex;
		RenderToyCOM::DXGI_FORMAT					Format;
		UINT										InputSlot;
		UINT										AlignedByteOffset;
		RenderToyCOM::D3D11_INPUT_CLASSIFICATION	InputSlotClass;
		UINT										InstanceDataStepRate;
	};
	void Marshal(D3D11_INPUT_ELEMENT_DESC &dst, MIDL_D3D11_INPUT_ELEMENT_DESC %src, msclr::interop::marshal_context ^ctx)
	{
		dst.SemanticName = ctx->marshal_as<LPCSTR>(src.SemanticName);
		dst.SemanticIndex = src.SemanticIndex;
		dst.Format = (DXGI_FORMAT)src.Format;
		dst.InputSlot = src.InputSlot;
		dst.AlignedByteOffset = src.AlignedByteOffset;
		dst.InputSlotClass = (D3D11_INPUT_CLASSIFICATION)src.InputSlotClass;
		dst.InstanceDataStepRate = src.InstanceDataStepRate;
	}
	template <class T>
	void MarshalArray(typename T::UnmanagedType *dst, cli::array<T> ^src, msclr::interop::marshal_context ^ctx)
	{
		for (int i = 0; i < src->Length; ++i)
		{
			Marshal(dst[i], src[i], ctx);
		}
	};
	public ref class D3D11Shim
	{
	public:
		static void Device_CreateInputLayout(RenderToyCOM::ID3D11Device ^device, cli::array<MIDL_D3D11_INPUT_ELEMENT_DESC> ^%src, System::IntPtr pShaderBytecodeWithInputSignature, int BytecodeLength, RenderToyCOM::ID3D11InputLayout ^%ppInputLayout)
		{
			msclr::interop::marshal_context ctx;
			auto dst = std::unique_ptr<D3D11_INPUT_ELEMENT_DESC[]>(new D3D11_INPUT_ELEMENT_DESC[src->Length]);
			MarshalArray<MIDL_D3D11_INPUT_ELEMENT_DESC>(dst.get(), src, %ctx);
			ID3D11InputLayout *ppLayout = nullptr;
			auto unmanagedDevice = Marshal::GetComInterfaceForObject(device, RenderToyCOM::ID3D11Device::typeid);
			Marshal::AddRef(unmanagedDevice);
			TRY_D3D(((ID3D11Device*)unmanagedDevice.ToPointer())->CreateInputLayout(dst.get(), src->Length, pShaderBytecodeWithInputSignature.ToPointer(), BytecodeLength, &ppLayout));
			ppInputLayout = (RenderToyCOM::ID3D11InputLayout^)Marshal::GetTypedObjectForIUnknown(System::IntPtr(ppLayout), RenderToyCOM::ID3D11InputLayout::typeid);
		}
	};
	public ref class Direct3D11
	{
	public:
		static RenderToyCOM::ID3D11Device^ D3D11CreateDevice()
		{
			ID3D11Device *ppDevice = nullptr;
			D3D_FEATURE_LEVEL featurelevel = D3D_FEATURE_LEVEL_12_1;
			D3D11_CREATE_DEVICE_FLAG flags;
#ifdef _DEBUG
			flags = D3D11_CREATE_DEVICE_DEBUG;
#else
			flags = (D3D11_CREATE_DEVICE_FLAG)0;
#endif
			TRY_D3D(::D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, &featurelevel, 1, D3D11_SDK_VERSION, &ppDevice, nullptr, nullptr));
			//TRY_D3D(::D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, flags, &featurelevel, 1, D3D11_SDK_VERSION, &ppDevice, nullptr, nullptr));
			return (RenderToyCOM::ID3D11Device^)Marshal::GetTypedObjectForIUnknown(System::IntPtr(ppDevice), RenderToyCOM::ID3D11Device::typeid);
		}
	};
}