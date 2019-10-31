////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

#include <d3dcompiler.h>
#include <msclr\marshal_cppstd.h>
#include "InteropCommon.h"

#define TRY_D3D(FUNCTION) { HRESULT result = FUNCTION; if (result != S_OK) throw gcnew System::Exception("Direct3DCompiler Error"); }

namespace RenderToy
{
	#pragma region - Direct3DCompiler Enumerations -
	public enum struct D3DBlobPart
	{
		InputSignatureBlob = D3D_BLOB_INPUT_SIGNATURE_BLOB,
		OutputSignatureBlob = D3D_BLOB_OUTPUT_SIGNATURE_BLOB,
		InputAndOutputSignatureBlob = D3D_BLOB_INPUT_AND_OUTPUT_SIGNATURE_BLOB,
		PatchConstantSignatureBlob = D3D_BLOB_PATCH_CONSTANT_SIGNATURE_BLOB,
		AllSignatureBlob = D3D_BLOB_ALL_SIGNATURE_BLOB,
		DebugInfo = D3D_BLOB_DEBUG_INFO,
		LegacyShader = D3D_BLOB_LEGACY_SHADER,
		XNAPrepassShader = D3D_BLOB_XNA_PREPASS_SHADER,
		XNAShader = D3D_BLOB_XNA_SHADER,
		PDB = D3D_BLOB_PDB,
		PrivateData = D3D_BLOB_PRIVATE_DATA,
		RootSignature = D3D_BLOB_ROOT_SIGNATURE,
		DebugName = D3D_BLOB_DEBUG_NAME,

		// Test parts are only produced by special compiler versions and so
		// are usually not present in shaders.
		TestAlternateShader = D3D_BLOB_TEST_ALTERNATE_SHADER,
		CompileDetails = D3D_BLOB_TEST_COMPILE_DETAILS,
		CompilePerf = D3D_BLOB_TEST_COMPILE_PERF,
		CompileReport = D3D_BLOB_TEST_COMPILE_REPORT,
	};
	#pragma endregion
	#pragma region - Direct3D Compiler -
	public ref class D3DBlob : COMWrapper<ID3DBlob>
	{
	public:
		D3DBlob() : COMWrapper(nullptr)
		{
		}
		D3DBlob(ID3DBlob *pWrapped) : COMWrapper(pWrapped)
		{
		}
		System::IntPtr GetBufferPointer()
		{
			if (WrappedInterface() == nullptr) return System::IntPtr(nullptr);
			return System::IntPtr(WrappedInterface()->GetBufferPointer());
		}
		size_t GetBufferSize()
		{
			if (WrappedInterface() == nullptr) return 0;
			return WrappedInterface()->GetBufferSize();
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
		static D3DBlob^ D3DGetBlobPart(cli::array<byte> ^pSrcData, D3DBlobPart Part, UINT Flags)
		{
			ID3DBlob *ppPart = nullptr;
			pin_ptr<byte> pSrcDataM = &pSrcData[0];
			TRY_D3D(::D3DGetBlobPart(pSrcDataM, pSrcData->Length, (D3D_BLOB_PART)Part, Flags, &ppPart));
			return gcnew D3DBlob(ppPart);
		}
	};
	#pragma endregion
}