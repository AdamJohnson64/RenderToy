#pragma once
////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

namespace RenderToy
{
	public ref class COMWrapperBase
	{
	public:
		property System::IntPtr ManagedPtr
		{
			System::IntPtr get()
			{
				return System::IntPtr(pWrapped);
			}
		}
	protected:
		COMWrapperBase()
		{
			this->pWrapped = nullptr;
		}
		COMWrapperBase(IUnknown* pWrapped)
		{
			this->pWrapped = pWrapped;
		}
		!COMWrapperBase()
		{
			Destroy();
		}
		~COMWrapperBase()
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
		IUnknown *pWrapped;
	};
	template <typename T>
	public ref class COMWrapper : public COMWrapperBase
	{
	public:
		COMWrapper() : COMWrapperBase(nullptr)
		{
		}
		COMWrapper(T* pWrapped) : COMWrapperBase(pWrapped)
		{
		}
		inline T* WrappedInterface()
		{
			return reinterpret_cast<T*>(pWrapped);
		}
	};
}