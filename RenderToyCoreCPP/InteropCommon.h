#pragma once
////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

namespace RenderToy
{
	template <typename T>
	public ref class COMWrapper
	{
	public:
		COMWrapper()
		{
			this->pWrapped = nullptr;
		}
		COMWrapper(T* pWrapped)
		{
			this->pWrapped = pWrapped;
		}
		!COMWrapper()
		{
			Destroy();
		}
		~COMWrapper()
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
		T * pWrapped;
	};
}