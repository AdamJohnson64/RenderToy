#pragma once
////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

namespace RenderToy
{
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
		T * pWrapped;
	};
}