#pragma once

#include "AMPArrayViewWrap.h"

namespace RenderToy
{
	public ref class AMPArrayView
	{
	public:
		AMPArrayView(System::IntPtr data, int length);
		~AMPArrayView();
		!AMPArrayView();
		IAMPMemory* GetView();
	private:
		IAMPMemory *View;
	};
}