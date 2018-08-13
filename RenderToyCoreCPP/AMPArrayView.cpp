#include "AMPArrayView.h"

namespace RenderToy
{
	AMPArrayView::AMPArrayView(System::IntPtr data, int length)
	{
		View = CreateAMPBlob(data.ToPointer(), length);
	}
	AMPArrayView::~AMPArrayView()
	{
		if (View != nullptr)
		{
			delete View;
			View = nullptr;
		}
	}
	AMPArrayView::!AMPArrayView()
	{
		if (View != nullptr)
		{
			delete View;
			View = nullptr;
		}
	}
	IAMPMemory* AMPArrayView::GetView()
	{
		return View;
	}
}