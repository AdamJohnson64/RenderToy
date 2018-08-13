#include <amp.h>
#include "AMPArrayViewWrap.h"

namespace RenderToy
{
	class AMPArrayViewWrap : public IAMPMemory
	{
	public:
		AMPArrayViewWrap(const void *data, int length)
		{
			View = new Concurrency::array_view<int, 1>(length / sizeof(int), (int*)data);
		}
		virtual ~AMPArrayViewWrap()
		{
			if (View != nullptr)
			{
				delete View;
				View = nullptr;
			}
		}
		virtual void* GetBlob() const
		{
			return View;
		}
	private:
		Concurrency::array_view<int, 1> *View;
	};
	IAMPMemory* CreateAMPBlob(const void *data, int length)
	{
		return new AMPArrayViewWrap(data, length);
	}
}