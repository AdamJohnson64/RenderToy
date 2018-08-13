#pragma once

namespace RenderToy
{
	class IAMPMemory
	{
	public:
		virtual void* GetBlob() const = 0;
	};
	IAMPMemory *CreateAMPBlob(const void *data, int length);
}