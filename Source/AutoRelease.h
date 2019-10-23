#pragma once

#include <assert.h>

namespace Arcturus
{
    template <class T>
    class AutoRelease
    {
    public:
        AutoRelease() : p(nullptr)
        {
        }
        AutoRelease(T* p) : p(p)
        {
        }
        ~AutoRelease()
        {
            Destroy();
        }
        void Destroy()
        {
            if (p != nullptr)
            {
                p->Release();
                p = nullptr;
            }
        }
        T** operator&()
        {
            // We usually take this pointer for initialization in COM calls.
            // Be careful: If we already have an object then we won't release the previous object and we'll leak.
            return &p;
        }
        T* operator->() const
        {
            return p;
        }
        operator T* () const
        {
            return p;
        }
        T* p;
    };
}