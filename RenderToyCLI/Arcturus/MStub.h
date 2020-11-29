#pragma once

#include "IObject.h"
#include <assert.h>

namespace Arcturus
{
    public ref class CObjectBase
    {
    public:
        CObjectBase(Arcturus::IObject* object);
        Arcturus::IObject* m_object;
    private:
        virtual ~CObjectBase();
        !CObjectBase();
        void Destroy();
    };

    template <class T>
    public ref class CObjectStub : CObjectBase
    {
    public:
        CObjectStub(T* object) : CObjectBase(object)
        {
        }
        T* Typed()
        {
            return dynamic_cast<T*>(m_object);
        }
    };

    template <class T>
    T* GetStubTarget(System::Object^ p)
    {
        if (p == nullptr)
        {
            return nullptr;
        }
        auto base = dynamic_cast<CObjectBase^>(p);
        assert(base != nullptr);
        auto interior = dynamic_cast<T*>(base->m_object);
        assert(interior != nullptr);
        return interior;
    }
}