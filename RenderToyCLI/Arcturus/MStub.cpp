#include "MStub.h"

#include <assert.h>

namespace Arcturus
{
    CObjectBase::CObjectBase(Arcturus::IObject* object) : m_object(object)
    {
        assert(object != nullptr);
    }

    CObjectBase::~CObjectBase()
    {
        Destroy();
    }

    CObjectBase::!CObjectBase()
    {
        Destroy();
    }

    void CObjectBase::Destroy()
    {
        if (m_object != nullptr)
        {
            delete m_object;
            m_object = nullptr;
        }
    }
}