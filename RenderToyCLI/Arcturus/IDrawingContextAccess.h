#pragma once

#include "Vector.h"

#include <stdint.h>

namespace Arcturus
{
    class IDrawingContextAccess
    {
        virtual uint32_t vertexCount() const = 0;
        virtual const void* vertexPointer() const = 0;
        virtual uint32_t indexCount() const = 0;
        virtual const uint32_t* indexPointer() const = 0;
    };
}