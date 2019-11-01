#pragma once

#include <stdint.h>

namespace Arcturus
{
    namespace Managed
    {
        public interface class IDrawingContextAccess
        {
        public:
            virtual uint32_t vertexCount() = 0;
            virtual System::IntPtr vertexPointer() = 0;
            virtual uint32_t indexCount() = 0;
            virtual System::IntPtr indexPointer() = 0;
        };
    }
}