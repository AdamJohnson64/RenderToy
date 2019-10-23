#pragma once

#include <stdint.h>

namespace Arcturus
{
    namespace Managed
    {
        public value struct Matrix { float M11, M12, M13, M14, M21, M22, M23, M24, M31, M32, M33, M34, M41, M42, M43, M44; };
        public value struct Vec2 { float X, Y; };
        public value struct Vec4 { float X, Y, Z, W; };
        public value struct Vertex { Vec2 Position; uint32_t Color; };
    }
}