#pragma once

#include <stdint.h>

namespace Arcturus
{
    struct Vec2 { float X, Y; };
    struct Vec4 { float X, Y, Z, W; };
    struct Vertex { Vec2 Position; uint32_t Color; };

    Vec4 Add(const Vec4& lhs, const Vec4& rhs);
    Vec4 Divide(const Vec4& lhs, float rhs);
    Vec4 Multiply(const Vec4& lhs, float rhs);
    Vec4 Multiply(float lhs, const Vec4& rhs);
}