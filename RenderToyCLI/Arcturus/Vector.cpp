#include "Vector.h"

namespace Arcturus
{
    Vec4 Add(const Vec4& lhs, const Vec4& rhs) { return Vec4 { lhs.X + rhs.X, lhs.Y + rhs.Y, lhs.Z + rhs.Z, lhs.W + rhs.W }; }
    Vec4 Divide(const Vec4& lhs, float rhs) { return Multiply(lhs, 1 / rhs); }
    Vec4 Multiply(const Vec4& lhs, float rhs) { return Vec4 { lhs.X * rhs, lhs.Y * rhs, lhs.Z * rhs, lhs.W * rhs }; }
    Vec4 Multiply(float lhs, const Vec4& rhs) { return Multiply(rhs, lhs); }
}