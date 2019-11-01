#pragma once

#include <stdint.h>

namespace Arcturus
{
    struct Vec2 { float X, Y; };
    struct Vec4 { float X, Y, Z, W; };
    struct Vertex { Vec2 Position; uint32_t Color; };
}