#pragma once

#include <cstdint>
#include <vector>

#include "Vector.h"

namespace Arcturus
{
    enum class PrimitiveType : uint32_t
    {
        UNKNOWN = 0,
        DRAW_CIRCLE = 1,
        DRAW_LINE = 2,
        DRAW_RECTANGLE = 3,
        FILL_CIRCLE = 4,
        FILL_RECTANGLE = 5,
        END = 6,
    };

    struct DrawCircle
    {
        Vec4 _color;
        Vec2 _center;
        float _radius;
        float _width;
    };

    struct DrawLine
    {
        Vec4 _color;
        Vec2 _p1;
        Vec2 _p2;
        float _width;
    };

    struct DrawRectangle
    {
        Vec4 _color;
        Vec2 _topLeft;
        Vec2 _bottomRight;
        float _width;
    };

    struct FillCircle
    {
        Vec4 _color;
        Vec2 _center;
        float _radius;
    };

    struct FillRectangle
    {
        Vec4 _color;
        Vec2 _topLeft;
        Vec2 _bottomRight;
    };

    struct DrawPrimitive
    {
        PrimitiveType primitive;
        union
        {
            DrawCircle drawCircle;
            DrawLine drawLine;
            DrawRectangle drawRectangle;
            FillCircle fillCircle;
            FillRectangle fillRectangle;
        };
    };

    // Take a stream of tagged primitives and extract their head pointers as DrawPrimitives.
    // IMPORTANT: The pointers in this vector are derived from the stream - do NOT free the stream data!
    std::vector<const DrawPrimitive*> RenderTo_Deserialize(const void* stream);

    void RenderTo_Baseline(void* pixels, uint32_t width, uint32_t height, uint32_t stride, const std::vector<const DrawPrimitive*>& primitives);

    void RenderTo_Fast(void* pixels, uint32_t width, uint32_t height, uint32_t stride, const std::vector<const DrawPrimitive*> primitives);
}