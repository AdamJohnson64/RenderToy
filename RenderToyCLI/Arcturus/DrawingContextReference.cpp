#define _USE_MATH_DEFINES

#include "DrawingContextReference.h"

#include "DrawingContextCPU.h"
#include "Vector.h"

#include <stdint.h>

namespace Arcturus
{
    static float Max(float a, float b)
    {
        return a > b ? a : b;
    }

    static float Min(float a, float b)
    {
        return a < b ? a : b;
    }

    static float Saturate(float x)
    {
        return Max(0, Min(x, 1));
    }

    static uint32_t ToColorUint32(const Vec4& c)
    {
        uint32_t r = Saturate(c.X) * 255;
        uint32_t g = Saturate(c.Y) * 255;
        uint32_t b = Saturate(c.Z) * 255;
        uint32_t a = Saturate(c.W) * 255;
        return (a << 24) | (b << 16) | (g << 8) | (r << 0);
    }

    DrawingContextReference::DrawingContextReference()
    {
        reset();
    }

    void DrawingContextReference::reset()
    {
        _color = Vec4{ 1, 1, 1, 1 };
        _cursor = Vec2{ 0, 0 };
        _width = 1;
        _data.clear();
    }

    void DrawingContextReference::renderTo(void* pixels, uint32_t width, uint32_t height, uint32_t stride)
    {
        // Close the stream with an END tag.
        {
            uint32_t sizeCurrent = _data.size();
            uint32_t sizeRequired = sizeof(uint32_t);
            _data.resize(sizeCurrent + sizeRequired);
            uint8_t* p = &_data[sizeCurrent];
            *(PrimitiveType*)p = PrimitiveType::END;
        }
        // Extract the list of primitives so we don't have to keep reparsing the input.
        RenderTo_Fast(pixels, width, height, stride, RenderTo_Deserialize(&_data[0]));
    }

    void DrawingContextReference::setColor(const Vec4& color)
    {
        _color = color;
    }

    void DrawingContextReference::setWidth(float width)
    {
        _width = width;
    }

    void DrawingContextReference::moveTo(const Vec2& point)
    {
        _cursor = point;
    }

    void DrawingContextReference::lineTo(const Vec2& point)
    {
        uint32_t sizeCurrent = _data.size();
        uint32_t sizeRequired = sizeof(uint32_t) + sizeof(DrawLine);
        _data.resize(sizeCurrent + sizeRequired);
        uint8_t* p = &_data[sizeCurrent];
        *(PrimitiveType*)p = PrimitiveType::DRAW_LINE;
        p += sizeof(PrimitiveType);
        DrawLine* primitive = (DrawLine*)p;
        primitive->_color = _color;
        primitive->_p1 = _cursor;
        primitive->_p2 = point;
        primitive->_width = _width;
        _cursor = point;
    }

    void DrawingContextReference::drawCircle(const Vec2& point, float radius)
    {
        uint32_t sizeCurrent = _data.size();
        uint32_t sizeRequired = sizeof(uint32_t) + sizeof(DrawCircle);
        _data.resize(sizeCurrent + sizeRequired);
        uint8_t* p = &_data[sizeCurrent];
        *(PrimitiveType*)p = PrimitiveType::DRAW_CIRCLE;
        p += sizeof(PrimitiveType);
        DrawCircle* primitive = (DrawCircle*)p;
        primitive->_color = _color;
        primitive->_center = point;
        primitive->_radius = radius;
        primitive->_width = _width;
    }

    void DrawingContextReference::drawRectangle(const Vec2& topLeft, const Vec2& bottomRight)
    {
        uint32_t sizeCurrent = _data.size();
        uint32_t sizeRequired = sizeof(uint32_t) + sizeof(DrawRectangle);
        _data.resize(sizeCurrent + sizeRequired);
        uint8_t* p = &_data[sizeCurrent];
        *(PrimitiveType*)p = PrimitiveType::DRAW_RECTANGLE;
        p += sizeof(PrimitiveType);
        DrawRectangle* primitive = (DrawRectangle*)p;
        primitive->_color = _color;
        primitive->_topLeft = topLeft;
        primitive->_bottomRight = bottomRight;
        primitive->_width = _width;
    }

    void DrawingContextReference::fillCircle(const Vec2& point, float radius)
    {
        uint32_t sizeCurrent = _data.size();
        uint32_t sizeRequired = sizeof(uint32_t) + sizeof(FillCircle);
        _data.resize(sizeCurrent + sizeRequired);
        uint8_t* p = &_data[sizeCurrent];
        *(PrimitiveType*)p = PrimitiveType::FILL_CIRCLE;
        p += sizeof(PrimitiveType);
        FillCircle* primitive = (FillCircle*)p;
        primitive->_color = _color;
        primitive->_center = point;
        primitive->_radius = radius;
    }

    void DrawingContextReference::fillRectangle(const Vec2& topLeft, const Vec2& bottomRight)
    {
        uint32_t sizeCurrent = _data.size();
        uint32_t sizeRequired = sizeof(uint32_t) + sizeof(FillRectangle);
        _data.resize(sizeCurrent + sizeRequired);
        uint8_t* p = &_data[sizeCurrent];
        *(PrimitiveType*)p = PrimitiveType::FILL_RECTANGLE;
        p += sizeof(PrimitiveType);
        FillRectangle* primitive = (FillRectangle*)p;
        primitive->_color = _color;
        primitive->_topLeft = topLeft;
        primitive->_bottomRight = bottomRight;
    }

}