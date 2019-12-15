#define _USE_MATH_DEFINES

#include "DrawingContextReference.h"
#include "Vector.h"

#include <stdint.h>

namespace Arcturus
{
    enum class PrimitiveType : uint32_t
    {
        UNKNOWN = 0,
        DRAW_CIRCLE = 1,
        FILL_CIRCLE = 2,
        END = 3,
    };

    struct DrawCircle
    {
        Vec2 _center;
        float _radius;
        float _width;
    };

    struct FillCircle
    {
        Vec2 _center;
        float _radius;
    };

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
        {
            uint32_t sizeCurrent = _data.size();
            uint32_t sizeRequired = sizeof(uint32_t);
            _data.resize(sizeCurrent + sizeRequired);
            uint8_t* p = &_data[sizeCurrent];
            *(PrimitiveType*)p = PrimitiveType::END;
        }
        // Lines next.
        //Vec2 p1 = { 10, 10 };
        //Vec2 p2 = { 99, 99 };
        //Vec2 d = { p2.X - p1.X, p2.Y - p1.Y };
        //Vec2 n = { d.Y, -d.X };
        // Walk every pixel in the image.
        for (uint32_t y = 0; y < height; ++y)
        {
            void *pRaster = (char*)pixels + stride * y;
            for (uint32_t x = 0; x < width; ++x)
            {
                // 16 sample uniform grid sampling.
                int samples = 0;
                const int SAMPLESIZE = 8;
                for (int sy = 0; sy < SAMPLESIZE; ++sy)
                {
                    for (int sx = 0; sx < SAMPLESIZE; ++sx)
                    {
                        Vec2 sp = { x + sx / (SAMPLESIZE + 1.0f), y + sy / (SAMPLESIZE + 1.0f) };
                        // Handle all primitive types.
                        uint8_t* pWalk = &_data[0];
                        while (true)
                        {
                            switch (*(PrimitiveType*)pWalk)
                            {
                            case PrimitiveType::DRAW_CIRCLE:
                                {
                                    DrawCircle* pPrimitive = (DrawCircle*)(pWalk + sizeof(uint32_t));
                                    Vec2 c = { pPrimitive->_center.X - sp.X, pPrimitive->_center.Y - sp.Y };
                                    float l = sqrtf(c.X * c.X + c.Y * c.Y) - pPrimitive->_radius;
                                    if (fabs(l) <= pPrimitive->_width / 2) ++samples;
                                    pWalk += sizeof(uint32_t) + sizeof(DrawCircle);
                                }
                                break;
                            case PrimitiveType::FILL_CIRCLE:
                                {
                                    FillCircle* pPrimitive = (FillCircle*)(pWalk + sizeof(uint32_t));
                                    Vec2 c = { pPrimitive->_center.X - sp.X, pPrimitive->_center.Y - sp.Y };
                                    float l = sqrtf(c.X * c.X + c.Y * c.Y);
                                    if (l <= pPrimitive->_radius) ++samples;
                                    pWalk += sizeof(uint32_t) + sizeof(FillCircle);
                                }
                                break;
                            case PrimitiveType::END:
                                goto DONE;
                            }
                        }
                    DONE:
                        int test = 0;
                    }
                }
                uint32_t value = samples * 255 / (SAMPLESIZE * SAMPLESIZE);
                // Set the final contribution for this pixel.
                void *pPixel = (char*)pRaster + sizeof(uint32_t) * x;
                uint32_t *tPixel = (uint32_t*)pPixel;
                *tPixel = (value << 24) | (value << 16) | (value << 8) | (value << 0);
            }
        }
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
        primitive->_center = point;
        primitive->_radius = radius;
        primitive->_width = _width;
    }

    void DrawingContextReference::drawRectangle(const Vec2& topLeft, const Vec2& bottomRight)
    {
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
        primitive->_center = point;
        primitive->_radius = radius;
    }

    void DrawingContextReference::fillRectangle(const Vec2& topLeft, const Vec2& bottomRight)
    {
    }

}