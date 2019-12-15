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
        // Walk every pixel in the image.
        for (uint32_t y = 0; y < height; ++y)
        {
            void *pRaster = (char*)pixels + stride * y;
            for (uint32_t x = 0; x < width; ++x)
            {
                // 16 sample uniform grid sampling.
                int samples = 0;
                const int SAMPLESIZE = 8;
                Vec4 colorTotal = {};
                for (int sy = 0; sy < SAMPLESIZE; ++sy)
                {
                    for (int sx = 0; sx < SAMPLESIZE; ++sx)
                    {
                        Vec2 sp = { x + sx / (SAMPLESIZE + 1.0f), y + sy / (SAMPLESIZE + 1.0f) };
                        // Handle all primitive types.
                        uint8_t* pWalk = &_data[0];
                        Vec4 colorPixel = {};
                        while (true)
                        {
                            switch (*(PrimitiveType*)pWalk)
                            {
                            case PrimitiveType::DRAW_CIRCLE:
                                {
                                    DrawCircle* pPrimitive = (DrawCircle*)(pWalk + sizeof(uint32_t));
                                    Vec2 c = { pPrimitive->_center.X - sp.X, pPrimitive->_center.Y - sp.Y };
                                    float l = sqrtf(c.X * c.X + c.Y * c.Y) - pPrimitive->_radius;
                                    if (fabs(l) <= pPrimitive->_width / 2) colorPixel = pPrimitive->_color;
                                    pWalk += sizeof(uint32_t) + sizeof(DrawCircle);
                                }
                                break;
                            case PrimitiveType::DRAW_LINE:
                                {
                                    DrawLine* pPrimitive = (DrawLine*)(pWalk + sizeof(uint32_t));
                                    Vec2 d = { pPrimitive->_p2.X - pPrimitive->_p1.X, pPrimitive->_p2.Y - pPrimitive->_p1.Y };
                                    float l = sqrtf(d.X * d.X + d.Y * d.Y);
                                    d.X /= l;
                                    d.Y /= l;
                                    Vec2 n = { -d.Y, d.X };
                                    bool inside = true;
                                    {
                                        float p1x = -d.X;
                                        float p1y = -d.Y;
                                        float p1d = -(p1x * pPrimitive->_p1.X + p1y * pPrimitive->_p1.Y + pPrimitive->_width / 2);
                                        float dp = p1x * sp.X + p1y * sp.Y + p1d;
                                        if (dp > 0) inside = false;
                                    }
                                    {
                                        float p1x = d.X;
                                        float p1y = d.Y;
                                        float p1d = -(p1x * pPrimitive->_p2.X + p1y * pPrimitive->_p2.Y + pPrimitive->_width / 2);
                                        float dp = p1x * sp.X + p1y * sp.Y + p1d;
                                        if (dp > 0) inside = false;
                                    }
                                    {
                                        float p1x = -n.X;
                                        float p1y = -n.Y;
                                        float p1d = -(p1x * pPrimitive->_p1.X + p1y * pPrimitive->_p1.Y + pPrimitive->_width / 2);
                                        float dp = p1x * sp.X + p1y * sp.Y + p1d;
                                        if (dp > 0) inside = false;
                                    }
                                    {
                                        float p1x = n.X;
                                        float p1y = n.Y;
                                        float p1d = -(p1x * pPrimitive->_p1.X + p1y * pPrimitive->_p1.Y + pPrimitive->_width / 2);
                                        float dp = p1x * sp.X + p1y * sp.Y + p1d;
                                        if (dp > 0) inside = false;
                                    }
                                    if (inside)
                                    {
                                        colorPixel = pPrimitive->_color;
                                    }
                                    else
                                    {
                                        int test = 0;
                                    }
                                    pWalk += sizeof(uint32_t) + sizeof(DrawLine);
                                }
                                break;
                            case PrimitiveType::DRAW_RECTANGLE:
                                {
                                    DrawRectangle* pPrimitive = (DrawRectangle*)(pWalk + sizeof(uint32_t));
                                    // Define two nested triangles; outer and inner, adjusted by half the stroke width.
                                    // Points inside the outer and not inside the inner are inside the rectangle.
                                    bool inside1 = true;
                                    if (sp.X <= pPrimitive->_topLeft.X - pPrimitive->_width / 2) inside1 = false;
                                    if (sp.X >= pPrimitive->_bottomRight.X + pPrimitive->_width / 2) inside1 = false;
                                    if (sp.Y <= pPrimitive->_topLeft.Y - pPrimitive->_width / 2) inside1 = false;
                                    if (sp.Y >= pPrimitive->_bottomRight.Y + pPrimitive->_width / 2) inside1 = false;
                                    bool inside2 = false;
                                    if (sp.X <= pPrimitive->_topLeft.X + pPrimitive->_width / 2) inside2 = true;
                                    if (sp.X >= pPrimitive->_bottomRight.X - pPrimitive->_width / 2) inside2 = true;
                                    if (sp.Y <= pPrimitive->_topLeft.Y + pPrimitive->_width / 2) inside2 = true;
                                    if (sp.Y >= pPrimitive->_bottomRight.Y - pPrimitive->_width / 2) inside2 = true;
                                    if (inside1 && inside2) colorPixel = pPrimitive->_color;
                                    pWalk += sizeof(uint32_t) + sizeof(DrawRectangle);
                                }
                                break;
                            case PrimitiveType::FILL_CIRCLE:
                                {
                                    FillCircle* pPrimitive = (FillCircle*)(pWalk + sizeof(uint32_t));
                                    Vec2 c = { pPrimitive->_center.X - sp.X, pPrimitive->_center.Y - sp.Y };
                                    float l = sqrtf(c.X * c.X + c.Y * c.Y);
                                    if (l <= pPrimitive->_radius) colorPixel = pPrimitive->_color;
                                    pWalk += sizeof(uint32_t) + sizeof(FillCircle);
                                }
                                break;
                            case PrimitiveType::FILL_RECTANGLE:
                                {
                                    FillRectangle* pPrimitive = (FillRectangle*)(pWalk + sizeof(uint32_t));
                                    bool inside = true;
                                    if (sp.X <= pPrimitive->_topLeft.X) inside = false;
                                    if (sp.X >= pPrimitive->_bottomRight.X) inside = false;
                                    if (sp.Y <= pPrimitive->_topLeft.Y) inside = false;
                                    if (sp.Y >= pPrimitive->_bottomRight.Y) inside = false;
                                    if (inside) colorPixel = pPrimitive->_color;
                                    pWalk += sizeof(uint32_t) + sizeof(FillRectangle);
                                }
                                break;
                            case PrimitiveType::END:
                                goto DONE;
                            default:
                                goto DONE;
                            }
                        }
                    DONE:
                        // Add in this sample contribution.
                        colorTotal.X += colorPixel.X;
                        colorTotal.Y += colorPixel.Y;
                        colorTotal.Z += colorPixel.Z;
                        colorTotal.W += colorPixel.W;
                    }
                }
                // Compute the average contribution;
                colorTotal.X /= SAMPLESIZE * SAMPLESIZE;
                colorTotal.Y /= SAMPLESIZE * SAMPLESIZE;
                colorTotal.Z /= SAMPLESIZE * SAMPLESIZE;
                colorTotal.W /= SAMPLESIZE * SAMPLESIZE;
                // Set the final contribution for this pixel.
                void *pPixel = (char*)pRaster + sizeof(uint32_t) * x;
                uint8_t *tPixel = (uint8_t*)pPixel;
                tPixel[0] = (uint8_t)(colorTotal.Z * 255);
                tPixel[1] = (uint8_t)(colorTotal.Y * 255);
                tPixel[2] = (uint8_t)(colorTotal.X * 255);
                tPixel[3] = (uint8_t)(colorTotal.W * 255);
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