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

    static bool Contains(const DrawCircle& circle, const Vec2& point)
    {
        Vec2 c = { circle._center.X - point.X, circle._center.Y - point.Y };
        float l = sqrtf(c.X * c.X + c.Y * c.Y) - circle._radius;
        return fabs(l) <= circle._width / 2;
    }

    static bool Contains(const DrawLine& line, const Vec2& sp)
    {
        Vec2 d = { line._p2.X - line._p1.X, line._p2.Y - line._p1.Y };
        float l = sqrtf(d.X * d.X + d.Y * d.Y);
        d.X /= l;
        d.Y /= l;
        Vec2 n = { -d.Y, d.X };
        {
            float p1x = -d.X;
            float p1y = -d.Y;
            float p1d = -(p1x * line._p1.X + p1y * line._p1.Y + line._width / 2);
            float dp = p1x * sp.X + p1y * sp.Y + p1d;
            if (dp > 0) return false;
        }
        {
            float p1x = d.X;
            float p1y = d.Y;
            float p1d = -(p1x * line._p2.X + p1y * line._p2.Y + line._width / 2);
            float dp = p1x * sp.X + p1y * sp.Y + p1d;
            if (dp > 0) return false;
        }
        {
            float p1x = -n.X;
            float p1y = -n.Y;
            float p1d = -(p1x * line._p1.X + p1y * line._p1.Y + line._width / 2);
            float dp = p1x * sp.X + p1y * sp.Y + p1d;
            if (dp > 0) return false;
        }
        {
            float p1x = n.X;
            float p1y = n.Y;
            float p1d = -(p1x * line._p1.X + p1y * line._p1.Y + line._width / 2);
            float dp = p1x * sp.X + p1y * sp.Y + p1d;
            if (dp > 0) return false;
        }
        return true;
    }

    static bool Contains(const DrawRectangle& rectangle, const Vec2& sp)
    {
        // Define two nested triangles; outer and inner, adjusted by half the stroke width.
        // Points inside the outer and not inside the inner are inside the rectangle.
        // Check that we're inside the outer rectangle.
        if (sp.X <= rectangle._topLeft.X - rectangle._width / 2 ||
            sp.X >= rectangle._bottomRight.X + rectangle._width / 2 ||
            sp.Y <= rectangle._topLeft.Y - rectangle._width / 2 ||
            sp.Y >= rectangle._bottomRight.Y + rectangle._width / 2) return false;
        // Check that we're NOT inside the inner rectangle.
        if (!(sp.X <= rectangle._topLeft.X + rectangle._width / 2 ||
            sp.X >= rectangle._bottomRight.X - rectangle._width / 2 ||
            sp.Y <= rectangle._topLeft.Y + rectangle._width / 2 ||
            sp.Y >= rectangle._bottomRight.Y - rectangle._width / 2)) return false;
        return true;
    }

    static bool Contains(const FillCircle& circle, const Vec2& sp)
    {
        Vec2 c = { circle._center.X - sp.X, circle._center.Y - sp.Y };
        float l = sqrtf(c.X * c.X + c.Y * c.Y);
        return l <= circle._radius;
    }

    static bool Contains(const FillRectangle& rectangle, const Vec2& sp)
    {
        if (sp.X <= rectangle._topLeft.X) return false;
        if (sp.X >= rectangle._bottomRight.X) return false;
        if (sp.Y <= rectangle._topLeft.Y) return false;
        if (sp.Y >= rectangle._bottomRight.Y) return false;
        return true;
    }

    // Take a stream of tagged primitives and extract their head pointers as DrawPrimitives.
    // IMPORTANT: The pointers in this vector are derived from the stream - do NOT free the stream data!
    static std::vector<const DrawPrimitive*> RenderTo_Deserialize(const void* stream)
    {
        std::vector<const DrawPrimitive*> primitives;
        // Handle all primitive types.
        const uint8_t* pWalk = reinterpret_cast<const uint8_t*>(stream);
        Vec4 colorPixel = {};
        while (true)
        {
            // Extract the leading tag from the primitive.
            switch (*reinterpret_cast<const PrimitiveType*>(pWalk))
            {
            case PrimitiveType::DRAW_CIRCLE:
                {
                    primitives.push_back(reinterpret_cast<const DrawPrimitive*>(pWalk));
                    pWalk += sizeof(uint32_t) + sizeof(DrawCircle);
                }
                break;
            case PrimitiveType::DRAW_LINE:
                {
                    primitives.push_back(reinterpret_cast<const DrawPrimitive*>(pWalk));
                    pWalk += sizeof(uint32_t) + sizeof(DrawLine);
                }
                break;
            case PrimitiveType::DRAW_RECTANGLE:
                {
                    primitives.push_back(reinterpret_cast<const DrawPrimitive*>(pWalk));
                    pWalk += sizeof(uint32_t) + sizeof(DrawRectangle);
                }
                break;
            case PrimitiveType::FILL_CIRCLE:
                {
                    primitives.push_back(reinterpret_cast<const DrawPrimitive*>(pWalk));
                    pWalk += sizeof(uint32_t) + sizeof(FillCircle);
                }
                break;
            case PrimitiveType::FILL_RECTANGLE:
                {
                    primitives.push_back(reinterpret_cast<const DrawPrimitive*>(pWalk));
                    pWalk += sizeof(uint32_t) + sizeof(FillRectangle);
                }
                break;
            case PrimitiveType::END:
                goto DONEEXTRACT;
            default:
                throw std::exception("Bad primitive in stream.");
            }
        }
    DONEEXTRACT:
        return primitives;
    }

    static void RenderTo_Baseline(void* pixels, uint32_t width, uint32_t height, uint32_t stride, const std::vector<const DrawPrimitive*>& primitives)
    {
        // Walk every pixel in the image.
        for (uint32_t y = 0; y < height; ++y)
        {
            void *pRaster = (char*)pixels + stride * y;
            for (uint32_t x = 0; x < width; ++x)
            {
                // 16 sample uniform grid sampling, centered around the mid-pixel.
                int samples = 0;
                const int SAMPLESIZE = 8;
                Vec4 colorTotal = {};
                for (int sy = 0; sy < SAMPLESIZE; ++sy)
                {
                    for (int sx = 0; sx < SAMPLESIZE; ++sx)
                    {
                        Vec2 sp = { x + (sx + 0.5f) / SAMPLESIZE, y + (sy + 0.5f) / SAMPLESIZE };
                        // Handle all primitive types.
                        Vec4 colorPixel = {};
                        for (const DrawPrimitive* primitive : primitives)
                        {
                            switch (primitive->primitive)
                            {
                            case PrimitiveType::DRAW_CIRCLE:
                                if (Contains(primitive->drawCircle, sp)) colorPixel = primitive->drawCircle._color;
                                break;
                            case PrimitiveType::DRAW_LINE:
                                if (Contains(primitive->drawLine, sp)) colorPixel = primitive->drawLine._color;
                                break;
                            case PrimitiveType::DRAW_RECTANGLE:
                                if (Contains(primitive->drawRectangle, sp)) colorPixel = primitive->drawRectangle._color;
                                break;
                            case PrimitiveType::FILL_CIRCLE:
                                if (Contains(primitive->fillCircle, sp)) colorPixel = primitive->fillCircle._color;
                                break;
                            case PrimitiveType::FILL_RECTANGLE:
                                if (Contains(primitive->fillRectangle, sp)) colorPixel = primitive->fillRectangle._color;
                                break;
                            }
                        }
                    DONE:
                        // Add in this sample contribution.
                        colorTotal = Add(colorTotal, colorPixel);
                    }
                }
                // Compute the average contribution.
                colorTotal = Divide(colorTotal, SAMPLESIZE * SAMPLESIZE);
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
        RenderTo_Baseline(pixels, width, height, stride, RenderTo_Deserialize(&_data[0]));
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