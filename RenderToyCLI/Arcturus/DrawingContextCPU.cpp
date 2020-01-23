#include "DrawingContextCPU.h"

#include <limits.h>
#include <math.h>

namespace Arcturus
{
    ////////////////////////////////////////////////////////////////////////////////
    // Basic Math Functions
    ////////////////////////////////////////////////////////////////////////////////

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

    ////////////////////////////////////////////////////////////////////////////////
    // Basic Types
    ////////////////////////////////////////////////////////////////////////////////

    struct Bitmap
    {
        void* pixels;
        uint32_t width, height, stride;
    };

    struct Rectangle
    {
        uint32_t x, y, w, h;
    };

    struct Quad
    {
        Vec2 topLeft;
        Vec2 bottomRight;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Primitive Sampling Functions
    ////////////////////////////////////////////////////////////////////////////////

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
        // Define two nested rectangles; outer and inner, adjusted by half the stroke width.
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

    ////////////////////////////////////////////////////////////////////////////////
    // Common Support Functions
    ////////////////////////////////////////////////////////////////////////////////

    // Take a stream of tagged primitives and extract their head pointers as DrawPrimitives.
    // IMPORTANT: The pointers in this vector are derived from the stream - do NOT free the stream data!
    std::vector<const DrawPrimitive*> RenderTo_Deserialize(const void* stream)
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
                primitives.push_back(reinterpret_cast<const DrawPrimitive*>(pWalk));
                pWalk += sizeof(PrimitiveType) + sizeof(DrawCircle);
                break;
            case PrimitiveType::DRAW_LINE:
                primitives.push_back(reinterpret_cast<const DrawPrimitive*>(pWalk));
                pWalk += sizeof(PrimitiveType) + sizeof(DrawLine);
                break;
            case PrimitiveType::DRAW_RECTANGLE:
                primitives.push_back(reinterpret_cast<const DrawPrimitive*>(pWalk));
                pWalk += sizeof(PrimitiveType) + sizeof(DrawRectangle);
                break;
            case PrimitiveType::FILL_CIRCLE:
                primitives.push_back(reinterpret_cast<const DrawPrimitive*>(pWalk));
                pWalk += sizeof(PrimitiveType) + sizeof(FillCircle);
                break;
            case PrimitiveType::FILL_RECTANGLE:
                primitives.push_back(reinterpret_cast<const DrawPrimitive*>(pWalk));
                pWalk += sizeof(PrimitiveType) + sizeof(FillRectangle);
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

    ////////////////////////////////////////////////////////////////////////////////
    // Baseline Rendering (SLOW)
    ////////////////////////////////////////////////////////////////////////////////

    struct PixelBGRA32
    {
        uint8_t b, g, r, a;
    };

    static void RenderTo_Baseline(const Bitmap& bitmap, const Rectangle& region, const std::vector<const DrawPrimitive*>& primitives)
    {
        // Walk every pixel in the image.
        for (uint32_t filly = 0; filly < region.h; ++filly)
        {
            uint32_t y = region.y + filly;
            void *pRaster = (char*)bitmap.pixels + bitmap.stride * y;
            for (uint32_t fillx = 0; fillx < region.w; ++fillx)
            {
                uint32_t x = region.x + fillx;
                // 16 sample uniform grid sampling, centered around the mid-pixel.
                const int SAMPLESIZE = 8;
                Vec4 colorTotal = {};
                for (uint32_t sy = 0; sy < SAMPLESIZE; ++sy)
                {
                    for (uint32_t sx = 0; sx < SAMPLESIZE; ++sx)
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
                void *pPixel = (char*)pRaster + sizeof(PixelBGRA32) * x;
                PixelBGRA32 *tPixel = reinterpret_cast<PixelBGRA32*>(pPixel);
                tPixel->b = (uint8_t)(colorTotal.Z * 255);
                tPixel->g = (uint8_t)(colorTotal.Y * 255);
                tPixel->r = (uint8_t)(colorTotal.X * 255);
                tPixel->a = (uint8_t)(colorTotal.W * 255);
            }
        }
    }

    void RenderTo_Baseline(void* pixels, uint32_t width, uint32_t height, uint32_t stride, const std::vector<const DrawPrimitive*> primitives)
    {
        RenderTo_Baseline(Bitmap { pixels, width, height, stride }, Rectangle { 0, 0, width, height}, primitives);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Fast Rendering (KD/Culled)
    ////////////////////////////////////////////////////////////////////////////////

    static Quad ConservativeBound(const DrawCircle& circle)
    {
        return {
            floorf(circle._center.X - circle._radius - circle._width / 2),
            floorf(circle._center.Y - circle._radius - circle._width / 2),
            ceilf(circle._center.X + circle._radius + circle._width / 2),
            ceilf(circle._center.Y + circle._radius + circle._width / 2)
        };
    }

    static Quad ConservativeBound(const DrawLine& line)
    {
        return {
            floorf(Min(line._p1.X, line._p2.X) - line._width / 2),
            floorf(Min(line._p1.Y, line._p2.Y) - line._width / 2),
            ceilf(Max(line._p1.X, line._p2.X) + line._width / 2),
            ceilf(Max(line._p1.Y, line._p2.Y) + line._width / 2)
        };
    }

    static Quad ConservativeBound(const DrawRectangle& rectangle)
    {
        return {
            floorf(rectangle._topLeft.X - rectangle._width / 2),
            floorf(rectangle._topLeft.Y - rectangle._width / 2),
            ceilf(rectangle._bottomRight.X + rectangle._width / 2),
            ceilf(rectangle._bottomRight.Y + rectangle._width / 2)
        };
    }

    static Quad ConservativeBound(const FillCircle& circle)
    {
        return {
            floorf(circle._center.X - circle._radius),
            floorf(circle._center.Y - circle._radius),
            ceilf(circle._center.X + circle._radius),
            ceilf(circle._center.Y + circle._radius)
        };
    }

    static Quad ConservativeBound(const FillRectangle& rectangle)
    {
        return {
            floorf(rectangle._topLeft.X),
            floorf(rectangle._topLeft.Y),
            ceilf(rectangle._bottomRight.X),
            ceilf(rectangle._bottomRight.Y)
        };
    }

    static bool Intersects(const Quad& quad, const DrawLine& line)
    {
        // Define the shape of the quad as 4 points.
        Vec2 shape1[4] =
        {
            quad.topLeft,
            { quad.bottomRight.X, quad.topLeft.Y },
            { quad.topLeft.X, quad.bottomRight.Y },
            quad.bottomRight,
        };
        // Compute line axis.
        Vec2 lineDir = { line._p2.X - line._p1.X, line._p2.Y - line._p1.Y };
        float length = sqrtf(lineDir.X * lineDir.X + lineDir.Y * lineDir.Y);
        Vec2 lineDirN = { lineDir.X / length, lineDir.Y / length };
        Vec2 lineNorN = { -lineDirN.Y, lineDirN.X };
        // Define a shape surrounding the line.
        Vec2 shape2[4] =
        {
            { line._p1.X + line._width * (-lineDirN.X - lineNorN.X), line._p1.Y + line._width * (-lineDirN.Y - lineNorN.Y) },
            { line._p1.X + line._width * (-lineDirN.X + lineNorN.X), line._p1.Y + line._width * (-lineDirN.Y + lineNorN.Y) },
            { line._p2.X + line._width * (lineDirN.X - lineNorN.X), line._p2.Y + line._width * (lineDirN.Y - lineNorN.Y) },
            { line._p2.X + line._width * (lineDirN.X + lineNorN.X), line._p2.Y + line._width * (lineDirN.Y + lineNorN.Y) },
        };
        // Separating axis theorem.
        Vec2 allAxis[4] =
        {
            { 1, 0 },
            { 0, 1 },
            lineDirN,
            lineNorN
        };
        for (const Vec2& axis : allAxis)
        {
            float min1 = std::numeric_limits<float>::infinity();
            float max1 = -std::numeric_limits<float>::infinity();
            for (const Vec2& p : shape1)
            {
                float projected = p.X * axis.X + p.Y * axis.Y;
                min1 = Min(min1, projected);
                max1 = Max(max1, projected);
            }
            float min2 = std::numeric_limits<float>::infinity();
            float max2 = -std::numeric_limits<float>::infinity();
            for (const Vec2& p : shape2)
            {
                float projected = p.X * axis.X + p.Y * axis.Y;
                min2 = Min(min2, projected);
                max2 = Max(max2, projected);
            }
            if (min1 > max2 || max1 < min2)
            {
                return false;
            }
        }
        return true;
    }

    static bool Intersects(const Quad& q1, const Quad& q2)
    {
        return
            q1.topLeft.X <= q2.bottomRight.X &&
            q1.topLeft.Y <= q2.bottomRight.Y &&
            q1.bottomRight.X >= q2.topLeft.X &&
            q1.bottomRight.Y >= q2.topLeft.Y;
    }

    struct Slicer
    {
        void Slice()
        {
            // Terminate if the rectangle has zero area. 
            if (region.w <= 0 || region.h <= 0)
            {
                return;
            }
            // Terminate if there is nothing to process.
            if (contents.size() <= 0)
            {
                // NOTE: We're currently filling with purple to show the culling efficiency.
                for (uint32_t y = 0; y < region.h; ++y)
                {
                    void* pRaster = reinterpret_cast<uint8_t*>(bitmap.pixels) + sizeof(uint32_t) * region.x + bitmap.stride * (region.y + y);
                    for (uint32_t x = 0; x < region.w; ++x)
                    {
                        void *pPixel = (uint8_t*)pRaster + sizeof(PixelBGRA32) * x;
                        PixelBGRA32 *tPixel = reinterpret_cast<PixelBGRA32*>(pPixel);
                        tPixel->b = 255;
                        tPixel->g = 0;
                        tPixel->r = 255;
                        tPixel->a = 255;
                    }
                }
                return;
            }
            if (region.w <= 8 && region.h <= 8)
            {
                RenderTo_Baseline(bitmap, region, contents);
                return;
            }
            // Determine the major axis.
            if (region.w >= region.h)
            {
                // Bisect on X.
                Slicer(BuildSubcontext({ region.x, region.y, region.w / 2, region.h })).Slice();
                Slicer(BuildSubcontext({ region.x + region.w / 2, region.y, region.w - region.w / 2, region.h })).Slice();
            }
            else
            {
                // Bisect on Y.
                Slicer(BuildSubcontext({ region.x, region.y, region.w, region.h / 2 })).Slice();
                Slicer(BuildSubcontext({ region.x, region.y + region.h / 2, region.w, region.h - region.h / 2 })).Slice();
            }
        }
        Slicer BuildSubcontext(const Rectangle& subregion)
        {
            Quad subregionQ = { subregion.x, subregion.y, subregion.x + subregion.w, subregion.y + subregion.h };
            std::vector<const DrawPrimitive*> newthings;
            for (const DrawPrimitive* check : contents)
            {
                switch (check->primitive)
                {
                case PrimitiveType::DRAW_CIRCLE:
                    if (!Intersects(subregionQ, ConservativeBound(check->drawCircle))) continue;
                    break;
                case PrimitiveType::DRAW_LINE:
                    // We're trying out the SAT version of the bounds check here.
                    //if (!Intersects(subregionQ, ConservativeBound(check->drawLine))) continue;
                    if (!Intersects(subregionQ, check->drawLine)) continue;
                    break;
                case PrimitiveType::DRAW_RECTANGLE:
                    if (!Intersects(subregionQ, ConservativeBound(check->drawRectangle))) continue;
                    break;
                case PrimitiveType::FILL_CIRCLE:
                    if (!Intersects(subregionQ, ConservativeBound(check->fillCircle))) continue;
                    break;
                case PrimitiveType::FILL_RECTANGLE:
                    if (!Intersects(subregionQ, ConservativeBound(check->fillRectangle))) continue;
                    break;
                default:
                    continue;
                }
                newthings.push_back(check);
            }
            return Slicer { bitmap, subregion, newthings };
        }
        Bitmap bitmap;
        Rectangle region;
        std::vector<const DrawPrimitive*> contents;
    };

    void RenderTo_Fast(void* pixels, uint32_t width, uint32_t height, uint32_t stride, const std::vector<const DrawPrimitive*> primitives)
    {
        Slicer slicer;
        slicer.bitmap = Bitmap { pixels, width, height, stride };
        slicer.region = Rectangle { 0, 0, width, height };
        slicer.contents = primitives;
        slicer.Slice();
    }
}