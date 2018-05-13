// The pipeline is our ticket to fully configurable render process from source
// data to display.
//
// By modelling the render pipeline as functional components we can combine
// them in many ways to define (e.g.):
//     - Empty pipelines.
//     - Degenerate pipelines.
//     - Simplified geometry.
//     - Simplified interpolators.
//
// We also provide a way to visualize these pipelines as a means to inspect how
// the code is producing output and also to facilitate interrogation and debug.
// Further, we may use this decomposition of the rendering pipeline to enable
// parallelism in the future.
//
// A well defined pipeline must, at a minimum, consist of a pixel output.

namespace RenderToy.PipelineModel
{
    /// <summary>
    /// Representation of a single colored pixel.
    /// 
    /// Coordinates are limited to 16-bit precision (to 65535 pixels).
    /// Colors are represented as 32bpp BGRA DWORDs.
    /// </summary>
    public struct PixelBgra32
    {
        public ushort X, Y;
        public uint Color;
    }
}