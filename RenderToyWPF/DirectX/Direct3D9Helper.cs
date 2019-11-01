////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System.Windows.Forms;

namespace RenderToy.DirectX
{
    public static class Direct3D9Helper
    {
        #region - Section : Direct3D Global Objects -
        static readonly Direct3D9Ex d3d = new Direct3D9Ex();
        static readonly Form form = new Form();
        public static readonly Direct3DDevice9Ex device = d3d.CreateDevice(form.Handle);
        #endregion
    }
}