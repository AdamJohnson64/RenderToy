////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph;
using System.Collections.Generic;

namespace RenderToy.ModelFormat
{
    public static class LoaderModel
    {
        public static IEnumerable<INode> LoadFromPath(string path)
        {
            if (path.ToUpperInvariant().EndsWith(".BPT"))
            {
                return LoaderBPT.LoadFromPath(path);
            }
            else if (path.ToUpperInvariant().EndsWith(".OBJ"))
            {
                return LoaderOBJ.LoadFromPath(path);
            }
            else if (path.ToUpperInvariant().EndsWith(".PLY"))
            {
                return LoaderPLY.LoadFromPath(path);
            }
            else
            {
                return null;
            }
        }
    }
}