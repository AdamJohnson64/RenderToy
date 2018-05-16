////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Primitives;
using RenderToy.Utility;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace RenderToy.ModelFormat
{
    class LoaderBPT
    {
        public static IEnumerable<IPrimitive> LoadFromPath(string path)
        {
            using (var stringreader = File.OpenText(path))
            {
                var numpatch = int.Parse(stringreader.ReadLine());
                for (int i = 0; i < numpatch; ++i)
                {
                    var patchsize = stringreader.ReadLine();
                    if (patchsize != "3 3") throw new FileLoadException();
                    Vector3D[] hull = new Vector3D[16];
                    for (int j = 0; j < 16; ++j)
                    {
                        var coord = stringreader.ReadLine().Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries).Select(v => double.Parse(v)).ToArray();
                        if (coord.Length != 3) throw new FileLoadException();
                        hull[j] = new Vector3D(coord[0], coord[1], coord[2]);
                    }
                    yield return new BezierPatch(hull);
                }
            }
        }
    }
}