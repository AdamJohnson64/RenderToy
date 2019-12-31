using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using RenderToy.Transforms;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace RenderToy.ModelFormat
{
    public class LoaderBPT
    {
        public static async Task<INode> LoadFromPathAsync(string path)
        {
            var loadednodes = new List<INode>();
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
                    loadednodes.Add(new Node("Patch " + i, new TransformMatrix(Matrix3D.Identity), new BezierPatch(hull), StockMaterials.White, StockMaterials.PlasticWhite));
                }
            }
            var root = new Node(path, new TransformMatrix(Matrix3D.Identity), null, StockMaterials.Black, null);
            root.children.AddRange(loadednodes);
            return root;
        }
    }
}