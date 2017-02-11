////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;

namespace RenderToy
{
    [DebuggerDisplay("{MethodInfo.Name}")]
    public class RenderCall
    {
        public RenderCall(MethodInfo methodinfo, VFillFunction action)
        {
            this.MethodInfo = methodinfo;
            this.Action = action;
        }
        public delegate void FillFunction(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
        public delegate void VFillFunction(Scene scene, Matrix3D mvp, object bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
        public readonly MethodInfo MethodInfo;
        public readonly VFillFunction Action;
        public void Fill(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            Action(scene, mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static string GetDisplayNameBare(string methodname)
        {
            var chop_mode = new[] { "AMP", "CPU", "CUDA", "D3D9" }
                .Select(x => new { Chop = methodname.IndexOf(x), Name = x })
                .Where(x => x.Chop != -1).OrderBy(x => x.Chop);
            var chop_prec = new[] { "F32", "F64" }
                .Select(x => new { Chop = methodname.IndexOf(x), Name = x })
                .Where(x => x.Chop != -1).OrderBy(x => x.Chop);
            var chop_any = chop_mode.Concat(chop_prec).OrderBy(x => x.Chop);
            // Return the formatted string.
            return methodname.Substring(0, chop_any.FirstOrDefault().Chop);
        }
        public static string GetDisplayNameFull(string methodname)
        {
            var chop_mode = new[] { "AMP", "CPU", "CUDA", "D3D9" }
                .Select(x => new { Chop = methodname.IndexOf(x), Name = x })
                .Where(x => x.Chop != -1).OrderBy(x => x.Chop);
            var chop_prec = new[] { "F32", "F64" }
                .Select(x => new { Chop = methodname.IndexOf(x), Name = x })
                .Where(x => x.Chop != -1).OrderBy(x => x.Chop);
            var chop_any = chop_mode.Concat(chop_prec).OrderBy(x => x.Chop);
            // Return the formatted string.
            var name = methodname.Substring(0, chop_any.FirstOrDefault().Chop);
            var categories = chop_any.Select(x => x.Name);
            var category = categories.Count() == 0 ? "" : " (" + string.Join("/", categories) + ")";
            return name + category;
        }
        public static IEnumerable<RenderCall> Generate(IEnumerable<Type> types)
        {
            var rendercalls = new List<RenderCall>();
            foreach (var method in types.SelectMany(x => x.GetMethods()))
            {
                if (((method.Attributes & MethodAttributes.Public) == MethodAttributes.Public) &&
                    ((method.Attributes & MethodAttributes.Static) == MethodAttributes.Static))
                {
                    bool isAMP = method.Name.Contains("AMP");
                    bool isCPU = method.Name.Contains("CPU");
                    bool isCUDA = method.Name.Contains("CUDA");
                    bool isF32 = method.Name.Contains("F32");
                    bool isF64 = method.Name.Contains("F64");
                    var generateargs = new List<Func<Dictionary<string, object>, Converted>>();
                    var hemisamples = MathHelp.HemiHaltonCosineBias(256).ToArray();
                    foreach (var param in method.GetParameters())
                    {
                        if (param.ParameterType == typeof(Scene) && param.Name == "scene")
                        {
                            generateargs.Add((args) => new ConvertedPrimitive((Scene)args["scene"]));
                        }
                        else if (param.ParameterType == typeof(Matrix3D) && param.Name == "mvp")
                        {
                            generateargs.Add((args) => new ConvertedPrimitive((Matrix3D)args["mvp"]));
                        }
                        else if (param.ParameterType == typeof(Matrix3D) && param.Name == "inverse_mvp")
                        {
                            generateargs.Add((args) => new ConvertedPrimitive(MathHelp.Invert((Matrix3D)args["mvp"])));
                        }
                        else if (param.ParameterType == typeof(byte[]) && param.Name == "scene")
                        {
                            if (isF32) generateargs.Add((args) => new ConvertedPrimitive(SceneFormatter.CreateFlatMemoryF32((Scene)args["scene"])));
                            if (isF64) generateargs.Add((args) => new ConvertedPrimitive(SceneFormatter.CreateFlatMemoryF64((Scene)args["scene"])));
                        }
                        else if (param.ParameterType == typeof(byte[]) && param.Name == "mvp")
                        {
                            if (isF32) generateargs.Add((args) => new ConvertedPrimitive(SceneFormatter.CreateFlatMemoryF32((Matrix3D)args["mvp"])));
                            if (isF64) generateargs.Add((args) => new ConvertedPrimitive(SceneFormatter.CreateFlatMemoryF64((Matrix3D)args["mvp"])));
                        }
                        else if (param.ParameterType == typeof(byte[]) && param.Name == "inverse_mvp")
                        {
                            if (isF32) generateargs.Add((args) => new ConvertedPrimitive(SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert((Matrix3D)args["mvp"]))));
                            if (isF64) generateargs.Add((args) => new ConvertedPrimitive(SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert((Matrix3D)args["mvp"]))));
                        }
                        else if (param.ParameterType == typeof(byte[]) && param.Name == "bitmap_ptr")
                        {
                            generateargs.Add((args) => new ConvertedPrimitive(args["bitmap_ptr"]));
                        }
                        else if (param.ParameterType == typeof(IntPtr) && param.Name == "bitmap_ptr")
                        {
                            generateargs.Add(
                                (args) => {
                                    var arg = args["bitmap_ptr"];
                                    if (arg is IntPtr) return new ConvertedPrimitive(arg);
                                    else if (arg is byte[]) return new ConvertedGCHandlePinnedIntPtr(arg);
                                    else throw new Exception("Unable to convert type '" + arg + "'.");
                                }
                            );
                        }
                        else if (param.ParameterType == typeof(int) && param.Name == "bitmap_width")
                        {
                            generateargs.Add((args) => new ConvertedPrimitive(args["bitmap_width"]));
                        }
                        else if (param.ParameterType == typeof(int) && param.Name == "bitmap_height")
                        {
                            generateargs.Add((args) => new ConvertedPrimitive(args["bitmap_height"]));
                        }
                        else if (param.ParameterType == typeof(int) && param.Name == "bitmap_stride")
                        {
                            generateargs.Add((args) => new ConvertedPrimitive(args["bitmap_stride"]));
                        }
                        else if (param.ParameterType == typeof(int) && param.Name == "superx")
                        {
                            generateargs.Add((args) => new ConvertedPrimitive(1));
                        }
                        else if (param.ParameterType == typeof(int) && param.Name == "supery")
                        {
                            generateargs.Add((args) => new ConvertedPrimitive(1));
                        }
                        else if (param.ParameterType == typeof(int) && param.Name == "hemisample_count")
                        {
                            generateargs.Add((args) => new ConvertedPrimitive(hemisamples.Length));
                        }
                        else if (param.ParameterType == typeof(byte[]) && param.Name == "hemisamples")
                        {
                            if (isF32) generateargs.Add((args) => new ConvertedPrimitive(SceneFormatter.CreateFlatMemoryF32(hemisamples)));
                            if (isF64) generateargs.Add((args) => new ConvertedPrimitive(SceneFormatter.CreateFlatMemoryF64(hemisamples)));
                        }
                        else
                        {
                            throw new Exception("Unrecognized argument '" + param.Name + "' in render provider '" + method.DeclaringType.Name + "." + method.Name + "'.");
                        }
                    }
                    if (generateargs.Count == 0) continue;
                    VFillFunction action = (scene, mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride) => {
                        var arguments = new Dictionary<string, object>();
                        arguments["scene"] = scene;
                        arguments["mvp"] = mvp;
                        arguments["bitmap_ptr"] = bitmap_ptr;
                        arguments["bitmap_width"] = bitmap_width;
                        arguments["bitmap_height"] = bitmap_height;
                        arguments["bitmap_stride"] = bitmap_stride;
                        var converters = generateargs.Select(x => x(arguments)).ToArray();
                        var parameters = converters.Select(x => x.Value).ToArray();
                        method.Invoke(null, parameters);
                    };
                    rendercalls.Add(new RenderCall(method, action));
                }
            }
            return rendercalls;
        }
        abstract class Converted
        {
            public abstract object Value { get; }
        }
        class ConvertedPrimitive : Converted
        {
            public ConvertedPrimitive(object value)
            {
                Value = value;
            }
            public override object Value { get; }
        }
        class ConvertedGCHandlePinnedIntPtr : Converted
        {
            public ConvertedGCHandlePinnedIntPtr(object value)
            {
                handle = GCHandle.Alloc(value, GCHandleType.Pinned);
            }
            ~ConvertedGCHandlePinnedIntPtr()
            {
                handle.Free();
            }
            public override object Value
            {
                get
                {
                    return handle.AddrOfPinnedObject();
                }
            }
            GCHandle handle;
        }
    }
}