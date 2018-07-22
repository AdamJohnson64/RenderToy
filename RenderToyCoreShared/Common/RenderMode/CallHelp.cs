////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

// Render calls have many constructions in CLI/CX code and many parameter types vary depending on the environment.
//
// e.g.
//   In CLI the byte[] is usually locked and marshaled as System::IntPtr.
//   In CX the byte[] is marshaled as Platform::Array<unsigned char>^.
//
// Handling the large number of signatures will quickly become unwieldy.
//
// This helper will automatically collect and convert parameters to match the signature of a renderer.
// Overridable defaults will be provided for arguments outside the normal FillFunction signature.

using RenderToy.Math;
using RenderToy.SceneGraph;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;

namespace RenderToy.RenderMode
{
    [DebuggerDisplay("{MethodInfo.Name}")]
    public class RenderCall
    {
        #region - Section : Call Generators -
        public static IEnumerable<RenderCall> Generate(IEnumerable<Type> types)
        {
            return types
                .SelectMany(x => x.GetMethods())
                .Select(x => Generate(x))
                .ToArray()
                .Where(x => x != null)
                .ToArray();
        }
        public static RenderCall Generate(MethodInfo method)
        {
            // If the render call isn't public and static then ignore it.
            if (((method.Attributes & MethodAttributes.Public) != MethodAttributes.Public) ||
                ((method.Attributes & MethodAttributes.Static) != MethodAttributes.Static))
            {
                return null;
            }
            bool knowMultipass = false;
            bool isMultipass = false;
            bool isF32 = method.Name.Contains("F32");
            bool isF64 = method.Name.Contains("F64");
            var generateargs = new List<Func<Dictionary<string, object>, Argument>>();
            foreach (var param in method.GetParameters())
            {
                if (param.ParameterType == typeof(IEnumerable<TransformedObject>) && param.Name == SCENE)
                {
                    generateargs.Add((args) => new ArgumentFixed((IEnumerable<TransformedObject>)args[SCENE]));
                }
                else if (param.ParameterType == typeof(Matrix3D) && param.Name == MVP)
                {
                    generateargs.Add((args) => new ArgumentFixed((Matrix3D)args[MVP]));
                }
                else if (param.ParameterType == typeof(Matrix3D) && param.Name == INVERSE_MVP)
                {
                    generateargs.Add((args) => new ArgumentFixed(MathHelp.Invert((Matrix3D)args[MVP])));
                }
                else if (param.ParameterType == typeof(byte[]) && param.Name == SCENE)
                {
                    if (isF32) generateargs.Add((args) => new ArgumentFixed(SceneSerializer.CreateFlatMemoryF32((IEnumerable<TransformedObject>)args[SCENE])));
                    if (isF64) generateargs.Add((args) => new ArgumentFixed(SceneSerializer.CreateFlatMemoryF64((IEnumerable<TransformedObject>)args[SCENE])));
                }
                else if (param.ParameterType == typeof(byte[]) && param.Name == MVP)
                {
                    if (isF32) generateargs.Add((args) => new ArgumentFixed(SceneSerializer.CreateFlatMemoryF32((Matrix3D)args[MVP])));
                    if (isF64) generateargs.Add((args) => new ArgumentFixed(SceneSerializer.CreateFlatMemoryF64((Matrix3D)args[MVP])));
                }
                else if (param.ParameterType == typeof(byte[]) && param.Name == INVERSE_MVP)
                {
                    if (isF32) generateargs.Add((args) => new ArgumentFixed(SceneSerializer.CreateFlatMemoryF32(MathHelp.Invert((Matrix3D)args[MVP]))));
                    if (isF64) generateargs.Add((args) => new ArgumentFixed(SceneSerializer.CreateFlatMemoryF64(MathHelp.Invert((Matrix3D)args[MVP]))));
                }
                else if (param.ParameterType == typeof(byte[]) && param.Name == ACCUMULATOR_PTR)
                {
                    if (knowMultipass) throw new Exception("Multipass type is already determined.");
                    knowMultipass = true;
                    isMultipass = true;
                    generateargs.Add((args) => new ArgumentFixed(args[BUFFER_PTR]));
                }
                else if (param.ParameterType == typeof(IntPtr) && param.Name == ACCUMULATOR_PTR)
                {
                    if (knowMultipass) throw new Exception("Multipass type is already determined.");
                    knowMultipass = true;
                    isMultipass = true;
                    generateargs.Add(
                        (args) => {
                            var arg = args[BUFFER_PTR];
                            if (arg is IntPtr) return new ArgumentFixed(arg);
                            else if (arg is byte[]) return new ArgumentGCHandlePinnedIntPtr(arg);
                            else throw new Exception("Unable to convert type '" + arg + "'.");
                        }
                    );
                }
                else if (param.ParameterType == typeof(byte[]) && param.Name == BITMAP_PTR)
                {
                    if (knowMultipass) throw new Exception("Multipass type is already determined.");
                    knowMultipass = true;
                    isMultipass = false;
                    generateargs.Add((args) => new ArgumentFixed(args[BUFFER_PTR]));
                }
                else if (param.ParameterType == typeof(IntPtr) && param.Name == BITMAP_PTR)
                {
                    if (knowMultipass) throw new Exception("Multipass type is already determined.");
                    knowMultipass = true;
                    isMultipass = false;
                    generateargs.Add(
                        (args) => {
                            var arg = args[BUFFER_PTR];
                            if (arg is IntPtr) return new ArgumentFixed(arg);
                            else if (arg is byte[]) return new ArgumentGCHandlePinnedIntPtr(arg);
                            else throw new Exception("Unable to convert type '" + arg + "'.");
                        }
                    );
                }
                else if (param.ParameterType == typeof(int) && param.Name == RENDER_WIDTH)
                {
                    generateargs.Add((args) => new ArgumentFixed(args[RENDER_WIDTH]));
                }
                else if (param.ParameterType == typeof(int) && param.Name == RENDER_HEIGHT)
                {
                    generateargs.Add((args) => new ArgumentFixed(args[RENDER_HEIGHT]));
                }
                else if (param.ParameterType == typeof(int) && param.Name == BITMAP_STRIDE)
                {
                    generateargs.Add((args) => new ArgumentFixed(args[BUFFER_STRIDE]));
                }
                else if (param.ParameterType == typeof(int) && param.Name == SAMPLE_OFFSET)
                {
                    generateargs.Add((args) => new ArgumentFixed(args.ContainsKey(SAMPLE_OFFSET) ? args[SAMPLE_OFFSET] : 0));
                }
                else if (param.ParameterType == typeof(int) && param.Name == SAMPLE_COUNT)
                {
                    generateargs.Add((args) => new ArgumentFixed(args.ContainsKey(SAMPLE_COUNT) ? args[SAMPLE_COUNT] : 64));
                }
                else if (param.ParameterType == typeof(int) && param.Name == SUPERX)
                {
                    generateargs.Add((args) => new ArgumentFixed(1));
                }
                else if (param.ParameterType == typeof(int) && param.Name == SUPERY)
                {
                    generateargs.Add((args) => new ArgumentFixed(1));
                }
                else
                {
                    return null;
                    throw new Exception("Unrecognized argument '" + param.Name + "' in render provider '" + method.DeclaringType.Name + "." + method.Name + "'.");
                }
            }
            // If we didn't find any arguments then ignore this method.
            if (generateargs.Count == 0)
            {
                return null;
            }
            // If we couldn't determine the buffer type (multipass) then ignore this method.
            if (!knowMultipass)
            {
                return null;
            }
            // Otherwise generate a signature for this method.
            FillFunction action = (scene, mvp, buffer_ptr, render_width, render_height, buffer_stride, overrides) => {
                var arguments = new Dictionary<string, object>();
                arguments[SCENE] = scene;
                arguments[MVP] = mvp;
                arguments[BUFFER_PTR] = buffer_ptr;
                arguments[RENDER_WIDTH] = render_width;
                arguments[RENDER_HEIGHT] = render_height;
                arguments[BUFFER_STRIDE] = buffer_stride;
                if (overrides != null)
                {
                    foreach (var inject in overrides)
                    {
                        arguments[inject.Key] = inject.Value;
                    }
                }
                var converters = generateargs.Select(x => x(arguments)).ToArray();
                var parameters = converters.Select(x => x.Value).ToArray();
                method.Invoke(null, parameters);
            };
            return new RenderCall(method, action, isMultipass);
        }
        #endregion
        #region - Section : Call Naming -
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
        #endregion
        #region - Section : Argument Converters -
        /// <summary>
        /// Container for a converted parameter.
        /// This object manages the lifetime of a parameter value.
        /// </summary>
        abstract class Argument
        {
            /// <summary>
            /// The value of the parameter to be passed in the call.
            /// </summary>
            public abstract object Value { get; }
        }
        /// <summary>
        /// Empty converter returning a fixed value.
        /// </summary>
        class ArgumentFixed : Argument
        {
            public ArgumentFixed(object value)
            {
                Value = value;
            }
            public override object Value { get; }
        }
        /// <summary>
        /// Convert a blittable type to IntPtr via GCHandle pinning.
        /// </summary>
        class ArgumentGCHandlePinnedIntPtr : Argument
        {
            public ArgumentGCHandlePinnedIntPtr(object value)
            {
                handle = GCHandle.Alloc(value, GCHandleType.Pinned);
            }
            ~ArgumentGCHandlePinnedIntPtr()
            {
                handle.Free();
            }
            public override object Value
            {
                get { return handle.AddrOfPinnedObject(); }
            }
            GCHandle handle;
        }
        #endregion
        #region - Section : Argument List -
        public static readonly string SCENE = "scene";
        public static readonly string MVP = "mvp";
        public static readonly string INVERSE_MVP = "inverse_mvp";
        public static readonly string RENDER_WIDTH = "render_width";
        public static readonly string RENDER_HEIGHT = "render_height";
        public static readonly string ACCUMULATOR_PTR = "accumulator_ptr";
        public static readonly string ACCUMULATOR_STRIDE = "accumulator_stride";
        public static readonly string BITMAP_PTR = "bitmap_ptr";
        public static readonly string BITMAP_STRIDE = "bitmap_stride";
        public static readonly string BUFFER_PTR = "buffer_ptr";
        public static readonly string BUFFER_STRIDE = "buffer_stride";
        public static readonly string SAMPLE_OFFSET = "sample_offset";
        public static readonly string SAMPLE_COUNT = "sample_count";
        public static readonly string SUPERX = "superx";
        public static readonly string SUPERY = "supery";
        #endregion
        #region - Section : Private Construction & Data -
        RenderCall(MethodInfo methodinfo, FillFunction action, bool ismultipass)
        {
            MethodInfo = methodinfo;
            Action = action;
            IsMultipass = ismultipass;
        }
        public delegate void FillFunction(IEnumerable<TransformedObject> scene, Matrix3D mvp, object buffer_ptr, int render_width, int render_height, int buffer_stride, Dictionary<string, object> overrides);
        public readonly MethodInfo MethodInfo;
        public readonly FillFunction Action;
        public readonly bool IsMultipass;
        #endregion
    }
}