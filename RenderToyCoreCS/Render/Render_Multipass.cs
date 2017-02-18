////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace RenderToy
{
    /// <summary>
    /// Multipass interface for multipass render modes.
    /// </summary>
    abstract class MultiPass
    {
        public static MultiPass Create(RenderCall fillwith)
        {
            if (!fillwith.IsMultipass)
            {
                return new SinglePassAdaptor(fillwith);
            }
            else
            {
                return new MultiPassAdaptor(fillwith);
            }
        }
        public abstract void SetScene(Scene scene);
        public abstract void SetCamera(Matrix3D mvp);
        public abstract void SetTarget(int width, int height);
        public abstract void Start();
        public abstract bool CopyTo(IntPtr buffer_ptr, int buffer_width, int buffer_height, int buffer_stride);
        /// <summary>
        /// Degenerate multipass handler that performs all work in a single pass.
        /// This model will be used for renderers that converge quickly or don't
        /// require any fancy sampling (i.e. the raycase, TBN or raytrace).
        /// </summary>
        class SinglePassAdaptor : MultiPass
        {
            public SinglePassAdaptor(RenderCall fillwith) { this.fillwith = fillwith; }
            public override string ToString() { return RenderCall.GetDisplayNameFull(fillwith.MethodInfo.Name); }
            public override void SetScene(Scene scene) { this.scene = scene; }
            public override void SetCamera(Matrix3D mvp) { this.mvp = mvp; }
            public override void SetTarget(int width, int height) { }
            public override void Start() { }
            public override bool CopyTo(IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
            {
                fillwith.Action(scene, mvp, bitmap_ptr, render_width, render_height, bitmap_stride, null);
                return false;
            }
            RenderCall fillwith;
            Scene scene;
            Matrix3D mvp;
        }
        class MultiPassAdaptor : MultiPass
        {
            public MultiPassAdaptor(RenderCall fillwith) { this.fillwith = fillwith; }
            public override string ToString() { return RenderCall.GetDisplayNameFull(fillwith.MethodInfo.Name); }
            public override void SetScene(Scene scene)
            {
                if (this.scene != scene)
                {
                    this.scene = scene;
                    PassDirty();
                }
            }
            public override void SetCamera(Matrix3D mvp)
            {
                if (this.mvp != mvp)
                {
                    this.mvp = mvp;
                    PassDirty();
                }
            }
            public override void SetTarget(int width, int height)
            {
                if (this.width != width)
                {
                    this.width = width;
                    PassDirty();
                }
                if (this.height != height)
                {
                    this.height = height;
                    PassDirty();
                };
            }
            public override void Start() { }
            public override bool CopyTo(IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
            {
                PassRun();
                GCHandle handle_accumulator = GCHandle.Alloc(accumulator, GCHandleType.Pinned);
                #if !WINDOWS_UWP
                // TODO(fixme): AMP doesn't have a tonemapper yet.
                RenderToyCLI.ToneMap(handle_accumulator.AddrOfPinnedObject(), sizeof(float) * 4 * width, bitmap_ptr, width, height, bitmap_stride, 1.0f / pass);
                #endif
                handle_accumulator.Free();
                return pass < (hemisamples.Length / SAMPLES_PER_PASS);
            }
            void PassDirty()
            {
                accumulator = null;
                pass = 0;
            }
            void PassRun()
            {
                if (accumulator == null)
                {
                    accumulator = new byte[sizeof(float) * 4 * width * height];
                    pass = 0;
                }
                var overrides = new Dictionary<string, object>();
                overrides[RenderCall.HEMISAMPLES] = hemisamples.Skip(SAMPLES_PER_PASS * pass).Take(SAMPLES_PER_PASS).ToArray();
                ++pass;
                fillwith.Action(scene, mvp, accumulator, width, height, sizeof(float) * 4 * width, overrides);
            }
            RenderCall fillwith;
            Scene scene;
            Matrix3D mvp;
            int width;
            int height;
            byte[] accumulator;
            int pass = 0;
            static Vector3D[] hemisamples = MathHelp.HemiHaltonCosineBias(65536).ToArray();
            const int SAMPLES_PER_PASS = 64;
        }
    }
}