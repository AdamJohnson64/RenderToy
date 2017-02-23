////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace RenderToy
{
    class MultiPassAsyncAdaptor : IMultiPass
    {
        #region - Section : Threaded Pump -
        public MultiPassAsyncAdaptor(RenderCall fillwith, BitmapReady onbitmapready)
        {
            this.fillwith = fillwith;
            Pump(new WeakReference(this), onbitmapready, ToString());
        }
        static void Pump(WeakReference weakhost, BitmapReady onbitmapready, string name)
        {
            var task = new Task(() =>
            {
                try
                {
                    while (true)
                    {
                        // This crazy looking WeakReference code underneath here allows this object to be garbage collected.
                        // Without it the containing object will hold a reference to 'this'.
                        // Wait for a dirty frame.
                        Func<ManualResetEvent> getlock = () =>
                        {
                            var hostgetlock = (MultiPassAsyncAdaptor)weakhost.Target;
                            if (hostgetlock == null) return null;
                            return hostgetlock.lock_frame;
                        };
                        var hostlock = getlock();
                        if (hostlock == null)
                        {
                            Debug.WriteLine("Render Pump Exited: " + name);
                            return;
                        }
                        hostlock.WaitOne(1000);
                        // Retrieve the renderer and process a frame.
                        Action doupdate = () =>
                        {
                            var host = (MultiPassAsyncAdaptor)weakhost.Target;
                            if (host == null) return;
                            // Lock the descriptor and copy it.
                            host.lock_desc.WaitOne();
                            AccumulateBuffer Next = host.Current;
                            host.lock_desc.ReleaseMutex();
                            // Process the frame.
                            GCHandle handle_accumulator = GCHandle.Alloc(Next.Buffer, GCHandleType.Pinned);
                            try
                            {
                                var overrides = new Dictionary<string, object>();
                                overrides[RenderCall.SAMPLE_OFFSET] = Next.Pass * SAMPLES_PER_PASS;
                                overrides[RenderCall.SAMPLE_COUNT] = SAMPLES_PER_PASS;
                                host.fillwith.Action(Next.Scene, Next.MVP, handle_accumulator.AddrOfPinnedObject(), Next.Width, Next.Height, sizeof(float) * 4 * Next.Width, overrides);
                                Next.Pass++;
                            }
                            finally
                            {
                                handle_accumulator.Free();
                            }
                            // Update the descriptor and signal frame readiness.
                            host.lock_desc.WaitOne();
                            host.Current = Next;
                            host.lock_desc.ReleaseMutex();
                            if (onbitmapready != null) onbitmapready();
                            DONE:
                            // Clear the dirty indicator.
                            if (Next.Pass > 100)
                            {
                                host.lock_frame.Reset();
                            }
                        };
                        doupdate();
                    }
                }
                catch (TaskCanceledException e)
                {
                }
            }, TaskCreationOptions.LongRunning);
            task.Start();
        }
        ManualResetEvent lock_frame = new ManualResetEvent(false);
        Mutex lock_desc = new Mutex();
        #endregion
        public override string ToString() { return RenderCall.GetDisplayNameFull(fillwith.MethodInfo.Name); }
        public void SetScene(Scene scene)
        {
            try
            {
                lock_desc.WaitOne();
                Current.Scene = scene;
                Current.Buffer = new byte[sizeof(float) * 4 * Current.Width * Current.Height];
                Current.Pass = 0;
            }
            finally
            {
                lock_desc.ReleaseMutex();
            }
            InvalidateFrame();
        }
        public void SetCamera(Matrix3D mvp)
        {
            try
            {
                lock_desc.WaitOne();
                if (mvp == Current.MVP) return;
                Current.MVP = mvp;
                Current.Buffer = new byte[sizeof(float) * 4 * Current.Width * Current.Height];
                Current.Pass = 0;
            }
            finally
            {
                lock_desc.ReleaseMutex();
            }
            InvalidateFrame();
        }
        public void SetTarget(int width, int height)
        {
            try
            {
                lock_desc.WaitOne();
                if (width == Current.Width && height == Current.Height) return;
                Current.Width = width;
                Current.Height = height;
                Current.Buffer = new byte[sizeof(float) * 4 * Current.Width * Current.Height];
                Current.Pass = 0;
            }
            finally
            {
                lock_desc.ReleaseMutex();
            }
            InvalidateFrame();
        }
        public void CopyTo(IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            GCHandle handle_accumulator = GCHandle.Alloc(Current.Buffer, GCHandleType.Pinned);
            try
            {
                #if !WINDOWS_UWP
                // TODO(fixme): AMP doesn't have a tonemapper yet.
                RenderToyCLI.ToneMap(handle_accumulator.AddrOfPinnedObject(), sizeof(float) * 4 * Current.Width, bitmap_ptr, Current.Width, Current.Height, bitmap_stride, 1.0f / (Current.Pass * SAMPLES_PER_PASS));
                #endif
            }
            finally
            {
                handle_accumulator.Free();
            }
        }
        void InvalidateFrame()
        {
            lock_frame.Set();
        }
        RenderCall fillwith;
        AccumulateBuffer Current;
        const int SAMPLES_PER_PASS = 4;
    }
}