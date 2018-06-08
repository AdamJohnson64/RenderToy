////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;
using RenderToy.RenderMode;
using RenderToy.SceneGraph;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace RenderToy.RenderControl
{
    /// <summary>
    /// Degenerate multipass handler that performs all work in a single pass.
    /// This model will be used for renderers that converge quickly or don't
    /// require any fancy sampling (i.e. the raycase, TBN or raytrace).
    /// </summary>
    class SinglePassAsyncAdaptor : IMultiPass
    {
        #region - Section : Threaded Pump -
        public SinglePassAsyncAdaptor(RenderCall fillwith, BitmapReady onbitmapready)
        {
            this.fillwith = fillwith;
            Pump(new WeakReference(this), onbitmapready, ToString());
        }
        static void Pump(WeakReference weakhost, BitmapReady onbitmapready, string name)
        {
            var task = new Task(() =>
            {
                while (true)
                {
                    // This crazy looking WeakReference code underneath here allows this object to be garbage collected.
                    // Without it the containing object will hold a reference to 'this'.
                    // Wait for a dirty frame.
                    Func<ManualResetEvent> getlock = () =>
                    {
                        SinglePassAsyncAdaptor hostgetlock = (SinglePassAsyncAdaptor)weakhost.Target;
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
                        SinglePassAsyncAdaptor host = (SinglePassAsyncAdaptor)weakhost.Target;
                        if (host == null) return;
                        AGAIN:
                        // Lock the descriptor and copy it.
                        host.lock_desc.WaitOne();
                        if (host.Current.Scene == host.Desired.Scene && host.Current.MVP == host.Desired.MVP && host.Current.Width == host.Desired.Width && host.Current.Height == host.Desired.Height)
                        {
                            host.lock_desc.ReleaseMutex();
                            goto DONE;
                        }
                        BitmapBuffer Next = host.Desired;
                        host.lock_desc.ReleaseMutex();
                        // Process the frame.
                        if (Next.Width == 0 || Next.Width == 0) goto DONE;
                        Next.Buffer = new byte[4 * Next.Width * Next.Height];
                        GCHandle handle_buffer = GCHandle.Alloc(Next.Buffer, GCHandleType.Pinned);
                        try
                        {
                            host.fillwith.Action(Next.Scene, Next.MVP, handle_buffer.AddrOfPinnedObject(), Next.Width, Next.Height, 4 * Next.Width, null);
                        }
                        finally
                        {
                            handle_buffer.Free();
                        }
                        // Update the descriptor and signal frame readiness.
                        host.lock_desc.WaitOne();
                        host.Current = Next;
                        if (host.Current.Scene != host.Desired.Scene || host.Current.MVP != host.Desired.MVP || host.Current.Width != host.Desired.Width || host.Current.Height != host.Desired.Height)
                        {
                            host.lock_desc.ReleaseMutex();
                            goto AGAIN;
                        }
                        host.lock_desc.ReleaseMutex();
                        if (onbitmapready != null) onbitmapready();
                        DONE:
                        // Clear the dirty indicator.
                        host.lock_frame.Reset();
                    };
                    doupdate();
                }
            }, TaskCreationOptions.LongRunning);
            task.Start();
        }
        ManualResetEvent lock_frame = new ManualResetEvent(false);
        Mutex lock_desc = new Mutex();
        #endregion
        public override string ToString() { return RenderCall.GetDisplayNameFull(fillwith.MethodInfo.Name); }
        public void SetScene(IScene scene)
        {
            try
            {
                lock_desc.WaitOne();
                if (scene == Desired.Scene) return;
                Desired.Scene = scene;
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
                if (mvp == Desired.MVP) return;
                Desired.MVP = mvp;
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
                if (width == Desired.Width && height == Desired.Height) return;
                Desired.Width = width;
                Desired.Height = height;
            }
            finally
            {
                lock_desc.ReleaseMutex();
            }
            InvalidateFrame();
        }
        public void CopyTo(IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            lock_desc.WaitOne();
            BitmapBuffer frame = Current;
            lock_desc.ReleaseMutex();
            if (frame.Buffer == null || frame.Width != render_width || frame.Height != render_height) return;
            GCHandle handle = GCHandle.Alloc(frame.Buffer, GCHandleType.Pinned);
            try
            {
                for (int y = 0; y < render_height; ++y)
                {
                    unsafe
                    {
                        uint* pIn = (uint*)((byte*)handle.AddrOfPinnedObject() + 4 * render_width * y);
                        uint* pOut = (uint*)((byte*)bitmap_ptr + bitmap_stride * y);
                        for (int x = 0; x < render_width; ++x)
                        {
                            pOut[x] = pIn[x];
                        }
                    }
                }
            }
            finally
            {
                handle.Free();
            }
        }
        void InvalidateFrame()
        {
            lock_frame.Set();
        }
        RenderCall fillwith;
        BitmapBuffer Current;
        BitmapBuffer Desired;
    }
}