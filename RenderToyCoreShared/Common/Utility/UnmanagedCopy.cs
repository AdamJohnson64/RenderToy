////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Runtime.InteropServices;

namespace RenderToy.Utility
{
    class UnmanagedCopy
    {
        public static UnmanagedCopy Create<TYPE>(TYPE[] data)
        {
            var result = new UnmanagedCopy();
            result.Pinned = GCHandle.Alloc(data, GCHandleType.Pinned);
            var hsrc = Marshal.UnsafeAddrOfPinnedArrayElement(data, 0);
            var size = Marshal.SizeOf(typeof(TYPE)) * data.Length;
            result.Marshaled = Marshal.AllocCoTaskMem(size);
            unsafe
            {
                byte* src = (byte*)hsrc.ToPointer();
                byte* dst = (byte*)result.Marshaled.ToPointer();
                for (int i = 0; i < size; ++i)
                {
                    dst[i] = src[i];
                }
            }
            return result;
        }
        ~UnmanagedCopy()
        {
            Pinned.Free();
            Marshal.FreeCoTaskMem(Marshaled);
        }
        public static implicit operator IntPtr(UnmanagedCopy marshal)
        {
            return marshal.Marshaled;
        }
        GCHandle Pinned;
        IntPtr Marshaled { get; set; }
    }
}