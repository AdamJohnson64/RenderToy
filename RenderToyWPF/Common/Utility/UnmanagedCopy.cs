using System;
using System.Runtime.InteropServices;

namespace RenderToy.Utility
{
    class UnmanagedCopy
    {
        public static UnmanagedCopy Create<TYPE>(TYPE[] data)
        {
            var result = new UnmanagedCopy();
            var pin = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                var hsrc = Marshal.UnsafeAddrOfPinnedArrayElement(data, 0);
                var size = Marshal.SizeOf(typeof(TYPE)) * data.Length;
                result.Marshaled = Marshal.AllocHGlobal(size);
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
            finally
            {
                pin.Free();
            }
        }
        ~UnmanagedCopy()
        {
            Marshal.FreeHGlobal(Marshaled);
        }
        public static implicit operator IntPtr(UnmanagedCopy marshal)
        {
            return marshal.Marshaled;
        }
        IntPtr Marshaled { get; set; }
    }
}