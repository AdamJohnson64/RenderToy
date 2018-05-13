////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Concurrent;

namespace RenderToy.Utility
{
    public class MementoServer
    {
        public T Get<T>(object token, Func<T> build)
        {
            return (T)GetBase(token, () => (T)build());
        }
        object GetBase(object token, Func<object> build)
        {
            object result;
            if (Data.TryGetValue(token, out result)) return result;
            return Data[token] = result = build();
        }
        public ConcurrentDictionary<object, object> Data = new ConcurrentDictionary<object, object>();
    }
}