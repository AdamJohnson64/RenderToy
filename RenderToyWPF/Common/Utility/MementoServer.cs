////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;

namespace RenderToy.Utility
{
    /// <summary>
    /// The MementoServer is a persistent cache which can store keyed data for an object.
    /// This object does not retain strong references; if objects fall out of scope and are GCed they are removed.
    /// Objects which have not been accessed for over 1 minute are automatically evicted.
    /// </summary>
    public class MementoServer
    {
        public static readonly MementoServer Default = new MementoServer();
        /// <summary>
        /// Retrieve a cached object via its owner and token, or build it as required.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="owner">The owning object of the cached data.</param>
        /// <param name="token">The token identifying this specific record.</param>
        /// <param name="build">A function which will build the data if missing.</param>
        /// <returns></returns>
        public T Get<T>(object owner, object token, Func<T> build)
        {
            return (T)GetBase(owner, token, () => (T)build()).Data;
        }
        /// <summary>
        /// Static initialization to prepare automatic eviction timers.
        /// </summary>
        public MementoServer()
        {
            cleanup = new Timer((o) =>
            {
                lockcache.EnterWriteLock();
                try
                {
                    var removethese = Cache.Where(i => DateTime.Now.Subtract(i.Value.LastAccess).TotalMinutes > 1).ToArray();
                    foreach (var key in removethese)
                    {
                        Cache.Remove(key.Key);
                    }
                    if (removethese.Length > 0)
                    {
                        Debug.WriteLine("Cleaning up cached data; " + Cache.Count + " entries remaining.");
                    }
                }
                finally
                {
                    lockcache.ExitWriteLock();
                }
            }, null, 0, 30000);
        }
        Value GetBase(object owner, object token, Func<object> build)
        {
            return GetBase(new Key(owner, token), build);
        }
        Value GetBase(Key key, Func<object> build)
        {
            Value result;
            lockcache.EnterReadLock();
            try
            {
                if (Cache.TryGetValue(key, out result))
                {
                    return result;
                }
                return Cache[key] = result = new Value(build);
            }
            finally
            {
                lockcache.ExitReadLock();
            }
        }
        ReaderWriterLockSlim lockcache = new ReaderWriterLockSlim(LockRecursionPolicy.SupportsRecursion);
        Dictionary<Key, Value> Cache = new Dictionary<Key, Value>();
        Timer cleanup;
        /// <summary>
        /// Keys for the cache dictionary comprising owner and token.
        /// </summary>
        struct Key
        {
            public Key(object owner, object token)
            {
                Owner = owner;
                Token = token;
            }
            object Owner;
            object Token;
        }
        struct Value
        {
            public Value(Func<object> build)
            {
                LastAccess = DateTime.Now;
                data = build();
            }
            public object Data
            {
                get
                {
                    LastAccess = DateTime.Now;
                    return data;
                }
            }
            internal DateTime LastAccess;
            readonly object data;
        }
    }
}