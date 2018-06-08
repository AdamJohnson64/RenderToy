////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Concurrent;
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
        /// Evict all cached data constructed with a specific token.
        /// </summary>
        /// <param name="token">The token identity of objects to be evicted.</param>
        public void EvictByToken(object token)
        {
            var removethese = Cache.Where(i => i.Key.Token.Target == token).ToArray();
            foreach (var key in removethese)
            {
                Cache.TryRemove(key.Key, out object value);
            }
            if (removethese.Length > 0)
            {
                Debug.WriteLine("Evicting cached data; " + Cache.Count + " entries remaining.");
            }
        }
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
            return (T)GetBase(owner, token, () => (T)build());
        }
        /// <summary>
        /// Static initialization to prepare automatic eviction timers.
        /// </summary>
        public MementoServer()
        {
            cleanup = new Timer((o) =>
            {
                var removethese = Cache.Where(i => !i.Key.Owner.IsAlive || !i.Key.Token.IsAlive || DateTime.Now.Subtract(i.Key.LastAccess).TotalMinutes > 10).ToArray();
                foreach (var key in removethese)
                {
                    Cache.TryRemove(key.Key, out object value);
                }
                if (removethese.Length > 0)
                {
                    Debug.WriteLine("Cleaning up cached data; " + Cache.Count + " entries remaining.");
                }
            }, null, 0, 30000);
        }
        object GetBase(object owner, object token, Func<object> build)
        {
            return GetBase(new Key(owner, token), build);
        }
        object GetBase(Key key, Func<object> build)
        {
            object result;
            if (Cache.TryGetValue(key, out result))
            {
                key.LastAccess = DateTime.Now;
                return result;
            }
            return Cache[key] = result = build();
        }
        ConcurrentDictionary<Key, object> Cache = new ConcurrentDictionary<Key, object>();
        Timer cleanup;
        /// <summary>
        /// Keys for the cache dictionary comprising owner and token.
        /// </summary>
        class Key
        {
            public Key(object owner, object token)
            {
                Hash = owner.GetHashCode();
                LastAccess = DateTime.Now;
                Owner = new WeakReference(owner);
                Token = new WeakReference(token);
            }
            public override bool Equals(object obj)
            {
                var key = obj as Key;
                if (key == null) return false;
                return Hash == key.Hash && Owner.Target == key.Owner.Target && Token.Target == key.Token.Target;
            }
            public override int GetHashCode()
            {
                return Hash;
            }
            public readonly int Hash;
            public DateTime LastAccess;
            public readonly WeakReference Owner;
            public readonly WeakReference Token;
        }
    }
}