////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace RenderToy.QueryEngine
{
    public partial class Query
    {
        class QueryDeferred<TResult> : Query, IQuery<TResult>
        {
            internal static IQuery<TResult> CreateInner(Func<TResult> generator)
            {
                var query = new QueryDeferred<TResult>();
                query.Generator = generator;
                Task.Run(() =>
                {
                    query.Result = query.Generator();
                    query.HandleQueryChanged();
                });
                return query;
            }
            internal static IQuery<TResult> CreateInner<T1, TResult>(Func<T1, TResult> generator, IQuery<T1> arg1)
            {
                var query = new QueryDeferred<TResult>();
                query.Generator = () => generator(arg1.Result);
                query.DependsOn.Add((Query)arg1);
                ((Query)arg1).Subscribers.Add(new WeakReference<QueryChangedHandler>(query.HandleQueryChanged));
                Task.Run(() =>
                {
                    query.Result = query.Generator();
                    query.HandleQueryChanged();
                });
                return query;
            }
            public TResult Result { get; private set; }
            internal Func<TResult> Generator { get; private set; }
            internal List<Query> DependsOn = new List<Query>();
            public override int GetHashCode() => Generator.GetHashCode();
            public override bool Equals(object obj)
            {
                if (!(obj is QueryDeferred<TResult> existing)) return false;
                var m1 = Generator.Method;
                var m2 = existing.Generator.Method;
                if (m1 != m2) return false;
                if (!DependsOn.SequenceEqual(existing.DependsOn)) return false;
                return true;
            }
        }
    }
}