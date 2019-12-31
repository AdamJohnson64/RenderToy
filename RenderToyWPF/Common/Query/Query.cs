using System;
using System.Collections.Concurrent;

namespace RenderToy.QueryEngine
{
    /// <summary>
    /// Delegate used to signal a change in the state of a query result.
    /// </summary>
    public delegate void QueryChangedHandler();
    /// <summary>
    /// Abstract query storing a typed datum and with the ability to register for change.
    /// </summary>
    /// <typeparam name="TResult">The data type stored by this query.</typeparam>
    public interface IQuery<TResult>
    {
        TResult Result { get; }
        void AddListener(QueryChangedHandler handler);
    }
    /// <summary>
    /// Base untyped query.
    /// </summary>
    public partial class Query
    {
        #region - Section : Query Request -
        /// <summary>
        /// Create an immediate query which does not defer execution.
        /// </summary>
        /// <typeparam name="TResult">The data type of this query.</typeparam>
        /// <param name="result">The data to be returned from the query.</param>
        /// <returns>A new query object.</returns>
        public static IQuery<TResult> Create<TResult>(TResult result)
            where TResult : class
        {
            return (IQuery<TResult>)GetExisting(new QueryImmediate<TResult>(result));
        }
        /// <summary>
        /// Create a deferred query which will be populated asynchronously with the result of a function.
        /// </summary>
        /// <typeparam name="TResult">The data type of this query.</typeparam>
        /// <param name="func">A function which will asynchronously execute to populate this query.</param>
        /// <returns>A new query object.</returns>
        public static IQuery<TResult> Create<TResult>(Func<TResult> func)
        {
            return (IQuery<TResult>)GetExisting((Query)QueryDeferred<TResult>.CreateInner(func));
        }
        /// <summary>
        /// Create a deferred query which will be populated asynchronously with the result of a function (called with the result of another query).
        /// </summary>
        /// <typeparam name="T1">The type of the parameter to the query.</typeparam>
        /// <typeparam name="TResult">The data type of this query.</typeparam>
        /// <param name="func">A function which will asynchronously execute to populate this query.</param>
        /// <param name="arg1">A query argument which will be passed to the function when ready.</param>
        /// <returns>A new query object.</returns>
        public static IQuery<TResult> Create<T1, TResult>(Func<T1, TResult> func, IQuery<T1> arg1)
        {
            var query = (IQuery<TResult>)GetExisting((Query)QueryDeferred<TResult>.CreateInner<T1>(func, arg1));
            arg1.AddListener(((Query)query).HandleQueryChanged);
            return query;
        }
        #endregion
        #region - Section : Subcriber -
        /// <summary>
        /// Register a delegate to handle changes in this query result.
        /// </summary>
        /// <param name="handler">The delegate to be called upon change.</param>
        public void AddListener(QueryChangedHandler handler)
        {
            Subscribers.Add(new WeakReference<QueryChangedHandler>(handler));
            handler();
        }
        /// <summary>
        /// Fire a change event and call all subscriber delegates.
        /// </summary>
        public void HandleQueryChanged()
        {
            foreach (var listener in Subscribers)
            {
                QueryChangedHandler handler;
                if (!listener.TryGetTarget(out handler))
                {
                    var remove = listener;
                    Subscribers.TryTake(out remove);
                    continue;
                }
                handler();
            }
        }
        private ConcurrentBag<WeakReference<QueryChangedHandler>> Subscribers = new ConcurrentBag<WeakReference<QueryChangedHandler>>();
        #endregion
        #region - Section : Idempotence -
        /// <summary>
        /// Attempt to match queries such that they can be reused.
        /// This function will take a query and attempt to find one already defined.
        /// </summary>
        /// <param name="query">The query to locate.</param>
        /// <returns>An existing query or a new one.</returns>
        static Query GetExisting(Query query)
        {
            var hash = query.GetHashCode();
            ConcurrentBag<WeakReference<Query>> queries;
            if (!Queries.TryGetValue(hash, out queries))
            {
                Queries[hash] = new ConcurrentBag<WeakReference<Query>>();
                Queries[hash].Add(new WeakReference<Query>(query));
                return query;
            }
            foreach (var existingquery in queries)
            {
                Query findquery;
                if (!existingquery.TryGetTarget(out findquery))
                {
                    var remove = existingquery;
                    queries.TryTake(out remove);
                    continue;
                }
                if (query.Equals(findquery))
                {
                    return findquery;
                }
            }
            Queries[hash].Add(new WeakReference<Query>(query));
            return query;
        }
        static ConcurrentDictionary<int, ConcurrentBag<WeakReference<Query>>> Queries = new ConcurrentDictionary<int, ConcurrentBag<WeakReference<Query>>>();
        #endregion
    }
}