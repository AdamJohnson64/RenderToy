namespace RenderToy.QueryEngine
{
    public partial class Query
    {
        class QueryImmediate<TResult> : Query, IQuery<TResult>
            where TResult : class
        {
            public QueryImmediate(TResult result)
            {
                Result = result;
            }
            public TResult Result { get; private set; }
            public override int GetHashCode() => Result.GetHashCode();
            public override bool Equals(object obj) => obj is QueryImmediate<TResult> other && Result == other.Result;
        }
    }
}