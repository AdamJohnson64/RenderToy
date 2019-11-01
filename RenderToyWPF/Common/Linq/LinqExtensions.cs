using System;
using System.Collections.Generic;

namespace RenderToy.Linq
{
    public static class LinqExtensions
    {
        public static long MaxOrDefault(this IEnumerable<long> data)
        {
            long? found = null;
            foreach (var t in data)
            {
                found = System.Math.Max(found ?? t, t);
            }
            return found ?? 0;
        }
        public static long MinOrDefault(this IEnumerable<long> data)
        {
            long? found = null;
            foreach (var t in data)
            {
                found = System.Math.Min(found ?? t, t);
            }
            return found ?? 0;
        }
        public static IEnumerable<int> GenerateIntegerSequence(int count)
        {
            for (int i = 0; i < count; ++i) yield return i;
        }
        public static IEnumerable<Tuple<T, T, T>> Split3<T>(IEnumerable<T> indices)
        {
            var iter = indices.GetEnumerator();
            while (iter.MoveNext())
            {
                var e0 = iter.Current;
                if (!iter.MoveNext()) throw new Exception();
                var e1 = iter.Current;
                if (!iter.MoveNext()) throw new Exception();
                var e2 = iter.Current;
                yield return new Tuple<T, T, T>(e0, e1, e2);
            }
        }
    }
}