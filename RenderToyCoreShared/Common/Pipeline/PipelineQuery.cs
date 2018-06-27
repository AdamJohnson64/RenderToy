////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;
using RenderToy.Primitives;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace RenderToy.PipelineModel
{
    class ExpressionReplaceCalls : ExpressionVisitor
    {
        protected override Expression VisitMethodCall(MethodCallExpression node)
        {
            var methodinfo = node.Method;
            var membertype = methodinfo.DeclaringType;
            var expressionname = methodinfo.Name;
            foreach (var param in methodinfo.GetParameters())
            {
                expressionname = expressionname + "_" + param.ParameterType.Name;
            }
            var expressionfield = membertype.GetField(expressionname);
            if (expressionfield == null) goto FAIL;
            var expressiontree = expressionfield.GetValue(null) as LambdaExpression;
            if (expressiontree == null) goto FAIL;
            return base.VisitInvocation(Expression.Invoke(expressiontree, node.Arguments));
            FAIL:
            if (!typeof(IQueryable).IsAssignableFrom(node.Method.ReturnType))
            {
                throw new NotSupportedException();
            }
            return base.VisitMethodCall(node);
        }
        protected override Expression VisitInvocation(InvocationExpression node)
        {
            if (!(node.Expression is MemberExpression member)) goto FAIL;
            var membertype = member.Member.DeclaringType;
            var expressionname = member.Member.Name + "Fn";
            var expressionfield = membertype.GetField(expressionname);
            if (expressionfield == null) goto FAIL;
            var expressiontree = expressionfield.GetValue(null) as LambdaExpression;
            if (expressiontree == null) goto FAIL;
            return base.VisitInvocation(Expression.Invoke(expressiontree, node.Arguments));
            FAIL:
            return base.VisitInvocation(node);
        }
    }
    public class TransformStage : IQueryProvider
    {
        public IQueryable CreateQuery(Expression expression)
        {
            throw new NotImplementedException();
        }
        public IQueryable<TElement> CreateQuery<TElement>(Expression expression)
        {
            var newexpression = new ExpressionReplaceCalls().Visit(expression);
            var methodcall = (MethodCallExpression)newexpression;
            var constant = (ConstantExpression)methodcall.Arguments[0];
            var vertexsource = (IEnumerable<Vector3D>)constant.Value;
            var unaryexpression = (UnaryExpression)methodcall.Arguments[1];
            var lambda = (Expression<Func<Vector3D, TElement>>)unaryexpression.Operand;
            var map = lambda.Compile();
            return vertexsource.Select(i => map(i)).AsQueryable();
        }
        public object Execute(Expression expression)
        {
            throw new NotImplementedException();
        }
        public TResult Execute<TResult>(Expression expression)
        {
            throw new NotImplementedException();
        }
        internal static TransformStage Singleton = new TransformStage();
    }
    public class ParametricUVToTriangles : IQueryable<Vector3D>
    {
        public ParametricUVToTriangles(IParametricUV primitive)
        {
            _primitive = primitive;
        }
        public Expression Expression => Expression.Constant(this);
        public Type ElementType => typeof(Vector3D);
        public IQueryProvider Provider => TransformStage.Singleton;
        public IEnumerator<Vector3D> GetEnumerator()
        {
            return PrimitiveAssembly.CreateTriangles(_primitive).GetEnumerator();
        }
        IEnumerator IEnumerable.GetEnumerator()
        {
            return PrimitiveAssembly.CreateTriangles(_primitive).GetEnumerator();
        }
        IParametricUV _primitive;
    }
}