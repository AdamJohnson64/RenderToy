////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Linq.Expressions;

namespace RenderToy.Math
{
    public class MatrixExpression
    {
        public Expression[,] M;
    }
    public static class MatrixExtensions
    {
        public static MatrixExpression CreateDX43()
        {
            var m = new Expression[4, 4];
            for (int j = 0; j < 4; ++j)
            {
                for (int i = 0; i < 4; ++i)
                {
                    m[i, j] = j == 3 ? (i == 3 ? ConstantOne : ConstantZero) : Expression.Variable(typeof(double), "M" + (i + 1) + (j + 1));
                }
            }
            return new MatrixExpression { M = m };
        }
        public static MatrixExpression CreateDX44()
        {
            var m = new Expression[4, 4];
            for (int j = 0; j < 4; ++j)
            {
                for (int i = 0; i < 4; ++i)
                {
                    m[i, j] = Expression.Variable(typeof(double), "M" + (i + 1) + (j + 1));
                }
            }
            return new MatrixExpression { M = m };
        }
        public static MatrixExpression Identity(int rank)
        {
            var m = new Expression[rank, rank];
            for (int j = 0; j < rank; ++j)
            {
                for (int i = 0; i < rank; ++i)
                {
                    m[i, j] = (i == j) ? ConstantOne : ConstantZero;
                }
            }
            return new MatrixExpression { M = m };
        }
        public static Expression Determinant(this MatrixExpression mat)
        {
            if (mat.M.GetLength(0) != mat.M.GetLength(1))
            {
                throw new InvalidOperationException();
            }
            if (mat.M.GetLength(0) == 1)
            {
                return mat.M[0, 0];
            }
            if (mat.M.GetLength(0) == 2)
            {
                return Subtract(Multiply(mat.M[0, 0], mat.M[1, 1]), Multiply(mat.M[1, 0], mat.M[0, 1]));
            }
            {
                Expression build = null;
                for (int row = 0; row < mat.M.GetLength(0); ++row)
                {
                    var minordet = Multiply(mat.M[row, 0], Determinant(Minor(mat, row, 0)));
                    if (build == null)
                    {
                        build = minordet;
                    }
                    else
                    {
                        switch (row % 2)
                        {
                            case 0:
                                build = Add(build, Multiply(mat.M[row, 0], minordet));
                                break;
                            case 1:
                                build = Subtract(build, Multiply(mat.M[row, 0], minordet));
                                break;
                        }
                    }
                }
                return build;
            }
        }
        public static MatrixExpression Minor(this MatrixExpression mat, int row, int col)
        {
            var m = new Expression[mat.M.GetLength(0) - 1, mat.M.GetLength(1) - 1];
            for (int j = 0; j < mat.M.GetLength(1) - 1; ++j)
            {
                for (int i = 0; i < mat.M.GetLength(0) - 1; ++i)
                {
                    m[i, j] = mat.M[i + (i < row ? 0 : 1), j + (j < col ? 0 : 1)];
                }
            }
            return new MatrixExpression { M = m };
        }
        public static MatrixExpression Transpose(this MatrixExpression mat)
        {
            var m = new Expression[mat.M.GetLength(1), mat.M.GetLength(0)];
            for (int j = 0; j < mat.M.GetLength(1); ++j)
            {
                for (int i = 0; i < mat.M.GetLength(0); ++i)
                {
                    m[j, i] = mat.M[i, j];
                }
            }
            return new MatrixExpression { M = m };
        }
        public static MatrixExpression Invert(this MatrixExpression mat)
        {
            if (mat.M.GetLength(0) != mat.M.GetLength(1))
            {
                throw new InvalidOperationException();
            }
            var det = Divide(ConstantOne, Determinant(mat));
            var m = new Expression[mat.M.GetLength(0), mat.M.GetLength(1)];
            for (int j = 0; j < mat.M.GetLength(1); ++j)
            {
                for (int i = 0; i < mat.M.GetLength(0); ++i)
                {
                    bool isNegated = (i + j) % 2 == 1;
                    m[i, j] = Multiply(det, Multiply(((i + j) % 2 == 0) ? ConstantOne : ConstantMinusOne, Determinant(Minor(mat, i, j))));
                }
            }
            return Transpose(new MatrixExpression { M = m });
        }
        public static string PrettyPrint(this Expression e)
        {
            return PrettyPrint(e, null);
        }
        static string PrettyPrint(Expression e, Expression pred)
        {
            string build = null;
            if (e.NodeType == ExpressionType.Constant)
            {
                return ((ConstantExpression)e).Value.ToString();
            }
            else if (e.NodeType == ExpressionType.Parameter)
            {
                return ((ParameterExpression)e).Name;
            }
            else if (e.NodeType == ExpressionType.Add)
            {
                var binop = (BinaryExpression)e;
                build = PrettyPrint(binop.Left, e) + " + " + PrettyPrint(binop.Right, e);
            }
            else if (e.NodeType == ExpressionType.Subtract)
            {
                var binop = (BinaryExpression)e;
                build = PrettyPrint(binop.Left, e) + " - " + PrettyPrint(binop.Right, e);
            }
            else if (e.NodeType == ExpressionType.Multiply)
            {
                var binop = (BinaryExpression)e;
                build = PrettyPrint(binop.Left, e) + " * " + PrettyPrint(binop.Right, e);
            }
            else if (e.NodeType == ExpressionType.Divide)
            {
                var binop = (BinaryExpression)e;
                build = PrettyPrint(binop.Left, e) + " / " + PrettyPrint(binop.Right, e);
            }
            if (build == null) throw new NotSupportedException("Cannot pretty print '" + e + "'.");
            return Precedence(e) >= Precedence(pred) ? build : "(" + build + ")";
        }
        public static int Precedence(object e)
        {
            if (e == null) return -1;
            if (e is Expression ex)
            {
                switch (ex.NodeType)
                {
                    case ExpressionType.Add: return 0;
                    case ExpressionType.Subtract: return 0;
                    case ExpressionType.Multiply: return 1;
                    case ExpressionType.Divide: return 1;
                }
            }
            if (e is MatrixExpression) return 99;
            throw new NotSupportedException();
        }
        static Expression Add(Expression lhs, Expression rhs)
        {
            bool constantlhs = lhs.NodeType == ExpressionType.Constant;
            bool constantrhs = rhs.NodeType == ExpressionType.Constant;
            if (constantlhs && constantrhs)
            {
                var clhs = (ConstantExpression)lhs;
                var crhs = (ConstantExpression)rhs;
                return Expression.Constant((double)clhs.Value + (double)crhs.Value);
            }
            else if (constantlhs)
            {
                var clhs = (ConstantExpression)lhs;
                if ((double)clhs.Value == 0) return rhs;
            }
            else if (constantrhs)
            {
                var crhs = (ConstantExpression)rhs;
                if ((double)crhs.Value == 0) return lhs;
            }
            return Expression.Add(lhs, rhs);
        }
        static Expression Subtract(Expression lhs, Expression rhs)
        {
            bool constantlhs = lhs.NodeType == ExpressionType.Constant;
            bool constantrhs = rhs.NodeType == ExpressionType.Constant;
            if (constantlhs && constantrhs)
            {
                var clhs = (ConstantExpression)lhs;
                var crhs = (ConstantExpression)rhs;
                return Expression.Constant((double)clhs.Value - (double)crhs.Value);
            }
            else if (constantlhs)
            {
                var clhs = (ConstantExpression)lhs;
                if ((double)clhs.Value == 0) return rhs;
            }
            else if (constantrhs)
            {
                var crhs = (ConstantExpression)rhs;
                if ((double)crhs.Value == 0) return lhs;
            }
            return Expression.Subtract(lhs, rhs);
        }
        static Expression Multiply(Expression lhs, Expression rhs)
        {
            bool constantlhs = lhs.NodeType == ExpressionType.Constant;
            bool constantrhs = rhs.NodeType == ExpressionType.Constant;
            if (constantlhs && constantrhs)
            {
                var clhs = (ConstantExpression)lhs;
                var crhs = (ConstantExpression)rhs;
                return Expression.Constant((double)clhs.Value * (double)crhs.Value);
            }
            else if (constantlhs)
            {
                var clhs = (ConstantExpression)lhs;
                if ((double)clhs.Value == 0) return ConstantZero;
                if ((double)clhs.Value == 1) return rhs;
            }
            else if (constantrhs)
            {
                var crhs = (ConstantExpression)rhs;
                if ((double)crhs.Value == 0) return ConstantZero;
                if ((double)crhs.Value == 1) return lhs;
            }
            return Expression.Multiply(lhs, rhs);
        }
        static Expression Divide(Expression lhs, Expression rhs)
        {
            bool constantlhs = lhs.NodeType == ExpressionType.Constant;
            bool constantrhs = rhs.NodeType == ExpressionType.Constant;
            if (constantlhs && constantrhs)
            {
                var clhs = (ConstantExpression)lhs;
                var crhs = (ConstantExpression)rhs;
                return Expression.Constant((double)clhs.Value / (double)crhs.Value);
            }
            else if (constantlhs)
            {
                var clhs = (ConstantExpression)lhs;
                if ((double)clhs.Value == 0) return ConstantZero;
            }
            else if (constantrhs)
            {
                var crhs = (ConstantExpression)rhs;
                if ((double)crhs.Value == 0) throw new InvalidOperationException();
                if ((double)crhs.Value == 1) return lhs;
            }
            return Expression.Divide(lhs, rhs);
        }
        static Expression ConstantOne = Expression.Constant(1.0);
        static Expression ConstantMinusOne = Expression.Constant(-1.0);
        static Expression ConstantZero = Expression.Constant(0.0);
    }
}
