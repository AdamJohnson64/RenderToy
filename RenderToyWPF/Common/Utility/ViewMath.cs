using RenderToy.Math;
using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Linq.Expressions;
using System.Windows;
using System.Windows.Media;
using LINQ = System.Linq.Expressions;

namespace RenderToy.WPF
{
    class ViewMath : FrameworkElement
    {
        public ViewMath()
        {
            DataContext = LINQ.Expression.Multiply(
                LINQ.Expression.Add(LINQ.Expression.Constant(1.0), LINQ.Expression.Constant(2.0)),
                LINQ.Expression.Add(LINQ.Expression.Constant(1.0), LINQ.Expression.Constant(2.0)));
            DataContext = LINQ.Expression.Add(LINQ.Expression.Constant(99999.0), LINQ.Expression.Constant(2.0));
            DataContext = MatrixExtensions.CreateDX43();
        }
        #region - Section : Overrides -
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            drawingContext.DrawRectangle(Brushes.White, null, new Rect(0, 0, ActualWidth, ActualHeight));
            var part = Layout(DataContext, null);
            if (part.Size.Width <= 0 || part.Size.Height <= 0) return;
            part.Draw(drawingContext, new Rect(0, 0, part.Size.Width, part.Size.Height));
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            var part = Layout(DataContext, null);
            return part.Size;
        }
        #endregion
        #region - Section : Rendering -
        class ArrangedPart
        {
            public Size Size;
            public Action<DrawingContext, Rect> Draw = (drawingContext, area) => { };
        }
        static ArrangedPart Layout(object expressioninner, object expressionouter)
        {
            #region - Section : Render Operators -
            if (expressioninner is LINQ.Expression expression)
            {
                if (expression.NodeType == LINQ.ExpressionType.Constant)
                {
                    var cast = (ConstantExpression)expression;
                    return AddText(cast.ToString());
                }
                else if (expression.NodeType == LINQ.ExpressionType.Parameter)
                {
                    var cast = (ParameterExpression)expression;
                    return AddText(cast.ToString());
                }
                else if (expression.NodeType == LINQ.ExpressionType.Add)
                {
                    var cast = (BinaryExpression)expression;
                    return AddBinaryOperator(Layout(cast.Left, expression), "+", Layout(cast.Right, expression), expressioninner, expressionouter);
                }
                else if (expression.NodeType == LINQ.ExpressionType.Subtract)
                {
                    var cast = (BinaryExpression)expression;
                    return AddBinaryOperator(Layout(cast.Left, expression), "−", Layout(cast.Right, expression), expressioninner, expressionouter);
                }
                else if (expression.NodeType == LINQ.ExpressionType.Multiply)
                {
                    var cast = (BinaryExpression)expression;
                    return AddBinaryOperator(Layout(cast.Left, expression), "×", Layout(cast.Right, expression), expressioninner, expressionouter);
                }
                else if (expression.NodeType == LINQ.ExpressionType.Divide)
                {
                    var cast = (BinaryExpression)expression;
                    return AddBinaryOperator(Layout(cast.Left, expression), "÷", Layout(cast.Right, expression), expressioninner, expressionouter);
                }
                Debug.Assert(false, "Cannot measure expression '" + expression.NodeType + "'.");
                return null;
            }
            #endregion
            #region - Section : Render Matrices -
            else if (expressioninner is MatrixExpression matrixexpression)
            {
                const int CELLPAD = 4;
                var allparts = new ArrangedPart[matrixexpression.M.GetLength(0), matrixexpression.M.GetLength(1)];
                for (int j = 0; j < matrixexpression.M.GetLength(1); ++j)
                {
                    for (int i = 0; i < matrixexpression.M.GetLength(0); ++i)
                    {
                        allparts[i, j] = Layout(matrixexpression.M[i, j], matrixexpression);
                    }
                }
                var colsize = new Size[matrixexpression.M.GetLength(1)];
                var rowsize = new Size[matrixexpression.M.GetLength(0)];
                for (int j = 0; j < matrixexpression.M.GetLength(1); ++j)
                {
                    for (int i = 0; i < matrixexpression.M.GetLength(0); ++i)
                    {
                        colsize[j].Width = System.Math.Max(colsize[j].Width, allparts[i, j].Size.Width);
                        colsize[j].Height += allparts[i, j].Size.Height;
                    }
                }
                for (int i = 0; i < matrixexpression.M.GetLength(0); ++i)
                {
                    for (int j = 0; j < matrixexpression.M.GetLength(1); ++j)
                    {
                        rowsize[i].Width += allparts[i, j].Size.Width;
                        rowsize[i].Height = System.Math.Max(rowsize[i].Height, allparts[i, j].Size.Height);
                    }
                }
                for (int i = 0; i < matrixexpression.M.GetLength(0); ++i)
                {
                    rowsize[i].Height += CELLPAD * 2;
                }
                for (int j = 0; j < matrixexpression.M.GetLength(1); ++j)
                {
                    colsize[j].Width += CELLPAD * 2;
                }
                var part = new ArrangedPart();
                part.Size = new Size(colsize.Sum(i => i.Width), rowsize.Sum(i => i.Height));
                part.Draw = (drawingContext, area) =>
                {
                    double x = 0;
                    for (int j = 0; j < matrixexpression.M.GetLength(1); ++j)
                    {
                        double y = 0;
                        for (int i = 0; i < matrixexpression.M.GetLength(0); ++i)
                        {
                            var cellall = new Rect(x + CELLPAD, y + CELLPAD, colsize[j].Width - CELLPAD * 2, rowsize[i].Height - CELLPAD * 2);
                            var cellcontent = allparts[i,j].Size;
                            var cellcenter = new Rect(cellall.Left + (cellall.Width - cellcontent.Width) / 2, cellall.Top + (cellall.Height - cellcontent.Height) / 2, cellcontent.Width, cellcontent.Height);
                            allparts[i, j].Draw(drawingContext, cellcenter);
                            y += rowsize[i].Height;
                        }
                        x += colsize[j].Width;
                    }
                };
                return part;
            }
            #endregion
            Debug.Assert(false, "Cannot measure expression '" + expressioninner.GetType() + "'.");
            return null;
        }
        static ArrangedPart AddBinaryOperator(ArrangedPart lhs, string op, ArrangedPart rhs, object expressioninner, object expressionouter)
        {
            var textparenopen = new FormattedText("(", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, fontface, fontsize, Brushes.Black, 1.0);
            var textoperator = new FormattedText(op, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, fontface, fontsize, Brushes.Black, 1.0);
            var textparenclose = new FormattedText(")", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, fontface, fontsize, Brushes.Black, 1.0);
            var sizeoperator = FrameText(textoperator);
            var heighttotal = System.Math.Max(sizeoperator.Height, System.Math.Max(lhs.Size.Height, rhs.Size.Height));
            var needparen = MatrixExtensions.Precedence(expressioninner) < MatrixExtensions.Precedence(expressionouter);
            var rectparenopen = new Rect(0, 0, 0, heighttotal);
            if (needparen) rectparenopen.Width = textparenopen.Width;
            var rectlhs = new Rect(rectparenopen.Right, 0, lhs.Size.Width, heighttotal);
            var rectoperator = new Rect(rectlhs.Right, 0, textoperator.Width, heighttotal);
            var rectrhs = new Rect(rectoperator.Right, 0, rhs.Size.Width, heighttotal);
            var rectparenclose = new Rect(rectrhs.Right, 0, 0, heighttotal);
            if (needparen) rectparenclose.Width = textparenclose.Width;
            var part = new ArrangedPart();
            part.Size = new Size(rectparenclose.Right, heighttotal);
            part.Draw = (drawingContext, area) =>
            {
                rectparenopen.Offset(area.Left, area.Top);
                rectlhs.Offset(area.Left, area.Top);
                rectoperator.Offset(area.Left, area.Top);
                rectrhs.Offset(area.Left, area.Top);
                rectparenclose.Offset(area.Left, area.Top);
                if (needparen) DrawTextCenter(drawingContext, textparenopen, rectparenopen);
                lhs.Draw(drawingContext, rectlhs);
                DrawTextCenter(drawingContext, textoperator, rectoperator);
                rhs.Draw(drawingContext, rectrhs);
                if (needparen) DrawTextCenter(drawingContext, textparenclose, rectparenclose);
            };
            return part;
        }
        static ArrangedPart AddText(string text)
        {
            var formattedtext = new FormattedText(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, fontface, fontsize, Brushes.Black, 1.0);
            var part = new ArrangedPart();
            part.Size = FrameText(formattedtext);
            part.Draw = (drawingContext, area) => DrawTextCenter(drawingContext, formattedtext, area);
            return part;
        }
        static void DrawTextCenter(DrawingContext drawingContext, FormattedText text, Rect area)
        {
            double offsetx = (area.Width - text.Width) / 2;
            double offsety = (area.Height - text.Height) / 2;
            drawingContext.DrawText(text, new Point(area.Left + offsetx, area.Top + offsety));
        }
        static Size FrameText(FormattedText text)
        {
            return new Size(text.Width, text.Height);
        }
        //static Typeface fontface = new Typeface("Lucida Bright Math");
        static Typeface fontface = new Typeface("Cambria Math");
        static double fontsize = 16;
        #endregion
    }
}