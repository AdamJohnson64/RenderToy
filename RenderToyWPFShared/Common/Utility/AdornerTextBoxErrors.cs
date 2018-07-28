using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;

namespace RenderToy.WPF
{
    class AdornerTextBoxErrors : Adorner
    {
        public AdornerTextBoxErrors(TextBox host) : base(host)
        {
            IsHitTestVisible = false;
            CompositionTarget.Rendering += (s, e) =>
            {
                InvalidateVisual();
            };
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            try
            {
                if (!IsVisible) return;
                if (Errors == null) return;
                var hosttextbox = (TextBox)AdornedElement;
                var brushTransparent = new SolidColorBrush(Color.FromArgb(32, 255, 0, 0));
                var penTransparent = new Pen(new SolidColorBrush(Color.FromArgb(64, 255, 0, 0)), 1);
                var penSolid = new Pen(new SolidColorBrush(Color.FromRgb(255, 0, 0)), 1);
                foreach (var error in Errors)
                {
                    if (error.line < hosttextbox.GetFirstVisibleLineIndex() || error.line > hosttextbox.GetLastVisibleLineIndex()) continue;
                    var linestart = hosttextbox.GetCharacterIndexFromLineIndex(error.line);
                    var rect1 = hosttextbox.GetRectFromCharacterIndex(linestart + error.charstart);
                    var rect2 = hosttextbox.GetRectFromCharacterIndex(linestart + error.charend + 1);
                    rect1.Union(rect2);
                    drawingContext.DrawRectangle(brushTransparent, penTransparent, rect1);
                    var formattedtext = new FormattedText(error.error, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 12, Brushes.Red, 1.0);
                    drawingContext.DrawRectangle(Brushes.Black, penSolid, new Rect(rect1.Left, rect1.Bottom, formattedtext.Width, formattedtext.Height));
                    drawingContext.DrawText(formattedtext, new Point(rect1.Left, rect1.Bottom));
                }
            }
            catch (Exception)
            {
            }
        }
        public void SetErrors(IEnumerable<ErrorDefinition> errors)
        {
            Errors = errors;
            InvalidateVisual();
        }
        IEnumerable<ErrorDefinition> Errors;
    }
    struct ErrorDefinition
    {
        public int line, charstart, charend;
        public string error;
        public static IEnumerable<ErrorDefinition> GetErrors(string text)
        {
            var matcherrors = new Regex("(?'ERRORFILENAME'[^\n\\(]*)\\((?'ERRORLINE'[0-9]*),(?'ERRORCHARSTART'[0-9]*)(-(?'ERRORCHAREND'[0-9]*))?\\):[\\s]*(?'ERRORSTRING'[^\n]*)");
            var match = matcherrors.Match(text);
            while (match.Success)
            {
                string errorfilename = match.Groups["ERRORFILENAME"].Value;
                int errorline = int.Parse(match.Groups["ERRORLINE"].Value) - 1;
                int errorcharstart = int.Parse(match.Groups["ERRORCHARSTART"].Value) - 1;
                int errorcharend;
                if (!int.TryParse(match.Groups["ERRORCHAREND"].Value, out errorcharend))
                {
                    errorcharend = errorcharstart;
                }
                string errorstring = match.Groups["ERRORSTRING"].Value;
                yield return new ErrorDefinition { line = errorline, charstart = errorcharstart, charend = errorcharend, error = errorstring };
                match = match.NextMatch();
            }
        }
    }
}