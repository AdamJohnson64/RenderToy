////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace RenderToy
{
    public class PerformanceTrackControl : FrameworkElement
    {
        public PerformanceTrackControl()
        {
            DataContextChanged += (s, e) => InvalidateVisual();
            SizeChanged += (s, e) => InvalidateVisual();
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            string data = DataContext as string;
            if (data == null) return;
            // Read all events in the stream.
            var events = PerformanceHelp.ReadEvents(new StringReader(data)).ToList();
            // Render the performance data using the most attrociously bone-headed slow method known to humankind.
            if (events.Count < 2) return;
            long timemin = events[0].Timestamp;
            long timemax = events[events.Count - 1].Timestamp;
            var tracks = PerformanceHelp.GetBins(events);
            int trackcount = 0;
            Pen pen_black = new Pen(Brushes.Black, -1);
            foreach (var track in tracks)
            {
                foreach (var block in track.Events)
                {
                    double x1 = (block.Begin.Timestamp - timemin) * ActualWidth / (timemax - timemin);
                    double y1 = (trackcount + 0) * ActualHeight / tracks.Count;
                    double x2 = (block.End.Timestamp - timemin) * ActualWidth / (timemax - timemin);
                    double y2 = (trackcount + 1) * ActualHeight / tracks.Count;
                    var rect = new Rect(x1, y1, x2 - x1, y2 - y1);
                    drawingContext.DrawRectangle(Brushes.LightGray, pen_black, rect);
                    drawingContext.PushClip(new RectangleGeometry(new Rect(x1, y1, x2 - x1, y2 - y1)));
                    drawingContext.DrawText(new FormattedText(block.Begin.Text, System.Globalization.CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 10, Brushes.Black), new Point(x1, y1));
                    drawingContext.Pop();
                }
                ++trackcount;
            }
        }
    }
    public partial class PerformanceTrace : UserControl
    {
        public PerformanceTrace()
        {
            InitializeComponent();
        }
    }
}
