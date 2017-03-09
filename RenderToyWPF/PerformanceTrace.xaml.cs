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
    public partial class PerformanceTrace : UserControl
    {
        public PerformanceTrace()
        {
            InitializeComponent();
        }
        private void TraceDataChanged(object sender, TextChangedEventArgs e)
        {
            var events = new List<PerformanceEvent>();
            // Read all events in the stream.
            {
                var stringreader = new StringReader(((TextBox)e.Source).Text);
                var line = stringreader.ReadLine();
                while (line != null)
                {
                    var match = Regex.Match(line, @"@(?'timestamp'[0-9]*)\s*TASKID\s*(?'taskid'[0-9]*)\s*THREADID\s*(?'threadid'[0-9]*)\s*(?'eventtype'BEGIN|END|EVENT)\s*""(?'text'.*)""$");
                    if (match != null && match.Success)
                    {
                        long timestamp = long.Parse(match.Groups["timestamp"].Value);
                        int taskid = int.Parse(match.Groups["taskid"].Value);
                        int threadid = int.Parse(match.Groups["threadid"].Value);
                        PerformanceEvent.EType eventtype;
                        {
                            string eventtypestring = match.Groups["eventtype"].Value;
                            if (eventtypestring == "BEGIN") eventtype = PerformanceEvent.EType.Begin;
                            else if (eventtypestring == "END") eventtype = PerformanceEvent.EType.End;
                            else if (eventtypestring == "EVENT") eventtype = PerformanceEvent.EType.Event;
                            else eventtype = PerformanceEvent.EType.Unknown;
                        }
                        string text = match.Groups["text"].Value;
                        events.Add(new PerformanceEvent(timestamp, taskid, threadid, eventtype, text));
                    }
                    line = stringreader.ReadLine();
                }
            }
            // Process the events into something usable.
            var bytaskid = events
                .GroupBy(x => x.TaskID)
                .ToDictionary(getkey => getkey.Key, getvalue => getvalue.ToArray());
            var bythreadid = events
                .GroupBy(x => x.ThreadID)
                .ToDictionary(getkey => getkey.Key, getvalue => getvalue.ToArray());
            var blocksbytag = events
                .Where(x => x.EventType == PerformanceEvent.EType.Begin || x.EventType == PerformanceEvent.EType.End)
                .GroupBy(x => x.Text)
                .ToDictionary(getkey => getkey.Key, getvalue => getvalue.ToArray());
            // Pack the executed blocks into a minimum of non-overlapping display tracks.
            var tracks = new List<PerformanceTrack>();
            foreach (var tag in blocksbytag)
            {
                var findtrack = tracks
                    .Select(track => new { Track = track, TimeOffset = tag.Value[0].Timestamp - track.LastTime })
                    .Where(time => time.TimeOffset >= 0)
                    .OrderBy(time => time.TimeOffset)
                    .ToArray();
                PerformanceTrack selecttrack;
                if (findtrack.Length == 0)
                {
                    selecttrack = new PerformanceTrack();
                    tracks.Add(selecttrack);
                } else {
                    selecttrack = findtrack[0].Track;
                }
                selecttrack.Events.AddRange(tag.Value);
                selecttrack.LastTime = tag.Value[tag.Value.Length - 1].Timestamp;
            }
            // TODO: I need more cores for this.
            throw new NotImplementedException();
        }
    }
    [DebuggerDisplay("{Timestamp} {TaskID} {ThreadID} {EventType} {Text}")]
    struct PerformanceEvent
    {
        public PerformanceEvent(long timestamp, int taskid, int threadid, EType eventtype, string text)
        {
            Timestamp = timestamp;
            TaskID = taskid;
            ThreadID = threadid;
            EventType = eventtype;
            Text = text;
        }
        public enum EType { Unknown, Begin, End, Event }
        public readonly long Timestamp;
        public readonly int TaskID;
        public readonly int ThreadID;
        public readonly EType EventType;
        public readonly string Text;
    }
    class PerformanceTrack
    {
        public long LastTime = 0;
        public List<PerformanceEvent> Events = new List<PerformanceEvent>();
    }
}
