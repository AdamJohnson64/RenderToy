////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace RenderToy
{
    public static class Performance
    {
        public static void LogEvent(string text)
        {
            WriteLine("@" + Stopwatch.GetTimestamp() + " TASKID " + (Task.CurrentId ?? 0) + " THREADID " + Thread.CurrentThread.ManagedThreadId + " EVENT \"" + text + "\"");
        }
        public static void LogBegin(string text)
        {
            WriteLine("@" + Stopwatch.GetTimestamp() + " TASKID " + (Task.CurrentId ?? 0) + " THREADID " + Thread.CurrentThread.ManagedThreadId + " BEGIN \"" + text + "\"");
        }
        public static void LogEnd(string text)
        {
            WriteLine("@" + Stopwatch.GetTimestamp() + " TASKID " + (Task.CurrentId ?? 0) + " THREADID " + Thread.CurrentThread.ManagedThreadId + " END \"" + text + "\"");
        }
        static void WriteLine(string text)
        {
#if !WINDOWS_UWP
            Console.WriteLine(text);
            Debug.WriteLine(text);
#endif
        }
    }
    public static class PerformanceHelp
    {
        public static IEnumerable<PerformanceEvent> ReadEvents(TextReader textreader)
        {
            var line = textreader.ReadLine();
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
                    yield return new PerformanceEvent(timestamp, taskid, threadid, eventtype, text);
                }
                line = textreader.ReadLine();
            }
        }
        public static IReadOnlyList<PerformanceTrack> GetBins(IEnumerable<PerformanceEvent> events)
        {
            // Pack the executed blocks into a minimum of non-overlapping display tracks.
            var tracks = new List<PerformanceTrack>();
            foreach (var block in GetBlocks(events))
            {
                var findtrack = tracks
                    .Select(track => new { Track = track, TimeOffset = block.Begin.Timestamp - track.LastTime })
                    .Where(time => time.TimeOffset >= 0)
                    .OrderBy(time => time.TimeOffset)
                    .ToArray();
                PerformanceTrack selecttrack;
                if (findtrack.Length == 0)
                {
                    selecttrack = new PerformanceTrack();
                    tracks.Add(selecttrack);
                }
                else
                {
                    selecttrack = findtrack[0].Track;
                }
                selecttrack.Events.Add(block);
                selecttrack.LastTime = block.End.Timestamp;
            }
            return tracks;
        }
        static IReadOnlyList<PerformanceBlock> GetBlocks(IEnumerable<PerformanceEvent> events)
        {
            return EnumerateBlocks(events).OrderBy(x => x.Begin.Timestamp).ToArray();
        }
        static IEnumerable<PerformanceBlock> EnumerateBlocks(IEnumerable<PerformanceEvent> events)
        {
            var open = new Dictionary<string, Stack<PerformanceEvent>>();
            foreach (var e in events)
            {
                if (e.EventType == PerformanceEvent.EType.Begin)
                {
                    if (!open.ContainsKey(e.Text))
                    {
                        open[e.Text] = new Stack<RenderToy.PerformanceEvent>();
                    }
                    open[e.Text].Push(e);
                }
                if (e.EventType == PerformanceEvent.EType.End)
                {
                    if (!open.ContainsKey(e.Text))
                    {
                        // We'll allow this for now.
                        continue;
                        throw new InvalidDataException("Closed block '" + e.Text + "' at @" + e.Timestamp + " was never opened.");
                    }
                    yield return new PerformanceBlock(open[e.Text].Pop(), e);
                    if (open[e.Text].Count == 0)
                    {
                        open.Remove(e.Text);
                    }
                }
            }
        }
    }
    [DebuggerDisplay("{Begin.Timestamp} {End.Timestamp} {Begin.Text}")]
    public struct PerformanceBlock
    {
        public PerformanceBlock(PerformanceEvent begin, PerformanceEvent end)
        {
            Begin = begin;
            End = end;
        }
        public readonly PerformanceEvent Begin;
        public readonly PerformanceEvent End;
    }
    [DebuggerDisplay("{Timestamp} {TaskID} {ThreadID} {EventType} {Text}")]
    public struct PerformanceEvent
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
    public class PerformanceTrack
    {
        public long LastTime = 0;
        public List<PerformanceBlock> Events = new List<PerformanceBlock>();
    }
}