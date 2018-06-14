﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.WPF.Xps;
using System.Windows;

namespace RenderToy.WPF
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            Material.Root = StockMaterials.Brick;
            Reference.Document = RenderToyReference.CreateDocument();
        }
    }
}
