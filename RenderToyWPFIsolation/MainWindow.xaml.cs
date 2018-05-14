﻿using RenderToy.Materials;
using System.Windows;

namespace RenderToyWPFIsolation
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            Material.Root = StockMaterials.Brick();
        }
    }
}
