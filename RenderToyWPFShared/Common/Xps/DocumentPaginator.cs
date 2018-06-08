using System.Windows;
using System.Windows.Documents;
using System.Windows.Media;

namespace RenderToy.WPF.Xps
{
    public class DocumentPaginatorWrapper : DocumentPaginator
    {
        public static DependencyProperty PageNumberProperty = DependencyProperty.RegisterAttached("PageNumber", typeof(int), typeof(DocumentPaginatorWrapper), new FrameworkPropertyMetadata(0, FrameworkPropertyMetadataOptions.AffectsArrange | FrameworkPropertyMetadataOptions.AffectsMeasure | FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.Inherits));
        public DocumentPaginatorWrapper(DocumentPaginator paginator, double header, double footer, DataTemplate templateHeader, DataTemplate templateFooter)
        {
            heightHeader = header;
            heightFooter = footer;
            innerPaginator = paginator;
            this.templateHeader = templateHeader;
            this.templateFooter = templateFooter;
        }
        public override DocumentPage GetPage(int pageNumber)
        {
            DocumentPage page = null;
            page = (this == innerPaginator) ? new DocumentPage(new ContainerVisual()) : innerPaginator.GetPage(pageNumber);
            var newpage = new ContainerVisual();
            newpage.SetValue(PageNumberProperty, pageNumber);
            {
                var sectiondocument = (FlowDocument)templateHeader.LoadContent();
                sectiondocument.ColumnWidth = PageSize.Width;
                var sectionpaginator = ((IDocumentPaginatorSource)sectiondocument).DocumentPaginator;
                sectionpaginator.PageSize = PageSize;
                var sectionpage = sectionpaginator.GetPage(0);
                var sectioncontainer = new ContainerVisual();
                sectioncontainer.Offset = new Vector(0, 0);
                sectioncontainer.Children.Add(sectionpage.Visual);
                newpage.Children.Add(sectioncontainer);
            }
            {
                var sectioncontainer = new ContainerVisual();
                sectioncontainer.Children.Add(page.Visual);
                sectioncontainer.Offset = new Vector(0, heightHeader);
                newpage.Children.Add(sectioncontainer);
            }
            {
                var sectiondocument = (FlowDocument)templateFooter.LoadContent();
                sectiondocument.SetValue(PageNumberProperty, pageNumber);
                sectiondocument.ColumnWidth = PageSize.Width;
                var sectionpaginator = ((IDocumentPaginatorSource)sectiondocument).DocumentPaginator;
                sectionpaginator.PageSize = PageSize;
                var sectionpage = sectionpaginator.GetPage(0);
                var sectioncontainer = new ContainerVisual();
                sectioncontainer.Offset = new Vector(0, heightHeader + page.Size.Height);
                sectioncontainer.Children.Add(sectionpage.Visual);
                newpage.Children.Add(sectioncontainer);
            }
            return new DocumentPage(newpage, PageSize, page.BleedBox, page.ContentBox);
        }
        public override bool IsPageCountValid { get { return innerPaginator.IsPageCountValid; } }
        public override int PageCount { get { return innerPaginator.PageCount; } }
        public override Size PageSize
        {
            get
            {
                var size = innerPaginator.PageSize;
                size.Height = size.Height + heightHeader + heightFooter;
                return size;
            }
            set
            {
                if (this != innerPaginator)
                {
                    innerPaginator.PageSize = value;
                }
            }
        }
        public override IDocumentPaginatorSource Source { get { return innerPaginator.Source; } }
        DataTemplate templateHeader;
        DataTemplate templateFooter;
        double heightHeader;
        double heightFooter;
        DocumentPaginator innerPaginator;
    }
}