using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Windows.Documents;

namespace RenderToy.WPF.Xps
{
    public static class RenderToyReference
    {
        public static FlowDocument CreateDocument()
        {
            var document = new FlowDocument();
            foreach (var namespacegroup in Assembly.GetExecutingAssembly().GetTypes().Where(i => !string.IsNullOrWhiteSpace(i.Namespace)).GroupBy(i => i.Namespace).OrderBy(i => i.Key))
            {
                {
                    var paragraph = new Paragraph();
                    paragraph.Inlines.Add(new Run { Text = "Namespace '" + namespacegroup.Key + "'", FontSize = 18 });
                    document.Blocks.Add(paragraph);
                }
                var typelist = new List();
                foreach (var type in namespacegroup.OrderBy(i => i.FullName))
                {
                    var typelistitem = new ListItem();
                    {
                        var paragraph = new Paragraph();
                        paragraph.Inlines.Add(new Run { Text = "Type '" + type.FullName + "'" });
                        typelistitem.Blocks.Add(paragraph);
                    }
                    var memberlist = new List();
                    foreach (var method in type.GetMethods(BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public).Where(i => i.GetCustomAttribute(typeof(CompilerGeneratedAttribute)) == null).OrderBy(i => i.Name))
                    {
                        var methodlistitem = new ListItem();
                        var methodparagraph = new Paragraph();
                        methodparagraph.Inlines.Add(new Run { Text = "Method '" + method.ReturnType + " " + method.Name + "'" });
                        methodlistitem.Blocks.Add(methodparagraph);
                        memberlist.ListItems.Add(methodlistitem);
                    }
                    foreach (var field in type.GetFields(BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public).Where(i => i.GetCustomAttribute(typeof(CompilerGeneratedAttribute)) == null).OrderBy(i => i.Name))
                    {
                        var fieldlistitem = new ListItem();
                        var fieldparagraph = new Paragraph();
                        fieldparagraph.Inlines.Add(new Run { Text = "Field '" + field.FieldType + " " + field.Name + "'" });
                        fieldlistitem.Blocks.Add(fieldparagraph);
                        memberlist.ListItems.Add(fieldlistitem);
                    }
                    typelistitem.Blocks.Add(memberlist);
                    typelist.ListItems.Add(typelistitem);
                }
                document.Blocks.Add(typelist);
            }
            return document;
        }
    }
}