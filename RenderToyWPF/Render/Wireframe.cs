namespace RenderToy
{
    public interface IWireframeRenderer
    {
        void WireframeBegin();
        void WireframeColor(double r, double g, double b);
        void WireframeLine(double x1, double y1, double x2, double y2);
        void WireframeEnd();
    }
}