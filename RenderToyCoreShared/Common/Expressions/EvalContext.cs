////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

namespace RenderToy.Materials
{
    public class EvalContext
    {
        public double U, V;
        public EvalContext()
        {
            U = 0;
            V = 0;
        }
        public EvalContext(EvalContext clonefrom)
        {
            U = clonefrom.U;
            V = clonefrom.V;
        }
    }
}