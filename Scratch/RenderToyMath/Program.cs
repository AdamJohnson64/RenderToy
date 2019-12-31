using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RenderToy
{
    class MatrixSym
    {
        public MatrixSym(int rank)
        {
            this.rank = rank;
            this.elements = new string[rank * rank];
        }
        public static MatrixSym DefaultWPF4x4
        {
            get
            {
                MatrixSym m = new MatrixSym(4);
                for (int row = 0; row < m.rank; ++row)
                {
                    for (int col = 0; col < m.rank; ++col)
                    {
                        string e;
                        if (row == 3 && col == 0) { e = "OffsetX"; }
                        else if (row == 3 && col == 1) { e = "OffsetY"; }
                        else if (row == 3 && col == 2) { e = "OffsetZ"; }
                        else { e = "M" + (row + 1) + (col + 1); }
                        m.elements[col + m.rank * row] = e;
                    }
                }
                return m;
            }
        }
        public static MatrixSym Create(int rank)
        {
            return Create(rank, "");
        }
        public static MatrixSym Create(int rank, string prefix)
        {
            MatrixSym m = new MatrixSym(rank);
            for (int row = 0; row < m.rank; ++row)
            {
                for (int col = 0; col < m.rank; ++col)
                {
                    m.elements[col + m.rank * row] = prefix + "M" + (row + 1) + (col + 1); ;
                }
            }
            return m;
        }
        public MatrixSym Determinant()
        {
            if (rank < 1)
            {
                throw new InvalidOperationException();
            }
            if (rank == 1)
            {
                return this;
            }
            if (rank == 2)
            {
                MatrixSym m = new MatrixSym(rank - 1);
                m.elements[0] = "(" + elements[0] + "*" + elements[3] + "-" + elements[1] + "*" + elements[2] + ")";
                return m;
            }
            {
                // Laplacian.
                string result = "(";
                for (int row = 0; row < rank; ++row)
                {
                    // Always us the last column, in the 4x4 case we can take advantage of this for affine transforms.
                    int col = rank - 1;
                    bool isnegate = ((row + col) % 2) == 1;
                    var det = Minor(row, col).Determinant();
                    Debug.Assert(det.rank == 1);
                    result += (isnegate ? "-" : "+") + elements[col + row * rank] + "*" + det.elements[0];
                }
                result += ")";
                MatrixSym m = new MatrixSym(1);
                m.elements[0] = result;
                return m;
            }
        }
        public MatrixSym Inverse()
        {
            MatrixSym transpose = Transpose();
            MatrixSym result = new MatrixSym(rank);
            for (int row = 0; row < rank; ++row)
            {
                for (int col = 0; col < rank; ++col)
                {
                    bool isNegated = (col + row) % 2 == 1;
                    result.elements[col + rank * row] = (isNegated ? "-" : "+") + transpose.Minor(row, col).Determinant().elements[0];
                }
            }
            return result;
        }
        public MatrixSym Minor(int getrow, int getcol)
        {
            MatrixSym m = new MatrixSym(rank - 1);
            for (int row = 0; row < m.rank; ++row)
            {
                for (int col = 0; col < m.rank; ++col)
                {
                    int selectcol = col >= getcol ? col + 1 : col;
                    int selectrow = row >= getrow ? row + 1 : row;
                    m.elements[col + row * m.rank] =
                        elements[selectcol + selectrow * rank];
                }
            }
            return m;
        }
        public MatrixSym Transpose()
        {
            MatrixSym m = new MatrixSym(rank);
            for (int row = 0; row < m.rank; ++row)
            {
                for (int col = 0; col < m.rank; ++col)
                {
                    m.elements[col + row * m.rank] = elements[row + col * m.rank];
                }
            }
            return m;
        }
        public readonly int rank;
        public readonly string[] elements;
    }
    class Program
    {
        static void Main(string[] args)
        {
            string buildcode = "";
            ////////////////////////////////////////////////////////////////////////////////
            {
                var m = MatrixSym.Create(4, "val.");
                var det = m.Determinant();
                buildcode +=
                    "public static double Determinant(RenderToy.Matrix3D val) {\n" +
                    "\treturn " + det.elements[0] + ";\n" +
                    "}\n";
            }
            ////////////////////////////////////////////////////////////////////////////////
            {
                var m = MatrixSym.Create(4, "val.");
                var inv = m.Inverse();
                buildcode +=
                    "public static RenderToy.Matrix3D Inverse(RenderToy.Matrix3D val) {\n" +
                    "\tdouble invdet = 1 / Determinant(val);\n" +
                    "\treturn new RenderToy.Matrix3D(\n";
                bool addcomma = false;
                for (int row = 0; row < inv.rank; ++row)
                {
                    for (int col = 0; col < inv.rank; ++col)
                    {
                        if (addcomma)
                        {
                            buildcode += ",\n";
                        }
                        addcomma = true;
                        buildcode +=
                            "\t\tinvdet * (" + inv.elements[col + row * inv.rank] + ")";
                    }
                }
                buildcode +=
                    ");\n" +
                    "}\n";
            }
            Console.WriteLine(buildcode);
        }
    }
}
