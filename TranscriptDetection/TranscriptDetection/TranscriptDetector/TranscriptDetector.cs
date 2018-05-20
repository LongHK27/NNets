using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.Util;
using System.Drawing;
using Emgu.CV.Util;
using System.Diagnostics;

namespace TranscriptDetection
{
    /// <summary>
    ///  a(x - x0) + b(y - y0) = 0
    /// </summary>
    public class ILine
    {
        public double a;
        public double b;
        public double c;
        public Point M1;
        public Point M2;
        public Point M11;
        public Point M22;
        public Point n;

        public ILine (Point A, Point B)
        {
            var d1 = A.X * A.X + A.Y * A.Y;
            var d2 = B.X * B.X + B.Y * B.Y;

            this.M1 = d1 < d2 ? A : B;
            this.M2 = d1 < d2 ? B : A;
            this.M11 = new Point(this.M1.X, -this.M1.Y);
            this.M22 = new Point(this.M2.X, -this.M2.Y);

            this.n = new Point(this.M1.Y - this.M2.Y, this.M2.X - this.M1.X);    //vector phap tuyen n(-b, a) v(a, b)
            this.a = this.M1.Y - this.M2.Y;
            this.b = this.M2.X - this.M1.X;
            this.c = -(this.n.X * this.M1.X + this.n.Y * this.M1.Y);
        }

        /// <summary>
        /// Caculate cos(Alpha) of 2 vector
        /// </summary>
        /// <param name="v2">vector v2</param>
        /// <returns></returns>
        public double CosAlpha(Point v2)
        {
            var v1 = this.n;
            return (double)(v1.X * v2.X + v1.Y * v2.Y) / (double)(Math.Sqrt(v1.X * v1.X + v1.Y * v1.Y) * Math.Sqrt(v2.X * v2.X + v2.Y * v2.Y));
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="line"></param>
        /// <returns></returns>
        public Point FindIntersection(ILine line)
        {
            var d   = this.a * line.b - this.b * line.a;
            var dx  = -this.c * line.b + line.c * this.b;
            var dy  = -this.a * line.c + this.c * line.a;

            if(d != 0)
            {
                return new Point((int)(dx / d), (int) (dy / d));
            } else
            {
                return new Point();
            }
        }

        public double Distances (ILine line)
        {
            return Math.Abs(this.a * line.M1.X + this.b * line.M1.Y + this.c) / Math.Sqrt(this.a * this.a + this.b * this.b);
        }
    }

    public class Subject
    {
        public string name;
        public double score;
        public Image<Gray, byte> image_s1;
        public Image<Gray, byte> image_s2;
        public Image<Gray, byte> image_avg;

        public Subject() { }

        public Subject(string name, double score, Image<Gray, byte> image_s1, Image<Gray, byte> image_s2, Image<Gray, byte> image_avg)
        {
            this.name = name;
            this.score = score;
            this.image_s1 = image_s1;
            this.image_s2 = image_s2;
            this.image_avg = image_avg;
        }
    }

    public static class TranscriptDetector
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        public static void Detect(Image<Bgr, byte> source)
        {
            using (var source_rotate = Rotate(source))
            {
                Image<Bgr, byte> transcript_image;
                List<ILine> vertical_line, horizontal_line;
                var source_drawed = FindTranscriptLocation(source_rotate, out transcript_image, out horizontal_line, out vertical_line);

                DetectSubject(new Image<Gray, byte>(transcript_image.Bitmap), vertical_line);

                //CvInvoke.Imshow("source_drawed", source_drawed.Resize(0.5, Inter.Linear));
                CvInvoke.Imshow("transcript_image", transcript_image);
            }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        private static Image<Bgr, byte> Rotate(Image<Bgr, byte> source, bool debug = false)
        {
            using (var gray_image = new Image<Gray, byte>(source.Bitmap))
            {

                gray_image._GammaCorrect(3.5);

                using (var binary_imge = gray_image.ThresholdBinaryInv(new Gray(160), new Gray(255)))
                {

                    var element = CvInvoke.GetStructuringElement(ElementShape.Cross, new Size(3, 3), new Point(-1, -1));

                    var canny_image = binary_imge.Canny(100.0, 0.0);

                    if ( debug ) CvInvoke.Imshow("binary_imge", binary_imge.Resize(0.5, Inter.Linear));

                    var color_image = new Image<Bgr, byte>(binary_imge.Bitmap);

                    var lines = binary_imge.HoughLinesBinary(1, Math.PI / 180, 100, 30, 2);

                    var list = new List<List<ILine>>{
                        new List<ILine>(),  // vertical
                        new List<ILine>()   // horizontal
                    };

                    foreach (LineSegment2D[] lineSegment in lines)
                    {
                        foreach (LineSegment2D line in lineSegment)
                        {
                            var line_i = new ILine(new Point(line.P1.X, -line.P1.Y), new Point(line.P2.X, -line.P2.Y));

                            if (line_i.CosAlpha(new Point(0, 1)) < Math.Cos(Math.PI / 6))
                            {
                                list[0].Add(line_i);    // vertical cos(x) < cos(45deg)
                            }
                            else
                            {
                                list[1].Add(line_i);    // horizontal cos(x) > cos(45deg)
                            }

                            CvInvoke.Line(color_image, line.P1, line.P2, new MCvScalar(0, 0, 255), 2);
                        }
                    }

                    var vertical_ILine_group = Grouping(list[0], true, 20);
                    var horizontal_ILine_group = Grouping(list[1], false, 20);

                    var horizontal_line_list = new List<ILine>();

                    foreach (List<ILine> horizontal_line in horizontal_ILine_group)
                    {

                        var l = new ILine(horizontal_line[0].M1, horizontal_line[horizontal_line.Count - 1].M1);

                        horizontal_line_list.Add(l);
                    }

                    var horizontal_line_parallel = ClassifyVector(horizontal_line_list);

                    bool k = false;

                    horizontal_line_parallel[0] = horizontal_line_parallel[0].OrderBy(x => x.M1.Y).ToList();

                    var index = GroupingDistance(horizontal_line_parallel[0], 15);

                    var length = 0.0;
                    for (var i = index[0]; i < index[1]; i++)
                    {
                        if (Math.Sqrt(Math.Pow(horizontal_line_parallel[0][i].M22.X - horizontal_line_parallel[0][i].M11.X, 2) + Math.Pow(horizontal_line_parallel[0][i].M22.Y - horizontal_line_parallel[0][i].M11.Y, 2)) > length)
                        {
                            index[0] = i;
                            length = Math.Sqrt(Math.Pow(horizontal_line_parallel[0][i].M22.X - horizontal_line_parallel[0][i].M11.X, 2) + Math.Pow(horizontal_line_parallel[0][i].M22.Y - horizontal_line_parallel[0][i].M11.Y, 2));
                        }
                        CvInvoke.Line(
                            color_image,
                            horizontal_line_parallel[0][i].M11,
                            horizontal_line_parallel[0][i].M22,
                            new MCvScalar(0, 255, 0),
                            2
                        );
                    }



                    var alpha = 90 - Math.Acos(horizontal_line_parallel[0][index[0]].CosAlpha(new Point(1, 0))) * 180 / Math.PI;

                    if (horizontal_line_parallel[0][index[0]].b != 0)
                    {
                        if (horizontal_line_parallel[0][index[0]].a > 0) k = true;
                        alpha = Math.Abs(Math.Atan(-(horizontal_line_parallel[0][index[0]].a / horizontal_line_parallel[0][index[0]].b)) * 180 / Math.PI);
                    }
                    else
                    {
                        alpha = 0;
                    }

                    CvInvoke.Line(
                        color_image,
                        horizontal_line_parallel[0][index[0]].M11,
                        horizontal_line_parallel[0][index[0]].M22,
                        new MCvScalar(255, 255, 0),
                        4
                    );
                    var gray_image_2 = gray_image.Rotate(k ? -alpha : alpha, new Gray(0));

                    CvInvoke.PutText(
                        color_image,
                        alpha + " deg ( " + k + " ) :  >> " + horizontal_line_parallel[0][index[0]].a + " :: " + horizontal_line_parallel[0][index[0]].b,
                        new Point(50, 50),
                        FontFace.HersheySimplex,
                        1.0,
                        new MCvScalar(0, 255, 0)
                    );

                    if (debug) CvInvoke.Imshow("line", color_image.Resize(0.5, Inter.Linear));
                    if (debug) CvInvoke.Imshow("binary_imge2", gray_image_2.Resize(0.5, Inter.Linear));
                    if (debug) CvInvoke.Imshow("gray - 2", gray_image_2.ThresholdBinaryInv(new Gray(200), new Gray(255)).Resize(0.5, Inter.Linear));
                    return source.Rotate(k ? -alpha : alpha, new Bgr(255, 255, 255));
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <param name="transcript"></param>
        /// <returns></returns>
        private static Image<Bgr, byte> FindTranscriptLocation( Image<Bgr, byte> source, out Image<Bgr, byte> transcript, out List<ILine> horizontal_Iline, out List<ILine> vertical_Iline, bool debug = false)
        {
            var source_draw = new Image<Bgr, byte>(source.Bitmap);
            var gray_image = new Image<Gray, byte>(source.Bitmap);
            gray_image._GammaCorrect(2.5);

            var bin = gray_image.ThresholdBinaryInv(new Gray(180), new Gray(255)).Resize(0.5, Inter.Linear);
            var list_k = new List<int>();
            var histogram_iamge = new Image<Gray, byte>(bin.Size);
            histogram_iamge.SetZero();

            var sum = 0;
            for (var i = 0; i < bin.Size.Width; i++)
            {
                sum = 0;
                list_k.Add(0);
                for (var j = 0; j < bin.Size.Height; j++)
                {

                    if (bin.Data[j, i, 0] != 0)
                    {
                        list_k[i]++;
                        sum++;
                        histogram_iamge.Data[histogram_iamge.Height - sum, i, 0] = 255;
                    }
                }
            }

            var avg = histogram_iamge.GetAverage();

            var start = 0;
            var end = 0;
            var check = false;
            var delta = 0;

            var list_step = new List<List<int>>();

            for (var i = 0; i < list_k.Count; i++)
            {
                if (list_k[i] > (int)avg.MCvScalar.V0 + delta && check)
                {
                    end++;
                    if (end == list_k.Count - 1)
                    {
                        if (end - start > 100) list_step.Add(new List<int> { start, end });
                        break;
                    }
                }

                if (list_k[i] > (int)avg.MCvScalar.V0 + delta && !check) { start = i; end = i; check = true; }

                if (list_k[i] <= (int)avg.MCvScalar.V0 + delta && check)
                {
                    if (end - start > 100) list_step.Add(new List<int> { start, end });
                    check = false;
                }

            }

            CvInvoke.Line(
                histogram_iamge,
                new Point(0, histogram_iamge.Height - 70 - (int)avg.MCvScalar.V0),
                new Point(histogram_iamge.Size.Width, histogram_iamge.Height - 70 - (int)avg.MCvScalar.V0),
                new MCvScalar(255, 255, 255),
                2
            );

            CvInvoke.Line(
                histogram_iamge,
                new Point(list_step[0][0], 0),
                new Point(list_step[0][0], histogram_iamge.Height),
                new MCvScalar(255, 255, 255),
                1
            );

            CvInvoke.Line(
                histogram_iamge,
                new Point(list_step[list_step.Count - 1][1], 0),
                new Point(list_step[list_step.Count - 1][1], histogram_iamge.Height),
                new MCvScalar(255, 255, 255),
                1
            );

            var add_x = list_step[0][0] * 2;

            var image = new Mat(source.Mat, new Rectangle(new Point(list_step[0][0] * 2, 0), new Size(2 * (int)Math.Abs(list_step[0][0] - list_step[list_step.Count - 1][1]), 2 * (bin.Size.Height - 1))));

            if ( debug ) CvInvoke.Imshow("image cut", image.ToImage<Gray, byte>().Resize(0.5, Inter.Linear));

            var source_transcript = new Image<Gray, byte>(image.Bitmap);
            source_transcript._GammaCorrect(2.5);

            var binary_image = source_transcript.ThresholdBinaryInv(new Gray(185), new Gray(255));

            var color_image = new Image<Bgr, byte>(binary_image.Bitmap);

            var canny_iamge = source_transcript.ThresholdBinaryInv(new Gray(150), new Gray(255)).Canny(180.0, 2.0);

            if (debug) CvInvoke.Imshow("canny iamge", source_transcript.ThresholdBinaryInv(new Gray(100), new Gray(255)).Resize(0.5, Inter.Linear));

            var lines = canny_iamge.HoughLinesBinary(2, Math.PI / 180, 150, 50, 2);

            var list = new List<List<ILine>>{
                        new List<ILine>(),  // vertical
                        new List<ILine>()   // horizontal
                    };

            foreach (LineSegment2D[] lineSegment in lines)
            {
                foreach (LineSegment2D line in lineSegment)
                {
                    var line_i = new ILine(new Point(line.P1.X, -line.P1.Y), new Point(line.P2.X, -line.P2.Y));

                    if (line_i.CosAlpha(new Point(0, 1)) < Math.Cos(Math.PI / 6))
                    {
                        list[0].Add(line_i);    // vertical cos(x) < cos(45deg)
                    }
                    else
                    {
                        if (line_i.CosAlpha(new Point(0, 1)) > 0.999)
                        {
                            list[1].Add(line_i);    // horizontal cos(x) > cos(45deg)

                        }

                    }

                    CvInvoke.Line(color_image, line.P1, line.P2, new MCvScalar(0, 0, 255), 1);
                }
            }

            var vertical_ILine_group = Grouping(list[0], true, 10);
            var horizontal_ILine_group = Grouping(list[1], false, 5);

            var horizontal_line_list = new List<ILine>();

            var intersection_point = new List<int>();

            var Oy = new ILine(new Point(1, 0), new Point(1, 1));

            foreach (List<ILine> vertical_line in vertical_ILine_group)
            {
                CvInvoke.Line(
                    color_image,
                    new Point(vertical_line[0].M11.X, 0),
                    new Point(vertical_line[0].M11.X, color_image.Size.Height - 1),
                    new MCvScalar(255, 0, 255),
                    2
                );
            }

            foreach (List<ILine> horizontal_line in horizontal_ILine_group)
            {
                var l = new ILine(new Point(0, horizontal_line[0].M11.Y), new Point(color_image.Size.Width - 1, horizontal_line[0].M11.Y));

                intersection_point.Add(-l.M11.Y);

                horizontal_line_list.Add(l);

                CvInvoke.Circle(
                    color_image,
                    new Point(50, horizontal_line[0].M11.Y),
                    6,
                    new MCvScalar(0, 255, 0),
                    2
                );

                CvInvoke.Line(
                    color_image,
                    new Point(0, horizontal_line[0].M11.Y),
                    new Point(color_image.Size.Width - 1, horizontal_line[0].M11.Y),
                    new MCvScalar(255, 0, 255),
                    2
                );
            }

            var line_horizontal = horizontal_ILine_group.OrderBy(x => x[0].M11.Y).ToList();
            var list_order = intersection_point.OrderBy(x => x).ToList();

            for (var i = 0; i < list_order.Count; i++)
            {
                list_order[i] = (int)(Math.Round(list_order[i] / 5.0) * 5.0);
            }

            var distance_matrix = new List<List<int>>();
            var count_value = new List<List<int>>();

            for (var i = 0; i < list_order.Count; i++)
            {
                var distance_i = new List<int>();

                for (var j = 0; j < list_order.Count; j++)
                {
                    distance_i.Add(Math.Abs(list_order[i] - list_order[j]));

                    var isHas = false;
                    for (var k = 0; k < count_value.Count; k++)
                    {
                        if (distance_i[j] == count_value[k][0])
                        {
                            isHas = true;
                            count_value[k][1]++;
                        }
                    }
                    if (!isHas)
                    {
                        count_value.Add(new List<int> { distance_i[j], 1 });
                    }
                }

                distance_matrix.Add(distance_i);
            }

            count_value = count_value.OrderByDescending(x => x[1]).ToList();

            var repeat_step = count_value[0][0] != 0 ? count_value[0][0] : count_value[1][0];

            var startIndex = 0;
            for (var i = 0; i < distance_matrix.Count; i++)
            {
                for (var j = i + 1; j < distance_matrix[i].Count; j++)
                {
                    if (distance_matrix[i][j] == repeat_step)
                    {
                        if (startIndex == 0) startIndex = i;
                        CvInvoke.Circle(
                            color_image,
                            new Point(100, line_horizontal[i][0].M11.Y),
                            10,
                            new MCvScalar(0, 0, 255),
                            2
                        );
                        CvInvoke.Circle(
                            color_image,
                            new Point(150, line_horizontal[j][0].M11.Y),
                            10,
                            new MCvScalar(0, 255, 255),
                            2
                        );
                    }
                }
            }

            for (var i = startIndex; i > 0; i--)
            {
                if (line_horizontal[i][0].M11.Y - line_horizontal[i - 1][0].M11.Y < 50)
                {
                    startIndex = i - 1;
                }
                else
                {
                    break;
                }
            }

            var count_horizontal_line = 1;
            var end_index = startIndex;
            repeat_step = line_horizontal[startIndex + 1][0].M11.Y - line_horizontal[startIndex][0].M11.Y;
            for (var i = startIndex + 1; count_horizontal_line < 19; i++)
            {
                var c = (line_horizontal[i][0].M11.Y - line_horizontal[i - 1][0].M11.Y) / (repeat_step * 1.0);
                count_horizontal_line += (int)Math.Round(c);
                if (count_horizontal_line == 18) end_index = i;
            }

            vertical_ILine_group = vertical_ILine_group.OrderBy(x => x[0].M11.X).ToList();

            var step_subject = line_horizontal[startIndex + 1][0].M11.Y - line_horizontal[startIndex][0].M11.Y;

            CvInvoke.Circle(
                color_image,
                new Point(100, line_horizontal[startIndex][0].M11.Y),
                15,
                new MCvScalar(255, 0, 255),
                2
            );

            CvInvoke.Circle(
                color_image,
                new Point(100, line_horizontal[end_index][0].M11.Y),
                15,
                new MCvScalar(255, 0, 255),
                2
            );

            var transcript_horizontal = (new Mat(
                source_transcript.Mat,
                new Rectangle(
                    new Point(30, line_horizontal[startIndex][0].M11.Y),
                    new Size(source_transcript.Size.Width - 30, line_horizontal[startIndex + 12][0].M11.Y - line_horizontal[startIndex][0].M11.Y))
            )).ToImage<Bgr, byte>();

            var gray_h = new Image<Gray, byte>(transcript_horizontal.Bitmap);

            var canny_iamge_h = gray_h.ThresholdBinaryInv(new Gray(150), new Gray(255)).Canny(180.0, 2.0);

            var lines_h = canny_iamge_h.HoughLinesBinary(2, Math.PI / 180, 150, 30, 2);

            var list_h = new List<List<ILine>>{
                        new List<ILine>(),  // vertical
                        new List<ILine>()   // horizontal
                    };

            foreach (LineSegment2D[] lineSegment in lines_h)
            {
                foreach (LineSegment2D line in lineSegment)
                {
                    var line_i = new ILine(new Point(line.P1.X, -line.P1.Y), new Point(line.P2.X, -line.P2.Y));

                    if (line_i.CosAlpha(new Point(0, 1)) < Math.Cos(Math.PI / 12))
                    {
                        list_h[0].Add(line_i);    // vertical cos(x) < cos(45deg)
                    }
                    else
                    {
                        if (line_i.CosAlpha(new Point(0, 1)) > 0.999)
                        {
                            list_h[1].Add(line_i);    // horizontal cos(x) > cos(45deg)

                        }

                    }

                    CvInvoke.Line(transcript_horizontal, line.P1, line.P2, new MCvScalar(0, 0, 255), 1);
                }
            }

            var vertical_group = Grouping(list_h[0], true, 25);

            vertical_group = vertical_group.OrderBy(x => x[0].M11.X).ToList();

            CvInvoke.Circle(
                transcript_horizontal,
                new Point(vertical_group[3][0].M11.X, 20),
                20,
                new MCvScalar(0, 255, 0),
                4
            );

            var X_vertical_end = (int)(vertical_group[3][0].M11.X + 25);

            var rect_vertical_i = new Rectangle(
                new Point(vertical_group[2][0].M11.X + 20, line_horizontal[startIndex][0].M11.Y),
                new Size(20, line_horizontal[line_horizontal.Count - 2][0].M11.Y - line_horizontal[startIndex][0].M11.Y - 20)
            );

            var rect_vertical_image = new Image<Gray, byte>(new Mat(source_transcript.Mat, rect_vertical_i).Bitmap);
            rect_vertical_image._GammaCorrect(3.0);

            rect_vertical_image = rect_vertical_image.ThresholdBinaryInv(new Gray(200), new Gray(255));

            var rect_vertical_image_color = new Image<Bgr, byte>(rect_vertical_image.Bitmap);

            var lines_c = rect_vertical_image.HoughLinesBinary(4, Math.PI / 180, 100, 25, 2);

            var max_Y = 0;
            foreach (LineSegment2D[] lineSegment in lines_c)
            {
                foreach (LineSegment2D line in lineSegment)
                {
                    if (line.P1.Y > max_Y) max_Y = line.P1.Y;
                    if (line.P2.Y > max_Y) max_Y = line.P2.Y;
                    CvInvoke.Line(rect_vertical_image_color, line.P1, line.P2, new MCvScalar(0, 255, 0), 2);
                }
            }

            CvInvoke.Circle(
                rect_vertical_image_color,
                new Point(10, max_Y),
                5,
                new MCvScalar(255, 255, 0),
                2
            );

            if (debug) CvInvoke.Imshow("rect_vertical_image", rect_vertical_image_color.Resize(0.5, Inter.Linear));

            CvInvoke.Circle(
                color_image,
                new Point(X_vertical_end, line_horizontal[startIndex][0].M11.Y),
                15,
                new MCvScalar(255, 255, 0),
                2
            );


            max_Y -= 10;

            var max_index = line_horizontal.Count - 2;
            for (var i = line_horizontal.Count - 1; i > 0; i--)
            {
                if (line_horizontal[i][0].M11.Y < (line_horizontal[startIndex][0].M11.Y + max_Y))
                {
                    max_index = i;
                    break;
                }
            }

            var rect_transcript = new Rectangle(
                new Point(0, line_horizontal[startIndex][0].M11.Y - 3),
                new Size(X_vertical_end, line_horizontal[max_index][0].M11.Y - line_horizontal[startIndex][0].M11.Y + 6)
            );

            var rect_transcript_draw = new Rectangle(
                new Point(add_x, line_horizontal[startIndex][0].M11.Y),
                new Size(X_vertical_end, line_horizontal[max_index][0].M11.Y - line_horizontal[startIndex][0].M11.Y)
            );

            horizontal_Iline = new List<ILine>();
            vertical_Iline = new List<ILine>();

            for (var i = startIndex; i <= max_index; i++)
            {
                horizontal_Iline.Add(
                    new ILine(
                        new Point(0, line_horizontal[i][0].M1.Y - line_horizontal[startIndex][0].M1.Y),
                        new Point(rect_transcript.Width - 1, line_horizontal[i][0].M2.Y - line_horizontal[startIndex][0].M1.Y)
                    )
                );
            }

            for (var i = 0; i < 4; i++)
            {
                vertical_Iline.Add(
                    new ILine(
                        new Point(vertical_group[i][0].M11.X + 30, 0),
                        new Point(vertical_group[i][0].M11.X + 30, -rect_transcript.Size.Height + 1)
                    )
                );
            }
            
            transcript = (new Mat(
                source_transcript.Mat,
                rect_transcript
            )).ToImage<Bgr, byte>();

            if (debug) CvInvoke.Imshow("cuter", transcript_horizontal);
            if (debug) CvInvoke.Imshow("source", color_image.Resize(0.5, Inter.Linear));

            CvInvoke.Rectangle(
                source_draw,
                rect_transcript_draw,
                new MCvScalar(0, 0, 255),
                3
            );

            CvInvoke.Rectangle(
                source_draw,
                new Rectangle(
                    new Point(rect_transcript_draw.Left, rect_transcript_draw.Top - 40),
                    new Size(150, 38)
                ),
                new MCvScalar(0, 0, 0),
                -1
            );
            CvInvoke.Rectangle(
                source_draw,
                new Rectangle(
                    new Point(rect_transcript_draw.Left, rect_transcript_draw.Top - 40),
                    new Size(150, 38)
                ),
                new MCvScalar(0, 0, 255),
                2
            );

            CvInvoke.PutText(
                source_draw,
                "Transcript",
                new Point(rect_transcript_draw.Left + 5, rect_transcript_draw.Top - 14),
                FontFace.HersheySimplex,
                0.8,
                new MCvScalar(0, 255, 0),
                2
            );

            return source_draw;
        }

        private static List<Subject> DetectSubject(Image<Gray, byte> transcript, List<ILine> vertical_line)
        {
            var subjects = new List<Subject>();

            var rect_1 = new Rectangle(
                new Point(vertical_line[vertical_line.Count - 3].M11.X + 4, 0),
                new Size(transcript.Size.Width - vertical_line[vertical_line.Count - 3].M11.X - 6, transcript.Size.Height)
            );

            var image_rect_1 = (new Mat(transcript.Mat, rect_1)).ToImage<Gray, byte>().ThresholdBinaryInv(new Gray(200), new Gray(255));

            var colorimage = new Image<Bgr, byte>(transcript.Bitmap);

            var lines = image_rect_1.Canny(10.0, 255.0).HoughLinesBinary(1, Math.PI / 180, 10, 25, 2);

            var list = new List<List<ILine>>{
                        new List<ILine>(),  // vertical
                        new List<ILine>()   // horizontal
                    };

            foreach (LineSegment2D[] lineSegment in lines)
            {
                foreach (LineSegment2D line in lineSegment)
                {
                    var line_i = new ILine(new Point(line.P1.X, -line.P1.Y), new Point(line.P2.X, -line.P2.Y));

                    if (line_i.CosAlpha(new Point(0, 1)) < Math.Cos(Math.PI / 3))
                    {
                        list[0].Add(line_i);    // vertical cos(x) < cos(45deg)
                    }
                    else
                    {
                        if (line_i.CosAlpha(new Point(0, 1)) == 1) list[1].Add(line_i);    // horizontal cos(x) > cos(45deg)
                    }
                }
            }

            var horizontal_group = Grouping(list[1], false, 15);

            var horizontal_line = new List<ILine>();

            for (var i = 0; i < horizontal_group.Count; i++)
            {
                horizontal_line.Add(new ILine(new Point(0, horizontal_group[i][0].M1.Y), new Point(transcript.Size.Width - 1, horizontal_group[i][0].M1.Y)));
            }

            horizontal_line = horizontal_line.OrderBy(x => x.M11.Y).ToList();

            var is_new_form = false;

            if((horizontal_line[horizontal_line.Count - 1].M11.Y - horizontal_line[horizontal_line.Count - 3].M11.Y) < 100)
            {
                var point_t = horizontal_line[horizontal_line.Count - 3].FindIntersection(new ILine(new Point(0, 0), new Point(0, -1)));
                var point_b = horizontal_line[horizontal_line.Count - 1].FindIntersection(vertical_line[vertical_line.Count - 4]);
                var image_check = (
                    new Mat(
                        transcript.Mat,
                        new Rectangle(
                            new Point(point_t.X + 10, -point_t.Y + 8),
                            new Size(point_b.X - point_t.X - 20, -point_b.Y + point_t.Y - 16)
                        )
                    )
                ).ToImage<Gray, byte>().ThresholdBinaryInv(new Gray(180), new Gray(255));

                if (CvInvoke.CountNonZero(image_check) * 1.0 / (image_check.Height * image_check.Width * 1.0) > 0.08) is_new_form = true;

            }

            var max_line_v = new ILine(new Point(transcript.Size.Width - 1, 0), new Point(transcript.Size.Width - 1, transcript.Size.Height - 1));
            var delta_x = 4;
            var delta_y = 2;

            string[] subject_name = {"toan", "vat_li", "hoa_hoc", "sinh_hoc", "tin_hoc", "ngu_van", "lich_su", "dia_ly", "tieng_anh"};

            for (var i = 1; i < 10; i++)
            {
                // HK I
                var point_1 = horizontal_line[i - 1].FindIntersection(vertical_line[vertical_line.Count - 4]);
                var point_2 = horizontal_line[i].FindIntersection(vertical_line[vertical_line.Count - 3]);

                var image_s1 = (new Mat(transcript.Mat, new Rectangle(new Point(point_1.X + delta_x, -point_1.Y + delta_y), new Size(point_2.X - point_1.X - 2 * delta_x, -point_2.Y + point_1.Y - 2 * delta_y)))).ToImage<Gray, byte>();

                // HK II
                var point_3 = horizontal_line[i - 1].FindIntersection(vertical_line[vertical_line.Count - 3]);
                var point_4 = horizontal_line[i].FindIntersection(vertical_line[vertical_line.Count - 2]);

                var image_s2 = (new Mat(transcript.Mat, new Rectangle(new Point(point_3.X + delta_x, -point_3.Y + delta_y), new Size(point_4.X - point_3.X - 2 * delta_x, -point_4.Y + point_3.Y - 2 * delta_y)))).ToImage<Gray, byte>();

                // CN
                var point_5 = horizontal_line[i - 1].FindIntersection(vertical_line[vertical_line.Count - 2]);
                var point_6 = horizontal_line[i].FindIntersection(max_line_v);

                var image_avg = (new Mat(transcript.Mat, new Rectangle(new Point(point_5.X + delta_x, -point_5.Y + delta_y), new Size(point_6.X - point_5.X - 2 * delta_x, -point_6.Y + point_5.Y - 2 * delta_y)))).ToImage<Gray, byte>();

                var subject = new Subject(subject_name[i - 1], 0.0, image_s1, image_s2, image_avg);

                subjects.Add(subject);
            }

            if(is_new_form)
            {
                string[] new_form_subject_name = { "GDCD", "cong_nghe", "the_duc", "QPAN"};
                for(var i = 10; i < 14; i++)
                {
                    // HK I
                    var point_1 = horizontal_line[i - 1].FindIntersection(vertical_line[vertical_line.Count - 4]);
                    var point_2 = horizontal_line[i].FindIntersection(vertical_line[vertical_line.Count - 3]);

                    var image_s1 = (new Mat(transcript.Mat, new Rectangle(new Point(point_1.X + delta_x, -point_1.Y + delta_y), new Size(point_2.X - point_1.X - 2 * delta_x, -point_2.Y + point_1.Y - 2 * delta_y)))).ToImage<Gray, byte>();

                    // HK II
                    var point_3 = horizontal_line[i - 1].FindIntersection(vertical_line[vertical_line.Count - 3]);
                    var point_4 = horizontal_line[i].FindIntersection(vertical_line[vertical_line.Count - 2]);

                    var image_s2 = (new Mat(transcript.Mat, new Rectangle(new Point(point_3.X + delta_x, -point_3.Y + delta_y), new Size(point_4.X - point_3.X - 2 * delta_x, -point_4.Y + point_3.Y - 2 * delta_y)))).ToImage<Gray, byte>();

                    // CN
                    var point_5 = horizontal_line[i - 1].FindIntersection(vertical_line[vertical_line.Count - 2]);
                    var point_6 = horizontal_line[i].FindIntersection(max_line_v);

                    var image_avg = (new Mat(transcript.Mat, new Rectangle(new Point(point_5.X + delta_x, -point_5.Y + delta_y), new Size(point_6.X - point_5.X - 2 * delta_x, -point_6.Y + point_5.Y - 2 * delta_y)))).ToImage<Gray, byte>();

                    var subject = new Subject(new_form_subject_name[i - 10], 0.0, image_s1, image_s2, image_avg);

                    subjects.Add(subject);
                }
            }
            else
            {
                string[] old_form_subject_name = { "cong_nghe", "QPAN", "the_duc" };
                for (var i = 10; i < 13; i++)
                {
                    // HK I
                    var point_1 = horizontal_line[i - 1].FindIntersection(vertical_line[vertical_line.Count - 4]);
                    var point_2 = horizontal_line[i].FindIntersection(vertical_line[vertical_line.Count - 3]);

                    var image_s1 = (new Mat(transcript.Mat, new Rectangle(new Point(point_1.X + delta_x, -point_1.Y + delta_y), new Size(point_2.X - point_1.X - 2 * delta_x, -point_2.Y + point_1.Y - 2 * delta_y)))).ToImage<Gray, byte>();

                    // HK II
                    var point_3 = horizontal_line[i - 1].FindIntersection(vertical_line[vertical_line.Count - 3]);
                    var point_4 = horizontal_line[i].FindIntersection(vertical_line[vertical_line.Count - 2]);

                    var image_s2 = (new Mat(transcript.Mat, new Rectangle(new Point(point_3.X + delta_x, -point_3.Y + delta_y), new Size(point_4.X - point_3.X - 2 * delta_x, -point_4.Y + point_3.Y - 2 * delta_y)))).ToImage<Gray, byte>();

                    // CN
                    var point_5 = horizontal_line[i - 1].FindIntersection(vertical_line[vertical_line.Count - 2]);
                    var point_6 = horizontal_line[i].FindIntersection(max_line_v);

                    var image_avg = (new Mat(transcript.Mat, new Rectangle(new Point(point_5.X + delta_x, -point_5.Y + delta_y), new Size(point_6.X - point_5.X - 2 * delta_x, -point_6.Y + point_5.Y - 2 * delta_y)))).ToImage<Gray, byte>();

                    var subject = new Subject(old_form_subject_name[i - 10], 0.0, image_s1, image_s2, image_avg);

                    subjects.Add(subject);
                }


            }

            CvInvoke.Imshow("colorimage", colorimage);

            return subjects;
        }
        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <returns></returns>
        private static List<List<ILine>> Grouping(List<ILine> list, bool isVertical,int batchSize)
        {
            var group = new List<List<ILine>>();

            foreach (ILine line in list)
            {
                if (group.Count == 0)
                {
                    group.Add(new List<ILine>{ line, line });
                }
                else
                {
                    bool has = false;
                    foreach(List<ILine> group_i in group)
                    {
                        if(isVertical)
                        {
                            if(Math.Abs(group_i[0].M1.X - line.M1.X) < batchSize)
                            {
                                group_i.Add(line);
                                group_i[0].M1.X = (int)((group_i[0].M1.X + line.M1.X + line.M2.X) / 3);
                                has = true;
                            }
                        }
                        else
                        {
                            if (Math.Abs(group_i[0].M1.Y - line.M1.Y) < batchSize)
                            {
                                group_i.Add(line);
                                group_i[0].M1.Y = (int)((group_i[0].M1.Y + line.M1.Y + line.M2.Y) / 3);
                                has = true;
                            }
                        }
                    }

                    if(!has)
                    {
                        group.Add(new List<ILine> { line, line });
                    }
                }
            }

            for(var i = 0; i < group.Count; i++)
            {
                group[i].Remove(group[i][0]);
                if(isVertical) group[i] = group[i].OrderBy(x => x.M1.X).ToList();
                else group[i] = group[i].OrderBy(x => x.M1.Y).ToList();
            }

            return group;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <returns></returns>
        private static List<List<ILine>> ClassifyVector(List<ILine> list)
        {
            var group = new List<List<ILine>>();
            
            foreach( ILine line in list)
            {
                if(group.Count == 0) { group.Add( new List<ILine> { line }); }
                else
                {
                    var has = false;
                    foreach( List<ILine> group_i in group)
                    {
                        if(Math.Abs(group_i[0].CosAlpha(line.n)) > 0.992)
                        {
                            group_i.Add(line);
                            has = true;
                        }
                    }
                    if(!has)
                    {
                        group.Add(new List<ILine> { line });
                    }
                }
            }

            group = group.OrderByDescending(x => x.Count).ToList();

            return group;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <returns></returns>
        private static List<int> GroupingDistance(List<ILine> list, int batchSize)
        {
            var startIndex = 0;
            var endIndex = 0;
            var step = 0.0;
            var count = 0;

            var currentStartIndex = 0;
            var currentEndIndex = 0;
            var currentStep = 0.0;
            var currentCount = 0;

            for (var i = 0; i < list.Count - 1; i++)
            {
                if (i == 0 || Math.Abs(currentStep - list[i].Distances(list[i + 1])) < batchSize)
                {
                    if (i == 0) currentStep = list[i].Distances(list[i + 1]);
                    currentCount++;
                    currentEndIndex++;
                    if (currentCount > count)
                    {
                        startIndex = currentStartIndex;
                        endIndex = currentEndIndex;
                        count = currentCount;
                        step = currentStep;
                    }
                }
                else
                {
                    currentStartIndex = i;
                    currentEndIndex = i;
                    currentCount = 0;
                    currentStep = list[i].Distances(list[i + 1]);
                }
            }
            endIndex++;

            var index = new List<int> { startIndex, endIndex};
            return index;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        private static Image<Gray, byte> Skelatanize(Image<Gray, byte> image)
        {
            Image<Gray, byte> imgOld    = new Image<Gray, byte>(image.Bitmap);
            Image<Gray, byte> img2      = (new Image<Gray, byte>(imgOld.Width, imgOld.Height, new Gray(255))).Sub(imgOld);
            Image<Gray, byte> eroded    = new Image<Gray, byte>(img2.Size);
            Image<Gray, byte> temp      = new Image<Gray, byte>(img2.Size);
            Image<Gray, byte> skel      = new Image<Gray, byte>(img2.Size);
            skel.SetValue(0);
            CvInvoke.Threshold(img2, img2, 127, 256, 0);
            var element = CvInvoke.GetStructuringElement(ElementShape.Cross, new Size(3, 3), new Point(-1, -1));
            bool done = false;

            while (!done)
            {
                CvInvoke.Erode(img2, eroded, element, new Point(-1, -1), 1, BorderType.Reflect, default(MCvScalar));
                CvInvoke.Dilate(eroded, temp, element, new Point(-1, -1), 1, BorderType.Reflect, default(MCvScalar));
                CvInvoke.Subtract(img2, temp, temp);
                CvInvoke.BitwiseOr(skel, temp, skel);
                eroded.CopyTo(img2);
                if (CvInvoke.CountNonZero(img2) == 0) done = true;
            }
            return skel;
        }


        public static void DetectDigit(Image<Gray,byte> source)
        {
            

            var bin = (new Mat(source.Mat, new Rectangle(new Point(4, 4), new Size(source.Width - 8, source.Height - 4)))).ToImage<Gray, byte>().ThresholdBinaryInv(new Gray(200), new Gray(255));

            var x1 = 0;
            var hist = new List<int>();
            for(var i = 0; i < bin.Width; i++)
            {
                hist.Add(0);
                for (var j = 0; j < bin.Height; j++)
                {
                    if (bin.Data[j, i, 0] != 0) hist[i]++;
                }
                if (hist[i] > 3 && x1 == 0) x1 = i;
            }

            var x2 = hist.Count - 1;
            for(var i = hist.Count - 1; i > x1; i--)
            {
                if (hist[i] > 3)
                {
                    x2 = i; break;
                }
            }
            try
            {
                if (x1 - 4 != 0 && x2 != hist.Count - 5)
                {
                    var rect_1 = new Rectangle(new Point(x1 - 4, 0), new Size((int)((x2 - x1 + 8) / 2) - 2, bin.Height));
                    var rect_2 = new Rectangle(new Point(x1 + (int)((x2 - x1) / 2) + 2, 0), new Size((int)((x2 - x1 + 8) / 2), bin.Height));

                    if (rect_1.X > 0 && rect_2.X > 0 && rect_1.Right < bin.Width && rect_2.Right < bin.Width && !rect_1.IsEmpty && !rect_2.IsEmpty)
                    {
                        using (var digit_1 = new Mat(bin.Mat, rect_1).ToImage<Gray, byte>().Resize(1.0, Inter.Linear))
                        using (var digit_2 = new Mat(bin.Mat, rect_2).ToImage<Gray, byte>().Resize(1.0, Inter.Linear))
                        {
                            var img_1 = new Mat(digit_1.Mat, new Rectangle(new Point(0, 0), digit_1.Size));
                            var img_2 = new Mat(digit_2.Mat, new Rectangle(new Point(0, 0), digit_2.Size));         

                            CvInvoke.Resize(img_1, img_1, new Size(28, 28));
                            CvInvoke.Resize(img_2, img_2, new Size(28, 28));

                            CvInvoke.Imshow("digit_1", img_1);
                            CvInvoke.Imshow("digit_2", img_2);

                            var ran = new Random();

                            CvInvoke.Imwrite(@"C:\xxxx\" + ran.Next() + "-1.jpg", img_1);
                            CvInvoke.Imwrite(@"C:\xxxx\" + ran.Next() + "-2.jpg", img_2);
                        }
                    }
                }
            } catch(Exception ex)
            {
                var k = 0;
            }
            

            CvInvoke.Line(
                bin,
                new Point(x1 - 4, 0),
                new Point(x1 - 4, bin.Height - 1),
                new MCvScalar(255, 255, 255),
                2    
            );

            CvInvoke.Line(
                bin,
                new Point(x2 + 4, 0),
                new Point(x2 + 4, bin.Height - 1),
                new MCvScalar(255, 255, 255),
                2
            );

            CvInvoke.Imshow("source", bin);

            //CvInvoke.Imshow("histogram", GetImage(hist));
        }

        public static Image<Gray, byte> GetImage(List<int> hist)
        {
            var img = new Emgu.CV.Image<Gray, byte>(400, 200, new Gray(255));
            for (int i = 0; i < hist.Count; i++)
            {
                CvInvoke.Line(img, new Point(i, hist.Max() + 40), new Point(i, hist.Max() + 40 - hist[i]), new MCvScalar(0, 0, 0));
            }
            return img;
        }
    }

}
