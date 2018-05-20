using System;
using System.Collections.Generic;
using System.Drawing;
using System.Diagnostics;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Flann;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;

namespace TranscriptDetection
{
    public static class FeatureMatching
    {
        public static VectorOfKeyPoint modelKeyPoints { get; set; }
        public static UMat uModelImage { get; set; }
        public static Mat modelDescriptors { get; set; }
        public static Mat modelImage { get; set; }

        public static void Init(Mat ModelImage)
        {
            double hessianThresh = 300;
            uModelImage = ModelImage.GetUMat(AccessType.Read);
            modelDescriptors = new Mat();
            modelImage = ModelImage.Clone();
            KAZE featureDetector = new KAZE();
            modelKeyPoints = new VectorOfKeyPoint();
            modelDescriptors = new Mat();

            featureDetector.DetectAndCompute(uModelImage, null, modelKeyPoints, modelDescriptors, false);
        }

        public static void FindMatch(
            Mat observedImage,
            out long matchTime,
            out VectorOfKeyPoint observedKeyPoints,
            VectorOfVectorOfDMatch matches,
            out Mat mask,
            out Mat homography)
        {
            int k = 2;
            double uniquenessThreshold = 0.80;
            double hessianThresh = 300;

            Stopwatch watch;
            homography = null;

            //modelKeyPoints = new VectorOfKeyPoint();
            observedKeyPoints = new VectorOfKeyPoint();

            using (UMat uObservedImage = observedImage.GetUMat(AccessType.Read))
            {
                KAZE featureDetector = new KAZE();

                watch = Stopwatch.StartNew();

                Mat observedDescriptors = new Mat();
                featureDetector.DetectAndCompute(uObservedImage, null, observedKeyPoints, observedDescriptors, false);

                using (LinearIndexParams ip = new LinearIndexParams())
                using (SearchParams sp = new SearchParams())
                using (DescriptorMatcher matcher = new FlannBasedMatcher(ip, sp))
                {
                    matcher.Add(modelDescriptors);

                    matcher.KnnMatch(observedDescriptors, matches, k, null);
                    mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                    mask.SetTo(new MCvScalar(255));
                    Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                    int nonZeroCount = CvInvoke.CountNonZero(mask);
                    if (nonZeroCount >= 4)
                    {
                        nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, matches, mask, 1.5, 20);
                        if (nonZeroCount >= 4)
                        {
                            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, matches, mask, 2);
                        }
                    }
                    watch.Stop();
                }
                matchTime = watch.ElapsedMilliseconds;
            }
        }

        /// <summary>
        /// Find the model image and observed image, the matched features and homography projection.
        /// </summary>
        /// <param name="modelImage">The model image</param>
        /// <param name="observedImage">The observed image</param>
        /// <param name="matchTime">The output total time for computing the homography matrix.</param>
        /// <returns>The position matched, the matched features and homography projection.</returns>
        public static VectorOfPoint Detect(Mat observedImage, out long matchTime)
        {
            Mat homography;
            VectorOfKeyPoint observedKeyPoints;

            using (VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch())
            {
                Mat mask;
                FindMatch(observedImage, out matchTime, out observedKeyPoints, matches, out mask, out homography);

                if (homography != null)
                {
                    Rectangle rect = new Rectangle(Point.Empty, modelImage.Size);
                    PointF[] pts = new PointF[]
                    {
                        new PointF(rect.Left, rect.Bottom),
                        new PointF(rect.Right, rect.Bottom),
                        new PointF(rect.Right, rect.Top),
                        new PointF(rect.Left, rect.Top)
                    };

                    pts = CvInvoke.PerspectiveTransform(pts, homography);

                    Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);

                    return new VectorOfPoint(points);
                }
                else
                {
                    return new VectorOfPoint();
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observedKeyPoints"></param>
        /// <param name="modelKeyPoint"></param>
        /// <returns></returns>
        public static VectorOfPoint Matching(VectorOfKeyPoint observedKeyPoints, VectorOfKeyPoint modelKeyPoints, Mat observedDescriptors, Mat modelDescriptors, Size modelSize)
        {
            KAZE featureDetector = new KAZE();

            using (var ip       = new LinearIndexParams())
            using (var sp       = new SearchParams())
            using (var matcher  = new FlannBasedMatcher(ip, sp))
            using (var matches  = new VectorOfVectorOfDMatch())   
            {
                Mat homography = new Mat();
                matcher.Add(modelDescriptors);

                matcher.KnnMatch(observedDescriptors, matches, 2, null);
                var mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                mask.SetTo(new MCvScalar(255));
                Features2DToolbox.VoteForUniqueness(matches, 0.80, mask);

                int nonZeroCount = CvInvoke.CountNonZero(mask);
                if (nonZeroCount >= 4)
                {
                    nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, matches, mask, 1.5, 20);
                    if (nonZeroCount >= 4)
                    {
                        homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, matches, mask, 2);
                    }
                }

                if (homography != null && !homography.Size.IsEmpty)
                {
                    Rectangle rect = new Rectangle(Point.Empty, modelSize);
                    PointF[] pts = new PointF[]
                    {
                        new PointF(rect.Left, rect.Bottom),
                        new PointF(rect.Right, rect.Bottom),
                        new PointF(rect.Right, rect.Top),
                        new PointF(rect.Left, rect.Top)
                    };

                    pts = CvInvoke.PerspectiveTransform(pts, homography);

                    Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);

                    return new VectorOfPoint(points);
                }
                else
                {
                    return new VectorOfPoint();
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static VectorOfKeyPoint FindFeature(Mat source, out Mat modelDescriptors)
        {
            var vectorOfKeyPoint = new VectorOfKeyPoint();
            modelDescriptors = new Mat();
            using (UMat uImage = source.GetUMat(AccessType.Read))
            {
                var kaze = new KAZE();
                kaze.DetectAndCompute(uImage, null, vectorOfKeyPoint, modelDescriptors, false);
            }
            
            return vectorOfKeyPoint;
        }
    }
}
