import com.sun.javafx.geom.Curve;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.io.File;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class CVScanApplication {

    public static void main(String[] args){
        CVScanApplication application = new CVScanApplication();
        application.run();
    }

    private void run() {
        ClassLoader classLoader = getClass().getClassLoader();
        transformBill(6);
    }

    private void transformBill(int i) {
        File file = new File("C:/Users/schneiderl/Downloads/bill"+i+".jpg");
        IplImage img = cvLoadImage(file.getAbsolutePath());
        IplImage cannyImg = applyCannySquareEdgeDetectionOnImage(img, 20);
        cvSaveImage("bill"+i+"_stage1.jpg", cannyImg);
        CvSeq contour = findLargestSquareOnCannyDetectedImage(cannyImg);
        img = applyPerspectiveTransformThresholdOnOriginalImage(img, contour, 30);
        img = cleanImage(img);
        cvSaveImage("bill"+i+"_stage2.jpg", img);
    }

    private IplImage downScaleImage(IplImage img, int percent) {
        IplImage destImage = cvCreateImage(cvSize((img.width()*percent)/100, (img.height()*percent)/100) , img.depth(), img.nChannels());
        cvResize(img, destImage);
        return destImage;
    }

    private IplImage applyCannySquareEdgeDetectionOnImage(IplImage img, int percent){
        IplImage destImage = downScaleImage(img, percent);
        IplImage grayImage = cvCreateImage(cvGetSize(destImage), IPL_DEPTH_8U, 1);
        cvCvtColor(destImage, grayImage, CV_BGR2GRAY);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        Frame grayImageFrame = converter.convert(grayImage);
        Mat grMat = converter.convert(grayImageFrame);
        GaussianBlur(grMat, grMat, new Size(5,5), 0.0, 0.0, BORDER_DEFAULT);
        destImage = converter.convertToIplImage(grayImageFrame);
        cvErode(destImage, destImage);
        cvDilate(destImage, destImage);
        cvCanny(destImage, destImage, 75.0, 200.0);
        return destImage;
    }

    private CvSeq findLargestSquareOnCannyDetectedImage(IplImage cannyEdgeDetectedIMage){
        IplImage foundedContoursImage = cvCloneImage(cannyEdgeDetectedIMage);
        CvMemStorage memory = CvMemStorage.create();
        CvSeq contours = new CvSeq();
        cvFindContours(foundedContoursImage, memory, contours, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
        int maxWidth = 0;
        int maxHeight = 0;
        CvRect contour;
        CvSeq seqFounded = null;
        CvSeq nextSeq;
        for(nextSeq = contours; nextSeq != null; nextSeq = nextSeq.h_next()){
            contour = cvBoundingRect(nextSeq, 0);
            if(contour.width() >= maxWidth && contour.height() >= maxHeight){
                maxWidth = contour.width();
                maxHeight = contour.height();
                seqFounded = nextSeq;
            }
        }
        CvSeq result = cvApproxPoly(seqFounded, Loader.sizeof(CvContour.class), memory, CV_POLY_APPROX_DP, cvContourPerimeter(seqFounded) * 0.02, 0);
        for(int i = 0; i< result.total(); i++){
            CvPoint v = new CvPoint(cvGetSeqElem(result, i));
            cvDrawCircle(foundedContoursImage, v, 5, CvScalar.BLUE, 20, 8, 0);
        }
        return result;
    }

    private IplImage applyPerspectiveTransformThresholdOnOriginalImage(IplImage srcImage, CvSeq contour, int percent){
        IplImage warpImage = cvCloneImage(srcImage);
        for(int i=0; i < contour.total(); i++){
            CvPoint point = new CvPoint(cvGetSeqElem(contour, i));
            point.x((int) (point.x()*100)/percent);
            point.y((int) (point.y()*100)/percent);
        }
        CvPoint topRightPoint = new CvPoint(cvGetSeqElem(contour, 0));
        CvPoint topLeftPoint = new CvPoint(cvGetSeqElem(contour, 1));
        CvPoint bottomLeftPoint = new CvPoint(cvGetSeqElem(contour, 2));
        CvPoint bottomRightPoint = new CvPoint(cvGetSeqElem(contour, 3));
        int resultWidth = topRightPoint.x() - topLeftPoint.x();
        int bottomWidth = bottomRightPoint.x() -bottomLeftPoint.x();
        if(bottomWidth > resultWidth)
            resultWidth = bottomWidth;
        int resultHeight = (int) (bottomLeftPoint.y() - topLeftPoint.y());
        int bottomHeight = (int) (bottomRightPoint.y() - topRightPoint.y());
        if(bottomHeight > resultHeight)
            resultHeight = bottomHeight;

        float[] sourcePoints = {topLeftPoint.x(), topLeftPoint.y(), topRightPoint.x(), topRightPoint.y(), bottomLeftPoint.x(), bottomLeftPoint.y(), bottomRightPoint.x(), bottomRightPoint.y()};
        float[] destinFloats = {0, 0, resultWidth, 0, 0, resultHeight, resultWidth, resultHeight};

        CvMat homography = cvCreateMat(3,3, CV_32FC1);
        cvGetPerspectiveTransform(sourcePoints, destinFloats, homography);
        IplImage destImage = cvCloneImage(warpImage);
        cvWarpPerspective(warpImage, destImage, homography, CV_INTER_LINEAR, CvScalar.ZERO);
        return  cropImage(destImage, 0, 0, resultWidth, resultHeight);
    }

    private IplImage cropImage(IplImage srcImage, int fromX, int fromY, int toWidth, int toHeight) {
        cvSetImageROI(srcImage, cvRect(fromX, fromY, toWidth, toHeight));
        IplImage destImage = cvCloneImage(srcImage);
        cvCopy(srcImage, destImage);
        return destImage;
    }

    private IplImage cleanImage(IplImage srcImage){
        IplImage destImage = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_8U, 1);
        cvCvtColor(srcImage, destImage, CV_BGR2GRAY);
        cvSmooth(destImage, destImage, CV_MEDIAN, 3, 0, 0, 0);
        cvThreshold(destImage, destImage, 0, 255,  CV_THRESH_OTSU);
        return destImage;
    }


}
