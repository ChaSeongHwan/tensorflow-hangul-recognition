package ibm.tf.hangul;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;


public class PaintView extends View {

    // Length in pixels of each dimension for the bitmap displayed on the screen.
    // 화면에 표시되는 비트맵에 대한 길이.
    public static final int BITMAP_DIMENSION = 128;

    // Length in pixels of each dimension for the bitmap to be fed into the model.
    // 모델에 공급되는 비트맵에 대한 길이.
    public static final int FEED_DIMENSION = 64;

    private boolean setup, drawHere;

    private Paint paint;
    private Bitmap bitmap;
    private Path path;
    private Canvas canvas;

    private Matrix transformMat = new Matrix();
    private Matrix inverseTransformMat = new Matrix();

    private PointF pointF = new PointF();

    private ArrayList<PaintPath> PaintPathList = new ArrayList<>();
    private View drawTextView;

    public PaintView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        path = new Path();

        paint = new Paint();
        paint.setAntiAlias(true);
        paint.setColor(Color.WHITE);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStrokeWidth(6);
    }

    /**
     * Remove all strokes and clear the canvas.
     * 모든 획 제거 및 캔버스 초기화.
     */
    public void reset() {
        path.reset();
        if (bitmap != null) {
            bitmap.eraseColor(Color.BLACK);
        }
    }

    public void setDrawText(View view) {
        drawTextView = view;
        drawHere = true;
    }

    /**
     * This function will create the transform matrix for scaling up and centering the bitmap
     * that will represent our character image inside the view. The inverse matrix is also created
     * for mapping touch coordinates to coordinates within the bitmap.
     * 뷰 내부의 캐릭터 이미지를 나타낼 비트맵 확대, 중앙에 배치하기 위한 변환 매트릭스 생성
     * 터치 좌표를 비트맵 내의 좌표에 매핑하기 위해 역행렬도 생성함.
     */
    private void setupScaleMatrices() {

        // View size.
        float width = getWidth();
        float height = getHeight();
        float scaleW = width / BITMAP_DIMENSION;
        float scaleH = height / BITMAP_DIMENSION;

        float scale = scaleW;
        if (scale > scaleH) {
            scale = scaleH;
        }

        // Translation to center bitmap in view after it is scaled up.
        float centerX = BITMAP_DIMENSION * scale / 2;
        float centerY = BITMAP_DIMENSION * scale / 2;
        float dx = width / 2 - centerX;
        float dy = height / 2 - centerY;

        transformMat.setScale(scale, scale);
        transformMat.postTranslate(dx, dy);
        transformMat.invert(inverseTransformMat);
        setup = true;
    }

    /**
     * This gets the coordinates in the bitmap based on the the coordinates of where the
     * user touched.
     * 사용자가 터치한 좌표를 기준으로 비트맵의 좌표를 가져옴.
     */
    public void getBitmapCoords(float x, float y, PointF out) {
        float points[] = new float[2];
        points[0] = x;
        points[1] = y;
        inverseTransformMat.mapPoints(points);
        out.x = points[0];
        out.y = points[1];
    }

    public void onResume() {
        createBitmap();
    }

    public void onPause() {
        releaseBitmap();
    }

    private void createBitmap() {
        if (bitmap != null) {
            bitmap.recycle();
        }
        bitmap = Bitmap.createBitmap(BITMAP_DIMENSION, BITMAP_DIMENSION, Bitmap.Config.ARGB_8888);
        canvas = new Canvas(bitmap);
        reset();
    }

    private void releaseBitmap() {
        if (bitmap != null) {
            bitmap.recycle();
            bitmap = null;
            canvas = null;
        }
        reset();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        // If this is the first time the user has touched the drawable canvas, hide the text that
        // tells the user where to draw.
        if (drawHere) {
            drawTextView.setVisibility(View.INVISIBLE);
            drawHere = false;
        }
        PaintPath paintPath = new PaintPath();
        canvas.drawPath(path, paint);

        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            getBitmapCoords(event.getX(), event.getY(), pointF);
            path.moveTo(pointF.x, pointF.y);
            path.lineTo(pointF.x, pointF.y);
        }
        else if (event.getAction() == MotionEvent.ACTION_MOVE) {
            getBitmapCoords(event.getX(), event.getY(), pointF);
            path.lineTo(pointF.x, pointF.y);
            paintPath.setPath(path);
            paint.setColor(Color.WHITE);
            paintPath.setPaint(paint);
            PaintPathList.add(paintPath);
        }
        invalidate();
        return true;
    }

    @Override
    protected void onDraw(Canvas c) {
        if (!setup) {
            setupScaleMatrices();
        }
        if (bitmap == null) {
            return;
        }

        c.drawBitmap(bitmap, transformMat, paint);
        if (PaintPathList.size() > 0) {
            canvas.drawPath(
                    PaintPathList.get(PaintPathList.size() - 1).getPath(),
                    PaintPathList.get(PaintPathList.size() - 1).getPaint());
        }
    }

    /**
     * This function will convert the bitmap pixels to usable input to our TensorFlow model.
     * 비트맵 픽셀을 Tensorflow 모델에 사용가능한 입력값으로 변환시킴.
     */
    public float[] getPixelData() {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, FEED_DIMENSION,
                                                         FEED_DIMENSION, false);

        int width = FEED_DIMENSION;
        int height = FEED_DIMENSION;

        int[] pixels = new int[width * height];
        resizedBitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] returnPixels = new float[pixels.length];

        // Here we want to convert each pixel to a floating point number between 0.0 and 1.0 with
        // 1.0 being white and 0.0 being black.
        // 각 픽셀의 값을 0과 1의 부동소수점 숫자로 전처리함.
        for (int i = 0; i < pixels.length; ++i) {
            int pix = pixels[i];
            int b = pix & 0xff;
            returnPixels[i] = (float) (b/255.0);
        }
        return returnPixels;
    }

    /**
     * This is an object to encapsulate the path and paint information related to drawing on the
     * bitmap/canvas.
     * 비트맵 / 캔버스에 드로잉과 관련된 경로과 페인트 정보를 캡슐화 하는 객체.
     */
    public class PaintPath {

        Path path;
        Paint paint;

        public Path getPath() {
            return path;
        }

        public void setPath(Path path) {
            this.path = path;
        }

        public Paint getPaint() {
            return paint;
        }

        public void setPaint(Paint paint) {
            this.paint = paint;
        }
    }
}
