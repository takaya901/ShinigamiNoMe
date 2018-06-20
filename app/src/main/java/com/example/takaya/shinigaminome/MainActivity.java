package com.example.takaya.shinigaminome;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Random;

public class MainActivity extends AppCompatActivity
        implements CameraBridgeViewBase.CvCameraViewListener, ActivityCompat.OnRequestPermissionsResultCallback
{
    private static final String TAG = "OCVSample::Activity";

    static final int REQUEST_CODE = 1;
    private CascadeClassifier faceDetector;
    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;
    private Size minFaceSize = new Size(0, 0);
    private Mat internalImg;
    Bitmap src;
    Random rand = new Random();
    HashMap<Rect, String> faceMap = new HashMap<>();
    ArrayList<Rect> faceHist = new ArrayList<>();

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this)
    {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        ActivityCompat.requestPermissions(this, new String[]{
                Manifest.permission.CAMERA
        }, REQUEST_CODE);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        // 顔認識を行うカスケード分類器インスタンスの生成（一度ファイルを書き出してファイルのパスを取得する）
        // 一度raw配下に格納されたxmlファイルを取得
        try{
            InputStream inStream = this.getResources().openRawResource(R.raw.haarcascade_frontalface_default);
            File cascadeDir = this.getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");
            // 取得したxmlファイルを特定ディレクトリに出力
            FileOutputStream outStream = new FileOutputStream(cascadeFile);
            byte[] buf = new byte[2048];
            int rdBytes;
            while ((rdBytes = inStream.read(buf)) != -1) {
                outStream.write(buf, 0, rdBytes);
            }
            outStream.close();
            inStream.close();
            // 出力したxmlファイルのパスをCascadeClassifierの引数にする
            faceDetector = new CascadeClassifier(cascadeFile.getAbsolutePath());
            // CascadeClassifierインスタンスができたら出力したファイルはいらないので削除
            if (faceDetector.empty()) {
                faceDetector = null;
            } else {
                cascadeDir.delete();
                cascadeFile.delete();
            }
        }
        catch (Exception ex){
            Log.d("aaa", ex.getMessage());
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case REQUEST_CODE: {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // パーミッションが必要な処理
                } else {
                    // パーミッションが得られなかった時
                    // 処理を中断する・エラーメッセージを出す・アプリケーションを終了する等
                }
            }
        }
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(Mat inputFrame)
    {
        Mat src = inputFrame;
        Mat detected = src.clone();

        try{
            Imgproc.cvtColor(src, src, Imgproc.COLOR_RGB2GRAY);
            Imgproc.cvtColor(detected, detected, Imgproc.COLOR_RGBA2RGB);
            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(src, faces, 1.1, 2, 2, minFaceSize, new Size());
            Rect[] facesArray = faces.toArray();

            for(Rect face : facesArray){
                String lifeSpan = "";

                for(Rect faceHist : faceHist){
                    if (Math.abs(faceHist.x - face.x) < 100){
                        lifeSpan = faceMap.get(faceHist);
                        break;
                    }
                }
                if (lifeSpan.isEmpty()){
                    lifeSpan = getLifespan(detected, face.tl());
                    faceMap.put(face, lifeSpan);
                    faceHist.add(face);
                }
                Point org = new Point(face.x - 300, face.y);
                Imgproc.putText(detected, lifeSpan, org, Core.FONT_HERSHEY_SCRIPT_SIMPLEX|Core.FONT_ITALIC, 6.0f, new Scalar(0 ,0 , 255), 10);
            }
            Imgproc.applyColorMap(detected, detected, Imgproc.COLORMAP_WINTER);
        }
        catch (Exception ex){
            Log.i("OpenCV", ex.getMessage());
        }

        return detected;
    }

    private String getLifespan(Mat src, Point org)
    {
        String lifeSpan = "";
        for (int i = 0; i < 6; i++){
            lifeSpan += String.valueOf(rand.nextInt(10));
        }
        return lifeSpan;

    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus && Build.VERSION.SDK_INT >= 19) {
            hideSystemUI();
        }
    }

    private void hideSystemUI() {
        View decorView = getWindow().getDecorView();
        minFaceSize.width = decorView.getWidth() / 8;
        minFaceSize.height = decorView.getHeight() / 8;
        decorView.setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_IMMERSIVE
                        | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_FULLSCREEN
        );
    }
}
