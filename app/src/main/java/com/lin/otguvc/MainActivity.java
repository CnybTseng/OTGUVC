package com.lin.otguvc;

import android.app.PendingIntent;
import android.app.ProgressDialog;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Color;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.hardware.usb.UsbManager;
import android.media.Ringtone;
import android.media.RingtoneManager;
import android.net.Uri;
import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;

import java.io.DataOutputStream;
import java.io.IOException;

import static java.lang.Thread.sleep;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native' library on application startup.
    static {
        System.loadLibrary("native");
    }
    private static final int OPEN_COMPLETE = 1001;
    private static final int FPS_REFRESH = 1002;
    private static final String TAG = "USB_DEVICE";
    private static boolean captureOpened = false;
    private static final String USER_UVC_PERMISSION = "chmod 777 /dev/video*";
    private static final String USER_USB_PERMISSION = "chmod 777 -R /dev/bus/usb/*";
    private static String USB_FS_NAME = "/dev/bus/usb";
    private static int USB_BUS_NUM = 0;
    private static int USB_DEV_NUM = 0;
    private static int USB_VENDOR_ID = 0;
    private static int USB_PRODUCT_ID = 0;
    private static int USB_BUS_FD = 0;

    private static int UVC_WIDTH = 640;
    private static int UVC_HEIGHT = 480;
    private static int UVC_MIN_FPS = 1;
    private static int UVC_MAX_FPS = 40;

    private static UsbManager mUsbManager;
    private static UsbDeviceConnection mConnection;
    private static final String ACTION_USB_PERMISSION_BASE = "com.lin.otguvc.USB_PERMISSION.";
    private final String ACTION_USB_PERMISSION = ACTION_USB_PERMISSION_BASE + hashCode();

    LinearLayout screenLinearLayout;

    SeekBar threshSeekBar;
    private SeekBar.OnSeekBarChangeListener threshSeekBarChangeListener;
    SurfaceView surfaceView;
    TextView fpsTextView;
    TextView thresholdTextView;
    ProgressDialog openProgressDialog;
    Button ocButton;

    private Handler msgHandler;
    Thread fpsThread;
    public int refreshFPS = 0;
    public boolean refreshFlag = true;

    Thread detectThread;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //suSystemShell(USER_USB_PERMISSION);
        //suSystemShell(USER_UVC_PERMISSION);
        ocButton = findViewById(R.id.oc_button);;

        threshSeekBar = (SeekBar) findViewById(R.id.seekbar_thresh);
        threshSeekBar.setMax(100);
        threshSeekBar.setProgress(50);
        threshSeekBarChangeListener = new SeekBar.OnSeekBarChangeListener() {

            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                //设置文本框的值
                thresholdTextView.setText(""+(progress/100.0));
                setAIThresholdFromJNI((float)(progress/100.0));
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        };
        threshSeekBar.setOnSeekBarChangeListener(threshSeekBarChangeListener);

        thresholdTextView = (TextView)findViewById(R.id.text_threshold);
        fpsTextView = (TextView)findViewById(R.id.text_fps);


        surfaceView = findViewById(R.id.displaySurfaceView);
        screenLinearLayout = (LinearLayout) findViewById(R.id.screen_layout);

        surfaceView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {
            }
            @Override
            public void surfaceChanged(final SurfaceHolder holder, int format, int width, int height) {
                Thread thread = new Thread(){
                    @Override
                    public void run() {
                        super.run();
                        setSurfaceviewFromJNI(holder.getSurface(),surfaceView.getWidth(),surfaceView.getHeight());
                    }
                };
                thread.start();
            }
            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {
            }
        });

        msgHandler = new Handler(){
            @Override
            public void handleMessage(Message msg) {
                super.handleMessage(msg);
                switch (msg.what) {
                    case OPEN_COMPLETE:
                        if(captureOpened){
                            ocButton.setText("Close");
                        }
                        else {
                            ocButton.setText("Open");
                        }
                        break;
                    case FPS_REFRESH:
                        fpsTextView.setText("" + refreshFPS);
                    default:
                        break;
                }
            }
        };

        fpsThread = new Thread(new Runnable() {
            @Override
            public void run() {
                while (refreshFlag) {
                    try {
                        sleep(1000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    refreshFPS = getFpsFromJNI();
                    Message msg = Message.obtain();
                    msg.what = FPS_REFRESH;
                    msgHandler.sendMessage(msg);
                }
            }
        });
        fpsThread.start();

        detectThread = new Thread(new Runnable() {
            @Override
            public void run() {
                int detect = 0;
                while (refreshFlag) {
                    try {
                        sleep(50);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    detect = getDetectFromJNI();
                    if(detect > 0){
                        playSound();
                        try {
                            sleep(3000);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        });
        detectThread.start();
    }

    public void captureOCClick(View view)
    {
        //playSound();
        /*
        converScreen();
        new Thread(new Runnable() {
            @Override
            public void run() {
                try
                {
                    sleep(5000);
                } catch (InterruptedException e)
                {
                    // TODO 自动生成的 catch 块
                    e.printStackTrace();
                }
                disconverScreen();

            }
        }).start();
        */
        if(!tryGetUsbPermission()){
            return;
        }

        if(captureOpened){
            destroyCaptureFromJNI();
            captureOpened = false;
            ocButton.setText("Open");
        }else{
            converScreen();
            new Thread(new Runnable() {
                @Override
                public void run() {
                    int ret;
                    ret = createCaptureFromJNI(USB_VENDOR_ID,USB_PRODUCT_ID,USB_BUS_FD,USB_BUS_NUM,USB_DEV_NUM,UVC_WIDTH,UVC_HEIGHT,UVC_MIN_FPS,UVC_MAX_FPS,surfaceView.getWidth(),surfaceView.getHeight());
                    if(ret < 0){
                        captureOpened = false;
                    }else{
                        captureOpened = true;
                    }
                    disconverScreen();
                    Message msg =Message.obtain();
                    msg.what = OPEN_COMPLETE;
                    msgHandler.sendMessage(msg);
                }
            }).start();
        }
    }

    private boolean tryGetUsbPermission()
    {
        String deviceName;
        String busList[];
        String tempBus;
        String numList[];
        mUsbManager = (UsbManager) getSystemService(Context.USB_SERVICE);

        PendingIntent mPermissionIntent = PendingIntent.getBroadcast(this, 0, new Intent(ACTION_USB_PERMISSION), 0);

        for (final UsbDevice usbDevice : mUsbManager.getDeviceList().values()) {
            try {
                mUsbManager.requestPermission(usbDevice, mPermissionIntent);
            } catch (final Exception e) {
                Log.w("requestPermission", e);
            }
            if(mUsbManager.hasPermission(usbDevice)){
                mConnection = mUsbManager.openDevice(usbDevice);
                if (mConnection == null) {
                    return false;
                }else {
                    USB_VENDOR_ID = usbDevice.getVendorId();
                    USB_PRODUCT_ID = usbDevice.getProductId();
                    USB_BUS_FD = mConnection.getFileDescriptor();
                    deviceName = usbDevice.getDeviceName();
                    busList = deviceName.split(USB_FS_NAME);
                    if(busList.length >= 2) {
                        tempBus = busList[1];
                        numList = tempBus.split("/");
                        if (numList.length >= 3) {
                            USB_BUS_NUM = Integer.parseInt(numList[1]);
                            USB_DEV_NUM = Integer.parseInt(numList[2]);
                        }
                    }
                    Log.w(TAG, "VendorId:"+USB_VENDOR_ID);
                    Log.w(TAG, "ProductId:"+USB_PRODUCT_ID);
                    Log.w(TAG, "fileDescriptor:"+USB_BUS_FD);
                    Log.w(TAG, "bus_num:"+USB_BUS_NUM);
                    Log.w(TAG, "dev_num:"+USB_DEV_NUM);
                    Log.w(TAG, "fs_name:"+USB_FS_NAME);
                    return true;
                }
            }else{
                return false;
            }
        }
        return false;
    }

    public void converScreen()
    {
        openProgressDialog = ProgressDialog.show(this,"Open", "Opening...",true,false);
    }

    public void disconverScreen()
    {
        openProgressDialog.dismiss();
    }

    public boolean suSystemShell(String cmd){
        Process process = null;
        DataOutputStream os = null;
        Runtime runtime = Runtime.getRuntime();
        try {
            process = runtime.exec("su");
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        try{
            os = new DataOutputStream(process.getOutputStream());

            os.writeBytes(cmd+ "\n");
            os.flush();

            os.writeBytes("exit\n");
            os.flush();
            process.waitFor();
        } catch (Exception e) {
            return false;
        } finally {
            try {
                if (os != null)   {
                    os.close();
                }
                process.destroy();
            } catch (Exception e) {
            }
        }
        return true;
    }

    public void playSound(){
        Uri uri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION);
        Ringtone rt = RingtoneManager.getRingtone(getApplicationContext(), uri);
        rt.play();
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native int createCaptureFromJNI(int vid, int pid, int fd, int busnum, int devnum,int width,int height,int minfps,int maxfps,int surfaceWidth,int surfaceHeight);
    public native int destroyCaptureFromJNI();
    public native int setSurfaceviewFromJNI(Surface surface,int screenWidth,int screenHeight);
    public native int setAIThresholdFromJNI(float threshold);
    public native int getFpsFromJNI();
    public native int getDetectFromJNI();
}
