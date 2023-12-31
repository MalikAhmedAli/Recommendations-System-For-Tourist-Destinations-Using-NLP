package com.fyp.kpktourism;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.view.WindowManager;

import androidx.appcompat.app.AppCompatActivity;

public class WelcomingActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_welcoming);

        getWindow().setFlags(
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS
        );


        new Handler().postDelayed(new Runnable(){
            @Override
            public void run() {
                //Redirect to login page after 4 seconds
                Intent mainIntent = new Intent(WelcomingActivity.this, SearchActivity.class);
                startActivity(mainIntent);
                finish();
            }
        }, 4000);
    }


}
