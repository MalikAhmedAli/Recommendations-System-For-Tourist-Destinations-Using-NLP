package n.rnu.isetr.tunisiatourism;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
 import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.ComponentName;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ViewFlipper;

import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import n.rnu.isetr.tunisiatourism.AllTourDestinations.DestinationsList;
import n.rnu.isetr.tunisiatourism.Dining.DiningList;
import n.rnu.isetr.tunisiatourism.HomeDestinations.Destinations_ADAPTER;
import n.rnu.isetr.tunisiatourism.HomeDestinations.Destinations_MODEL;

import java.util.ArrayList;
import java.util.List;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.net.Uri;



public class MainActivity extends AppCompatActivity implements BottomNavigationView.OnNavigationItemSelectedListener{

    private FirebaseUser user;
    private DatabaseReference reference;
    private String userID;
    ViewFlipper v_flipper;

    RecyclerView  destinations ;

    ArrayList<Destinations_MODEL> destinations_models;
     Destinations_ADAPTER destinations_adapter;
    LinearLayoutManager manager;
    TextView seetouristdestinations,explore,dining,festivals;
    BottomNavigationView bottomNavigationView;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().setFlags(
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS
        );


        int images[]={R.drawable.ribat, R.drawable.hammamet, R.drawable.tunismedina};

        v_flipper=findViewById(R.id.flipper);

        for (int image:images){
            flipperImages(image);}




        user= FirebaseAuth.getInstance().getCurrentUser();
        reference= FirebaseDatabase.getInstance().getReference("Users");
        userID=user.getUid();
      final TextView greetingTextView=(TextView)findViewById(R.id.greeting);
        reference.child(userID).addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                User userData=snapshot.getValue(User.class);

                if(userData!=null){
                    String fullName=userData.fullName;
                   greetingTextView.setText(fullName +"!");
                }

            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {
                Toast.makeText(MainActivity.this,"Something wrong happened!",Toast.LENGTH_LONG).show();
            }
        });

/***********************************************************************************/

        destinations = findViewById(R.id.destinations_recyclerview);

        destinations_models = new ArrayList<>();
        destinations_models.add(new Destinations_MODEL(R.drawable.djem, "K2 Mountains", "Second-Highest Mountain On Earth"));
        destinations_models.add(new Destinations_MODEL(R.drawable.djerba, "SWAT", "Switzerland of Pakistan"));
        destinations_models.add(new Destinations_MODEL(R.drawable.carthage, "Naran Kagan", "Picturesque, Scenic, Serene."));
        destinations_models.add(new Destinations_MODEL(R.drawable.bardo, "The National Bardo Museum", "Tunis, Bardo"));

        destinations_adapter = new Destinations_ADAPTER(this, destinations_models);
        manager = new LinearLayoutManager(this, RecyclerView.HORIZONTAL, false);

        destinations.setAdapter(destinations_adapter);
        destinations.setLayoutManager(manager);



        seetouristdestinations = findViewById(R.id.seealllink);
        explore=findViewById(R.id.discover_llink);
        Button search=findViewById(R.id.btn1);

        dining=findViewById(R.id.dining_link);
        festivals=findViewById(R.id.festivalslink);

        seetouristdestinations.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(MainActivity.this, DestinationsList.class));
            }
        });

         search.setOnClickListener(new View.OnClickListener() {
             Context context;
             @Override
             public void onClick(View view) {
//                 PackageManager packageManager = getPackageManager();
//
//// Get a list of all installed apps
//                 List<ApplicationInfo> installedApps = packageManager.getInstalledApplications(PackageManager.GET_META_DATA);
//
//// Iterate through the list and retrieve the package names
//                 for (ApplicationInfo appInfo : installedApps) {
//                     String packageName = appInfo.packageName;
//                     // Do whatever you want with the package name
//                     Log.d("Package Name", packageName);
//                 }
                 String packageName = "com.fyp.kpktourism"; // Replace with the package name of the target app
                 String activityName = "SearchActivity"; // Replace with the main activity name of the target app

                 Intent intent = getPackageManager().getLaunchIntentForPackage(packageName);
                 if (intent != null) {
//                     intent.setComponent(new ComponentName(packageName, activityName));
//                     intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                     startActivity(intent);
                 } else {
                     Log.i("1","not");

                 }



//                 AppOpener.openApp(getApplicationContext(), "com.example.kpk-tourism");


             }
         });
        explore.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                startActivity(new Intent(MainActivity.this, ExploreActivity.class));

            }
        });

        dining.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(MainActivity.this, DiningList.class));
            }
        });
        festivals.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(MainActivity.this, FestivalsActivity.class));
            }
        });
        /***********************************************************************************/


        bottomNavigationView = findViewById(R.id.bottomNavigationView);

        bottomNavigationView.setOnNavigationItemSelectedListener(this);


    }



    public static class AppOpener {
        public static void openApp(Context context, String packageName) {
            PackageManager packageManager = context.getPackageManager();

        }
    }



    public  void open(View view){
        Log.i("e", "open");

        Intent launchIntent = getPackageManager().getLaunchIntentForPackage("kpk-tourisum");
        if (launchIntent != null) {
            startActivity(launchIntent);//null pointer check in case package name was not found
        }
//        AppOpener.openApp(getApplicationContext(), "kpk-tourism");


    }
    public void flipperImages(int image){
        ImageView imageView=new ImageView(this);
        imageView.setBackgroundResource(image);

        v_flipper.addView(imageView);
        v_flipper.setFlipInterval(4000);
        v_flipper.setAutoStart(true);

//animation
        v_flipper.setInAnimation(this, android.R.anim.slide_in_left);
        v_flipper.setOutAnimation(this, android.R.anim.slide_out_right);

    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case R.id.logout:
                FirebaseAuth.getInstance().signOut();
                startActivity(new Intent(MainActivity.this,LoginActivity.class));
                return true;
        }
        return false;
    }
}
