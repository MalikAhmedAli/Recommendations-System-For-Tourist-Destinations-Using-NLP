package n.rnu.isetr.tunisiatourism;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.WindowManager;
import android.widget.TextView;

import com.google.android.material.imageview.ShapeableImageView;

public class ExploreActivity extends AppCompatActivity {

    ShapeableImageView mainimage;
    TextView description;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_explore);

        getWindow().setFlags(
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS
        );

         description=findViewById(R.id.tunisia_description);

         description.setText("Khyber Pakhtunkhwa is a region of great natural beauty, with rugged mountain ranges, lush green valleys, and picturesque rivers. It is home to some of the highest peaks in the world, including K2, the second-highest mountain on earth. The province also boasts a rich cultural heritage, with a long history of Buddhist, Hindu, and Islamic influences..\n" +
                "\n" +
                "Tourism is an important industry in Khyber Pakhtunkhwa, with visitors from all over the world drawn to its stunning landscapes, historic sites, and vibrant culture. Swat Valley: Known as the Switzerland of Pakistan," );
                
    }
}
