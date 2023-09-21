package n.rnu.isetr.tunisiatourism.AllTourDestinations;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;

import n.rnu.isetr.tunisiatourism.R;

import java.util.ArrayList;

public class DestinationsList extends AppCompatActivity {

    RecyclerView recycler;
    LinearLayoutManager manager;
    DestinationsList_Adapter adapter;
    ArrayList<DestinationsList_Model> array;
    ImageView img ;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tour_attractions_list);
        ImageView img = (ImageView) findViewById(R.id.imageView);



        array = new ArrayList<>();
        array.add(new DestinationsList_Model("Chitral", "KPK", R.drawable.chitralvalley));
        array.add(new DestinationsList_Model("Waziristan", "KPK", R.drawable.waziristan));
        array.add(new DestinationsList_Model("Peshawar", "KPK", R.drawable.khaber));
        array.add(new DestinationsList_Model("Kalam", "Swat", R.drawable.kalam));
        array.add(new DestinationsList_Model("Mansehra", "KPK", R.drawable.mansehra));
        array.add(new DestinationsList_Model("shogran", "mansehra", R.drawable.shogran));
        array.add(new DestinationsList_Model("Kumrat", "Swat", R.drawable.kumratvalley));
        array.add(new DestinationsList_Model("Nathia Gali", "Abbottabad", R.drawable.nathiagali));
        array.add(new DestinationsList_Model("Mingora", "Swat", R.drawable.mingora));
        array.add(new DestinationsList_Model("Kaghan", "Mansehra", R.drawable.kaghan));

        adapter = new DestinationsList_Adapter(this, array);

        manager = new LinearLayoutManager(this, LinearLayoutManager.VERTICAL, false);

        recycler = findViewById(R.id.tourattr_recycler);

        recycler.setLayoutManager(manager);
        recycler.setAdapter(adapter);


        img.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                finish();
            }
        });
    }



}