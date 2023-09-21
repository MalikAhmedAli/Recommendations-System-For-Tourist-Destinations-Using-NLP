package n.rnu.isetr.tunisiatourism;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.Toast;

import n.rnu.isetr.tunisiatourism.Dining.DiningList;

public class FestivalsActivity extends AppCompatActivity {

    ListView List;
    String festivalsList[] = {"Summer festivals", "Winter and Fall festivals", "Spring festivals"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_festivals);
        ImageView img = (ImageView) findViewById(R.id.imageView);

        List = (ListView)findViewById(R.id.ListView);
        ArrayAdapter<String> arrayAdapter = new ArrayAdapter<String>(this, R.layout.festivals_list_item ,R.id.textView, festivalsList);
        List.setAdapter(arrayAdapter);

        List.setOnItemClickListener(new AdapterView.OnItemClickListener() {

            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Intent i = new Intent(FestivalsActivity.this, ShowFestivals.class);
                i.putExtra("key",String.valueOf(position));
               startActivity(i);

                }



        });
        img.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                finish();
            }
        });
    }


}
