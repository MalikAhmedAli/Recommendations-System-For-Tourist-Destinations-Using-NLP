package com.fyp.kpktourism;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;

import com.fyp.kpktourism.databinding.ActivitySearchBinding;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;

import android.os.StrictMode;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TableLayout.LayoutParams;
import android.widget.TextView;

import com.fyp.kpktourism.databinding.ActivitySearchResultBinding;

import java.net.URL;
import java.util.List;

public class SearchResult extends AppCompatActivity {

    private ActivitySearchResultBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivitySearchResultBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        System.out.println("hiiiiiiiiii");

        Button backButton = (Button) findViewById(R.id.button2) ;
        List<SearchResultModel> searchResult = getIntent().getParcelableArrayListExtra("searchResult");

        System.out.println("Yayyyy");
        System.out.println(searchResult);
        backButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Intent mainIntent = new Intent(SearchResult.this, SearchActivity.class);
                startActivity(mainIntent);

            }
        });

        TableLayout tl = (TableLayout) findViewById(R.id.main_table);

        TableRow tr_head = new TableRow(this);
        tr_head.setId(10);
        tr_head.setBackgroundColor(Color.CYAN);
        tr_head.setLayoutParams(new LayoutParams(
                LayoutParams.MATCH_PARENT,
                LayoutParams.WRAP_CONTENT));

        TextView label_hello = new TextView(this);
        label_hello.setId(20);
        label_hello.setText("Suitable Destinations for You");
        label_hello.setTextColor(Color.BLACK);          // part2
        label_hello.setPadding(50, 50, 50, 5);
        label_hello.setTextSize(26);
        tr_head.addView(label_hello);// add the column to the table row here
        tr_head.setGravity(Gravity.CENTER_VERTICAL);
        tr_head.setGravity(Gravity.CENTER_HORIZONTAL);
        tl.addView(tr_head);

        for (SearchResultModel res: searchResult) {
            System.out.println("hereee");

            TableRow row = new TableRow(this);

            row.addView(formatCardView(res));
            tl.addView(row);
        }
    }

    private CardView formatCardView(SearchResultModel sm) {
        CardView cardView = new CardView(this);
        cardView.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.MATCH_PARENT, TableRow.LayoutParams.WRAP_CONTENT));
        cardView.setCardElevation(8);
        cardView.setUseCompatPadding(true);

        // create a new LinearLayout to hold the card content
        LinearLayout cardContent = new LinearLayout(this);
        cardContent.setOrientation(LinearLayout.VERTICAL);
        cardContent.setLayoutParams(new CardView.LayoutParams(CardView.LayoutParams.MATCH_PARENT, CardView.LayoutParams.WRAP_CONTENT));

        // create the title TextView and add it to the card content LinearLayout
        TextView title = new TextView(this);
        title.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        title.setText(sm.getCity());
        title.setPadding(16, 16, 16, 8);
        title.setTextSize(24);
        cardContent.addView(title);

        // create the description TextView and add it to the card content LinearLayout
        TextView description = new TextView(this);
        description.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        description.setText(sm.getDescription());
        description.setPadding(16, 8, 16, 16);
        description.setTextSize(16);
        description.setMaxLines(2);
        description.setEms(10);
        cardContent.addView(description);

        // create the image and add it to the card content LinearLayout
        Bitmap bmp;
        try {
            URL url = new URL("http://10.0.2.2:5000/image?q="+sm.getCity());
            bmp = BitmapFactory.decodeStream(url.openConnection().getInputStream());

            ImageView image = new ImageView(this);
            image.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
            image.setScaleType(ImageView.ScaleType.CENTER_CROP);
            image.setImageBitmap(bmp);

            cardContent.addView(image);
        } catch (Exception e) {
            System.out.println(e);
        }


// add the card content LinearLayout to the CardView
        cardView.addView(cardContent);
        return cardView;
    }
}