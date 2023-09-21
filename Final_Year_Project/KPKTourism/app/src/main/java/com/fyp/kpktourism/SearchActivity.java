package com.fyp.kpktourism;

import android.content.Intent;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;

import android.os.StrictMode;
import android.view.View;

import android.widget.Button;
import android.widget.EditText;
import com.fyp.kpktourism.databinding.ActivitySearchBinding;

import java.util.ArrayList;
import java.util.List;

public class SearchActivity extends AppCompatActivity {
    private ActivitySearchBinding binding;

    private Button searchButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivitySearchBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        searchButton = (Button) findViewById(R.id.button) ;

        searchButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder().permitAll().build();
                StrictMode.setThreadPolicy(policy);

                EditText description = (EditText) findViewById(R.id.textInputEditText);
                System.out.println(description.getText());
                ApiService api = new ApiService();
                List<SearchResultModel> result = api.search(description.getText().toString());

                Intent mainIntent = new Intent(SearchActivity.this, SearchResult.class);
                mainIntent.putExtra("searchResult", new ArrayList<SearchResultModel>(result));
                startActivity(mainIntent);

            }
        });
    }
}