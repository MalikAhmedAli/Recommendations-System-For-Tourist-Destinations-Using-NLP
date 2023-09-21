package com.fyp.kpktourism;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.List;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class ApiService {

    public List<SearchResultModel> search(String destination) {

        ObjectMapper mapper = new ObjectMapper();

        OkHttpClient client = new OkHttpClient();

        Request request = new Request.Builder()
                .url("http://10.0.2.2:5000/predict?q=" + destination)
                .build(); // defaults to GET

        Response response;
        try {
            response = client.newCall(request).execute();
            String responseBody = response.body().string();
            List<SearchResultModel> resultList = mapper.readValue(responseBody, new TypeReference<List<SearchResultModel>>() {});
            System.out.println(resultList);
//            List<String> flattened = flatten(resultList);
//            System.out.println(flattened);
            return resultList;

        } catch (Exception e) {
            System.out.println(e);
        }
        return new ArrayList<>();
    }

    public List<String> flatten(List<List<String>> nestedList) {
        List<String> flatList = new ArrayList<>();
        for (List<String> list : nestedList) {
            flatList.addAll(list);
        }
        return flatList;
    }

}
