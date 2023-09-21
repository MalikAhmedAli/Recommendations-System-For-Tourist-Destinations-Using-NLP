package com.fyp.kpktourism;

import android.os.Parcel;
import android.os.Parcelable;

public class SearchResultModel implements Parcelable {

    private String city;
    private String description;

    public SearchResultModel() {}

    public SearchResultModel(String city, String description) {
        this.city = city;
        this.description = description;
    }

    // Parcelable implementation
    public static final Creator<SearchResultModel> CREATOR = new Creator<SearchResultModel>() {
        @Override
        public SearchResultModel createFromParcel(Parcel in) {
            return new SearchResultModel(in);
        }

        @Override
        public SearchResultModel[] newArray(int size) {
            return new SearchResultModel[size];
        }
    };

    protected SearchResultModel(Parcel in) {
        city = in.readString();
        description = in.readString();
    }

    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeString(city);
        dest.writeString(description);
    }

    public String getCity() {
        return city;
    }

    public void setCity(String city) {
        this.city = city;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }
}
