import streamlit as st
import pickle
import pandas as pd
rest_types = ['Quick Bites', 'Casual Dining', 'Cafe', 'Delivery', 
              'Dessert Parlor', 'Bakery', 'Beverage Shop', 'Bar',
              'Food Court', 'Fine Dining', 'Lounge', 'Sweet Shop', 'Pub']

cuisines = ['Fast Food', 'Beverages', 'Biryani', 'Mughlai', 'Chinese',
            'BBQ', 'Continental', 'North Indian', 'Italian', 'Arabian',
            'Sandwich', 'Rolls', 'Burger', 'Kebab', 'Asian', 'European',
            'Pizza', 'Salad', 'Street Food', 'Thai', 'South Indian',
            'Kerala', 'Seafood', 'American', 'Steak', 'Momos',
            'Mediterranean', 'Finger Food', 'Ice Cream', 'Mithai',
            'Andhra', 'Desserts', 'Healthy Food', 'Hyderabadi',
            'Bengali', 'Mangalorean', 'Juices', 'Mexican', 'Indian']
st.title("ML-based Restaurant Rating Prediction System")
area_rating = pickle.load(open('area_rating.pkl', 'rb'))
all_areas = list(area_rating.keys())
area = st.selectbox("Select Area",all_areas)
cusine = st.multiselect("Select Cusine",cuisines)
res_type  = st.multiselect("Select Restaurant Type",rest_types)
online_order = st.selectbox("Online Order",["Yes","No"])
table_booking = st.selectbox("Table Booking",["Yes","No"])
avg_cost = st.slider("Select Average Cost",100,2000,500,50)
model = pickle.load(open('model.pkl','rb'))
feature_col = pickle.load(open("feature_columns.pkl","rb"))
if st.button("Predict"):
    if not cusine and not res_type:
        st.warning("Please select atleast one cusine or restaurant type")
    else:   
        st.write("Predicting...")
        input_dict = {col :0 for col in feature_col}
        input_dict['table_booking'] = 1 if table_booking == 'Yes' else 0 
        input_dict['online_order']=1 if online_order == 'Yes' else 0 
        input_dict['avg_cost']= avg_cost
        for c in cusine:
            input_dict[f"is_{c.lower().replace(' ','_')}"] = 1
        for r in res_type:
            input_dict[f"is_{r.lower().replace(' ','_')}"] = 1
        input_dict['encoded_area'] = area_rating[area]
        df = pd.DataFrame([input_dict])
        df = df[feature_col]

        # DEBUG - remove after fixing
        trained_features = model.feature_names_in_
        st.write("Columns in df:", list(df.columns))
        st.write("Trained features:", list(trained_features))
        mismatch = set(trained_features) - set(df.columns)
        st.write("Missing from df:", mismatch)
        extra = set(df.columns) - set(trained_features)
        st.write("Extra in df:", extra)

        prediction = model.predict(df)
        st.success(f"Predicted Rating: {prediction[0]:.2f} ⭐")