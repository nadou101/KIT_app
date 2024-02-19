import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Adding a radio button in the sidebar for page navigation
st.sidebar.header('Explore Your dataset')
page = st.sidebar.radio('Navigate', ["Dataframe", "Electrons Position", "Data Visualization", "Make Prediction"])

# File uploader widget
uploaded_file = st.sidebar.file_uploader("Upload your input file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    corr = df.corr()

    if page == "Dataframe":
        # Display the dataframe if the page selected is "Home"
        st.header('**Original input data**')
        st.write(df)
    
    elif page == "Data Visualization":
        st.header('Data Visualization')

        if df.empty:
            st.write("No data loaded. Please upload your dataset.")
        else:
            # Letting the user select a column to visualize
            column_to_plot = st.selectbox('Which column do you want to visualize?', df.columns)

            # Letting the user select the type of plot, including 'Heatmap'
            plot_type = st.selectbox('Select the type of plot', ['Histogram', 'Boxplot'])  # Removed 'Line Plot'
            
            # Button to generate plot
            if st.button('Generate Plot'):
                fig, ax = plt.subplots(figsize=(10, 8))  # Set the size of the figure
                
                if plot_type == 'Histogram':
                    # Generating a histogram
                    sns.histplot(df[column_to_plot], kde=True, color='skyblue', ax=ax, fill=True)  # Fill the bins
                    st.pyplot(fig)
                
                elif plot_type == 'Boxplot':
                    # Generating a boxplot
                    sns.boxplot(data=df, x=column_to_plot, ax=ax, palette='Pastel1')  # Set the palette for the boxplot
                    st.pyplot(fig)
                
    elif page == "Electrons Position":
        # Proceed with data visualization only if the dataframe is not empty and the page is "Data Visualization"
        
        # Selecting an event to visualize
        event_id = st.selectbox('Select an Event ID:', df['Event'].unique())

        # Filtering data for the selected event
        event_data = df[df['Event'] == event_id]

        # Plotting momentum components
        st.subheader('Momentum Components')
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=event_data['px1 '], y=event_data['py1'], z=event_data['pz1'],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Electron 1'))

        # Adding the second electron's momentum components to the plot
        fig.add_trace(go.Scatter3d(x=event_data['px2'], y=event_data['py2'], z=event_data['pz2'],mode='markers',marker=dict(size=5, color='blue'),name='Electron 2'))

        # Setting the layout for the 3D plot
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),scene=dict(xaxis_title='px (GeV)', yaxis_title='py (GeV)',zaxis_title='pz (GeV)'))

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Make Prediction":
        st.header('Predict Invariant Mass')
        
        # Model uploader
        uploaded_model = st.file_uploader("Upload your model (HDF5 format)", type=["h5"])


        # Check if both the model and scaler have been uploaded
        if uploaded_model is not None:
            # Load the model
            with st.spinner('Loading model...'):
                model_path = uploaded_model.name
                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                model = tf.keras.models.load_model(model_path)

            
            # User inputs for features
            E1 = st.number_input("Enter E1 value:", format="%f")
            E2 = st.number_input("Enter E2 value:",  format="%f")
            pt1 = st.number_input("Enter pt1 value:", min_value=0.0, format="%f")
            pt2 = st.number_input("Enter pt2 value:", min_value=0.0, format="%f")
            
            # Button to make prediction
            if st.button('Predict Invariant Mass'):
                # Preparing input data with normalization
                input_features = np.array([[E1, pt1, E2, pt2]])


                # Making prediction
                prediction = model.predict(input_features)
                
                # Displaying the prediction
                st.write(f'Predicted Invariant Mass: {prediction[0][0]}')
        else:
            st.warning('Please upload both the model and scaler to enable predictions.')

else:
    st.info('please upload your dataset')