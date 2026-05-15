from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Get the values first to check if they exist
            reading_val = request.form.get('reading_score')
            writing_val = request.form.get('writing_score')

            # Validation: If for some reason values are missing, send back to form
            if reading_val is None or writing_val is None or reading_val == '' or writing_val == '':
                return render_template('home.html', results="Error: Please provide all scores.")

            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(reading_val), # This was the crash point
                writing_score=float(writing_val)  # This was the crash point
            )
            
            pred_df = data.get_data_as_data_frame()
            print("Dataframe columns sent to preprocessor:", pred_df.columns)
            
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            return render_template('home.html', results=round(results[0], 2))
            
        except Exception as e:
            # This will catch any other errors and print them to your terminal
            print(f"Error occurred: {str(e)}")
            return render_template('home.html', results=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)