import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # We use os.path.join to ensure Linux (Render) and Windows both understand the path
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            # Debugging prints - These will show up in your Render Logs
            print("Checking paths...")
            if not os.path.exists(model_path):
                print(f"ERROR: {model_path} not found!")
            if not os.path.exists(preprocessor_path):
                print(f"ERROR: {preprocessor_path} not found!")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("Files loaded successfully. Scaling data...")
            data_scaled = preprocessor.transform(features)
            
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            # Printing the raw error helps us see the issue on Render instantly
            print(f"Exception occurred in Prediction: {str(e)}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)