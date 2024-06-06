import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 Client_Income: float,
                 Credit_Amount: float,
                 Loan_Annuity: float,
                 Age_Days: float,
                 Employed_Days: float,
                 Registration_Days: float,
                 ID_Days: float,
                 Population_Region_Relative: float,
                 Car_Owned: int,
                 Bike_Owned: int,
                 Active_Loan: int,
                 House_Own: int,
                 Client_Gender: str,
                 Client_Housing_Type: str,
                 Client_Occupation: str,
                 Type_Organization: str):
        
        self.Client_Income = Client_Income
        self.Credit_Amount = Credit_Amount
        self.Loan_Annuity = Loan_Annuity
        self.Age_Days = Age_Days
        self.Employed_Days = Employed_Days
        self.Registration_Days = Registration_Days
        self.ID_Days = ID_Days
        self.Population_Region_Relative = Population_Region_Relative
        self.Car_Owned = Car_Owned
        self.Bike_Owned = Bike_Owned
        self.Active_Loan = Active_Loan
        self.House_Own
