import os
import sys
import pandas as pd
from src.StudentsGradesPrediction.exception import customexception
from src.StudentsGradesPrediction.logger import logging
from src.StudentsGradesPrediction.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
            
        
        except Exception as e:
            raise customexception(e,sys)
        
class CustomData:
    def __init__(self,
                 Gender:int,
                 Ethnicity:int,
                 ParentEducation:int,
                 Tutoring:int,
                 ParentalSupport:int,
                 Extracurricular:int,
                 Sports:int,
                 Music:int,
                 Volunteering:int,
                 StudyTimeWeekly:float,
                 Absences:int):
        
        self.Gender=Gender
        self.Ethnicity=Ethnicity
        self.ParentEducation=ParentEducation
        self.Tutoring=Tutoring    
        self.ParentalSupport=ParentalSupport
        self.Extracurricular=Extracurricular
        self.Sports=Sports
        self.Music=Music
        self.Volunteering = Volunteering
        self.StudyTimeWeekly = StudyTimeWeekly
        self.Absences = Absences
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'Gender':[self.Gender],
                    'Ethnicity':[self.Ethnicity],
                    'ParentEducation':[self.ParentEducation],
                    'Tutoring':[self.Tutoring],
                    'ParentalSupport':[self.ParentalSupport],
                    'Extracurricular':[self.Extracurricular],
                    'Sports':[self.Sports],
                    'Music':[self.Music],
                    'Volunteering':[self.Volunteering],
                    'StudyTimeWeekly':[self.StudyTimeWeekly],
                    'Absences':[self.Absences]

                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)