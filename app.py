    
from src.StudentsGradesPrediction.pipelines.prediction_pipeline import CustomData,PredictPipeline

from flask import Flask,request,render_template,jsonify


app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data=CustomData(
            
            StudyTimeWeekly=float(request.form.get('StudyTimeWeekly')),
            Absences=int(request.form.get('Absences')),
            Volunteering=int(request.form.get('Volunteering')),
            Music=int(request.form.get('Music')),
            Sports=int(request.form.get('Sports')),
            Extracurricular=int(request.form.get('Extracurricular')),
            ParentalSupport=int(request.form.get('ParentalSupport')),
            Tutoring=int(request.form.get('Tutoring')),
            ParentalEducation=int(request.form.get('ParentalEducation')),
            Ethnicity=int(request.form.get('Ethnicity')),
            Gender=int(request.form.get('Gender')),
        )
        # this is my final data
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictPipeline()
        
        pred=int(predict_pipeline.predict(final_data))

        # Mapping dictionary for GPA to grades
        gpa_to_grade = {
            0: 'A (GPA >= 3.5)',
            1: 'B (3.0 <= GPA < 3.5)',
            2: 'C (2.5 <= GPA < 3.0)',
            3: 'D (2.0 <= GPA < 2.5)',
            4: 'F (GPA < 2.0)'
        }

        return render_template("result.html",final_result=gpa_to_grade.get(pred))

#execution begin
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)