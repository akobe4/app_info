from flask import render_template, request, jsonify,Flask
import flask
import numpy as np
import traceback #allows you to send error to user
import pickle
import pandas as pd

# App definition
app = Flask(__name__)

# importing models
with open(r'C:\Users\akobe\OneDrive\Desktop\Lighthouse\Final\Predicting_Success_First_Round_Picks_NHL\Notebooks\model21_transformed_data.pkl', 'rb') as f:
       classifier = pickle.load (f)

#importing columns 
with open(r'C:\Users\akobe\OneDrive\Desktop\Lighthouse\Final\Predicting_Success_First_Round_Picks_NHL\Notebooks\model_columns.pkl', 'rb') as f:
       model_columns = pickle.load (f)

#webpage 
@app.route('/')
def welcome():
    return "Welcome to the prospect predictor - predicts whether a prospect will play 200+ games in the NHL. For the prediction page '/predict'"

@app.route('/predict', methods=['POST','GET']) 
def predict():
    
    if flask.request.method == 'GET':   # get on the webpage - gives info about the application 
        return "Prediction page. Try using post with params to get specific prediction. 1 = predicted to play 200+ games, 0 = predicted to play <200 games"

    if flask.request.method == 'POST':  #how will access the app 
        try:
            json_ = request.json # '_' since 'json' is a special word
            print(json_)
            query_ = pd.DataFrame(json_) #covert json object to pandas dataframe, skikit learn model requires 
            query = query_.reindex(columns = model_columns, fill_value= 0) #reindexing, cleaning up 
            prediction = list(classifier.predict(query)) #cerate a list 

            return jsonify({
               "prediction":str(prediction) #final prediction
           })

        except:
            return jsonify({
               "trace": traceback.format_exc()
               })



if __name__ == "__main__":
    app.run()