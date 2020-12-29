# app.py
import csv
import pickle, os, glob
import loadFile
from io import StringIO
from flask import Flask, render_template, request, jsonify, redirect, url_for
results = {}
app = Flask(__name__)
UPLOAD_FOLDER = 'static/UploadedFiles'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FLASK_DEBUG'] = True
app.debug=1
app.after_request
image_path = os.path.join(app.config['UPLOAD_FOLDER'])
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'


@app.route("/", methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
           return render_template('form.html', results=results, image_path = image_path)
       
    if request.method == 'POST':
            for file in glob.glob(app.config['UPLOAD_FOLDER']+'/*'):
                os.remove(file)
            transformed = {}
            predict_result = {}
            results['table_results'] = {}    
            results['model_results'] = {}
            coincidences = {}
            # #open saved models
            with open('TrainedModels/usac_model.dat', 'rb') as f:
                models = pickle.load(f)
                usac_model = models[4]
            with open('TrainedModels/landivar_model.dat', 'rb') as f:
                models = pickle.load(f)
                landivar_model = models[4]
            with open('TrainedModels/marroquin_model.dat', 'rb') as f:
                models = pickle.load(f)
                marroquin_model = models[4]
            with open('TrainedModels/mariano_model.dat', 'rb') as f:
                models = pickle.load(f)
                mariano_model = models[4]
                    
            fileName = request.files['files'].filename
            f = request.files.getlist('files')
            for file in f:
                print(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            
            transformed = loadFile.transform_image(app.config['UPLOAD_FOLDER'])
            if len(f) <= 5: 
                print('mostrar las imagenes')
                for key in transformed:
                    result = usac_model.predict(transformed[key])
                    result1 = landivar_model.predict(transformed[key])
                    result2 = mariano_model.predict(transformed[key])
                    result3 = marroquin_model.predict(transformed[key])
                    if result[0]:
                        coincidences[key] = result[0]
                    elif result1[0]:
                        coincidences[key] = result[0]
                    elif result2[0]:
                        coincidences[key] = result[0]
                    elif result3[0]:
                        coincidences[key] = result[0]
                    else:
                        coincidences[key] = 0
                print('transformed:',len(transformed.keys()))
                print('coincidences:',len(coincidences.keys()))
                results['table_results'] = coincidences
            else:
                print('mostrar tabla')
                total = len(transformed.keys())

                print('Total:', total)
                usac = 0 
                landivar = 0 
                mariano = 0 
                marro = 0 
                unknown = 0
                for key in transformed:
                    result = usac_model.predict(transformed[key])
                    result1 = landivar_model.predict(transformed[key])
                    result2 = mariano_model.predict(transformed[key])
                    result3 = marroquin_model.predict(transformed[key])
                    if result[0]:
                        usac += 1
                    elif result1[0]:
                        landivar += 1
                    elif result2[0]:
                        mariano += 1
                    elif result3[0]:
                        marro += 1
                    else:
                        unknown += 1
                    predict_result['Usac'] = (usac/total) * 100
                    predict_result['Landivar'] = (landivar/total) * 100
                    predict_result['Marroquin'] = (marro/total) * 100
                    predict_result['Mariano'] = (mariano/total) * 100
                    predict_result['Desconocido'] = (unknown/total) * 100
                    results['model_results'] = predict_result
    return render_template('form.html',results=results, image_path = image_path)
@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('/UploadedFiles', filename='/' + filename), code=301)    