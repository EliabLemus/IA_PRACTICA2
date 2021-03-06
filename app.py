# app.py
import csv
import pickle, os, glob
import loadFile
from io import StringIO
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_bootstrap import Bootstrap
results = {}
usac_model = []
landivar_model = []
marroquin_model = []
mariano_model = []
app = Flask(__name__)
Bootstrap(app)
UPLOAD_FOLDER = 'static/UploadedFiles'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FLASK_DEBUG'] = True
app.debug=1
app.after_request
image_path = os.path.join(app.config['UPLOAD_FOLDER'])
# #open saved models
with open('TrainedModels/usac_model.dat', 'rb') as f:
    models = pickle.load(f)
    usac_model = models[0]
with open('TrainedModels/landivar_model.dat', 'rb') as f:
    models = pickle.load(f)
    landivar_model = models[4]
with open('TrainedModels/marroquin_model.dat', 'rb') as f:
    models = pickle.load(f)
    marroquin_model = models[3]
with open('TrainedModels/mariano_model.dat', 'rb') as f:
    models = pickle.load(f)
    mariano_model = models[4]
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
                        coincidences[key] = "Es USAC"
                    elif result1[0]:
                        coincidences[key] = "Es Landivar"
                    elif result2[0]:
                        coincidences[key] = "Es Mariano"
                    elif result3[0]:
                        coincidences[key] = "Es Marroquin"
                    else:
                        coincidences[key] = "No reconocida"
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

@app.route("/model-list", methods=['GET'])
def model():
    file_names = ['Landivar.png', 'Mariano.png', 'Marroquin.png', 'USAC.png']
    models_used = {}
    print(landivar_model.train_accuracy)
    models_used['Landivar.png'] = landivar_model
    models_used['Mariano.png'] = mariano_model
    models_used['Marroquin.png'] = marroquin_model
    models_used['USAC.png'] = usac_model
    return render_template('models.html',filenames=file_names, models_used = models_used )
@app.route('/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='UploadedFiles/' + filename), code=301)    
@app.route('/model/<filename>')
def display_model(filename):
	print('display_model filename: ' + filename)
	return redirect(url_for('static', filename='ModelGraphs/' + filename), code=301)    