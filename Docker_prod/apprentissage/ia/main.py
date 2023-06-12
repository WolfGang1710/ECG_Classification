import classify
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import random
import os

UPLOAD_FOLDER = './downloads'
ALLOWED_EXTENSIONS = {'tsv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath(UPLOAD_FOLDER)

@app.route ('/config', methods=['POST'])
def doConfiguration():
   model_name = "CNN_1.h5" 
   if 'graph' in request.form:
      model_name = request.form['graph']
   classify.config(model_name);
   return { "status" : "ok", 
            "data": "condiguration is done",
            "model_name" : model_name
   }
   
def allowed_file(filename):
   return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route ('/classify', methods=['POST'])
def doClassification():
    if 'tsv_file' not in request.files:
        return "<html><body><h1>Error: No TSV file in the request</h1></body></html>", 400

    tsv_file = request.files['tsv_file']
    if not tsv_file:
        return "<html><body><h1>Error: Invalid TSV file</h1></body></html>", 400

    if tsv_file.filename == '':
        return "<html><body><h1>Error: Empty TSV filename</h1></body></html>", 400

    if not allowed_file(tsv_file.filename):
        return "<html><body><h1>Error: Invalid file type. Only TSV files are allowed.</h1></body></html>", 400

    filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(tsv_file.filename))
    tsv_file.save(filename)
    result = classify.classify(filename)

    # Prepare data for chart
    labels = []
    values = []
    for data_point in result:
        for key, value in data_point.items():
            if key == 'probability':
                labels.append(str(key))
                values.append(value)

    # Generate HTML chart
    chart_html = """
    <div class='chart'>
        <div class='chart-container'>
            <canvas id='chart'></canvas>
        </div>
    </div>
    """

    # Generate HTML table
    table_html = """
    <table>
        <thead>
            <tr>
                <th>Probability</th>
            </tr>
        </thead>
        <tbody>
    """
    for data_point in result:
        for key, value in data_point.items():
            if key == 'probability':
                table_html += "<tr><td>{}</td></tr>".format(value)
    table_html += """
        </tbody>
    </table>
    """

    # Generate final HTML page with CSS and JavaScript
    html_result = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; }
            .chart { display: flex; height: 50vh; }
            .chart-container { flex: 1; padding: 10px; }
            table { margin-top: 20px; border-collapse: collapse; }
            table th, table td { padding: 8px 12px; border: 1px solid #ccc; }
            table th { background-color: #f2f2f2; }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                var ctx = document.getElementById('chart').getContext('2d');
                var data = {
                    labels: """ + str(labels) + """,
                    datasets: [{
                        label: 'Probability',
                        data: """ + str(values) + """,
                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                };
                var options = {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                };
                new Chart(ctx, {
                    type: 'bar',
                    data: data,
                    options: options
                });
            });
        </script>
    </head>
    <body>
        """ + chart_html + """
        """ + table_html + """
    </body>
    </html>
    """

    return html_result



@app.route ('/', methods=['GET'])
def doHome():
   return '''
    <!doctype html>
    <html>
        <head>
            <title>Classifying</title>
            <meta charset="UTF-8" />
        </head>
        <body>
            <h1>Classifying</h1>
            <h2>Configure a classifier</h2>
            <form method=post action="/config" enctype="application/x-www-form-urlencoded" name="config">
                <label for="graph">Graph name ?</label>
                <select name="graph" id="graph">
                    <option value="CNN_1.h5" selected="">CNN_1</option>
                    <option value="CNN_2.h5" selected="">CNN_2</option>
                    <option value="RNN_1.h5">RNN_1</option>
                    <option value="RNN_2.h5">RNN_2</option>
                </select>
                <input type="submit" value="Config">
            </form>
            <h2>Classify a TSV file</h2>
            <form method=post action="/classify" enctype="multipart/form-data" name="classify_tsv">
                <input type="file" name="tsv_file">
                <input type="submit" value="Classify TSV">
            </form>
        </body>
    </html>
    '''

if __name__ == '__main__' :
   print ("starting")
   app.run(host='0.0.0.0', port=80, debug=True, use_reloader=False)
   print ("done")

