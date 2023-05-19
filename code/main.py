from flask import Flask ,request
from flask.templating import render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
import os


rf=joblib.load('rf.joblib')
plt.switch_backend('agg')
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/out",methods=["GET","POST"])
def result():
    input_data = []
    result = "highest bar"
    if request.method == 'POST':
        
        if os.path.exists("static\out.png"):
            os.remove("static\out.png")
        input_data.append((request.form.get("sshb")).lower())
        input_data.append((request.form.get("cmc")).lower())
        input_data.append((request.form.get("sab")).lower())
        input_data.append((request.form.get("dap")).lower())
        input_data.append((request.form.get("drop")).lower())
        input_data.append((request.form.get("moh")).lower())
        input_data.append((request.form.get("aep")).lower())
        input_data.append((request.form.get("dnmtp")).lower())
        input_data.append((request.form.get("wsong")).lower())
        input_data.append((request.form.get("er")).lower())
        input_data.append((request.form.get("cbr")).lower())
        input_data.append((request.form.get("wbp")).lower())
        input_data.append((request.form.get("md")).lower())
        input_data.append((request.form.get("pmi")).lower())
        input_data.append((request.form.get("wbl")).lower())
        input_data.append((request.form.get("ksr")).lower())
        input_data.append((request.form.get("soc")).lower())
        input_data.append((request.form.get("sm")).lower())
        data = pd.Series(input_data)
        data = data.replace('no',0)
        data = data.replace('yes',1)
        data = np.array(data)
        inp_data =np.array([data])
        pb2 = rf.predict_proba(inp_data)
        datadict ={0:"health",1:"technical",2:"agriculture",3:"commerce",4:"Arts"}
        idx = np.argmax(pb2[0])
        result = datadict[idx]
        def addlabels(y):
            for i in range(len(y)):
                plt.text(i,y[i],y[i],color="red")
        res1 = list(np.round((pb2[0]*100),2))
        plt.bar(list(datadict.values()),res1,color =['black', 'yellow', 'green', 'blue', 'cyan'],
                width = 0.4)
        addlabels(res1)
        plt.yticks([])
        plt.savefig('static/out.png', bbox_inches='tight')
        while not os.path.exists("static\out.png"):
            print("loop")
        # time.sleep(20)
    return render_template("out.html",result=result)

@app.route("/test")
def test():
    return render_template("test.html")



app.run(debug=True)