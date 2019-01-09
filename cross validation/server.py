from flask import Flask
from flask import json
from flask import request
from flask import Response
import time

from models.model_b import train as trainB
from models.model_b import predictScore as predictB
from models.model_c import predictScore as predictC

start = time.time()

print("Training for model B...")
trainB()
end = time.time()
print("Training finished in {0:.2f} seconds.".format(end - start))


app = Flask(__name__)


@app.route('/predict', methods = ["POST"])
def api_articles():
    dataJson = json.dumps(request.json)
    print("Request: " + dataJson)

    parse = json.loads(dataJson)

    modelId = parse["modelId"]
    question = parse["question"]
    questionResponse = parse["questionResponse"]

    if modelId == "A":
        resScore = "ToDo za model A"
    elif modelId == "B":
        resScore = predictB(question, questionResponse)
    elif modelId == "C":
        resScore = predictC(question, questionResponse)

    data = {
        "score": resScore,
        "probability": None
    }
    js = json.dumps(data)

    return Response(js, status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run(port=8080)