from flask import Flask, request, Response, jsonify
import torch
import numpy as np
from torch.autograd import Variable
import neuronet.magnusnet as mnet
import jsonpickle
import json

app = Flask('neurohack')
model = mnet.model
model.load_state_dict(torch.load('trained.pt'))
model.cpu()

@app.route('/', methods=['POST'])
def eval():
    r = request
    nparr = np.fromstring(r.data, np.int32)
    print(nparr)
    x = torch.from_numpy(nparr).float()
    x = Variable(x).view(1, 3, 84, 84)
    x.cpu()
    ans = model(x).data.cpu().numpy()[0]
    print(ans)
    response = {'answer': ans[0], 'type': type(ans[0])}
    response_pickled = jsonpickle.encode(response)
    return Response(response=jsonify(response), status=200, mimetype="application/json")


app.run(host='0.0.0.0', port=81)
