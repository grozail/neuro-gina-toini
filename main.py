from flask import Flask, request, Response
import torch
import numpy as np
from torch.autograd import Variable
import neuronet.magnusnet as mnet
import jsonpickle

app = Flask('neurohack')
model = mnet.model
model.load_state_dict(torch.load('trained.pt'))


@app.route('/', methods=['POST'])
def eval():
    r = request
    nparr = np.fromstring(r.data, np.int32)
    x = torch.from_numpy(nparr).float()
    x = Variable(x).view(1, 3, 84, 84)
    ans = model(x).data.cpu().numpy()[0]
    response = {'answer': ans, 'type': type(ans)}
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


app.run(host='0.0.0.0', port=81)
