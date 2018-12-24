perceptron = {}

perceptron.inputNum = 0
perceptron.outputNum = 0
perceptron.weights = {}
perceptron.rate = 0.0001

function perceptron.Topology(topologyList)
  perceptron.inputNum = topologyList[1]
  perceptron.outputNum = topologyList[2]
  mt = {}
  for i=1,perceptron.inputNum do
    mt[i] = {}
    for j=1,perceptron.outputNum do
      mt[i][j] = 2*math.random()-1
    end
  end
  perceptron.weights=mt
end

function perceptron.activation(x,der)
  if der==nil then
    return 1/(1+math.exp(-x))
  else
    return (1/(1+math.exp(-x)))*(1-(1/(1+math.exp(-x))))
  end
end

function perceptron.error(y,yo,der)
  if der==nil then
    E=0
    for i=1, table.getn(y) do
      E = E + (y[i]-yo[i])*(y[i]-yo[i])
    end
    return E
  else
    E=0
    for i=1, table.getn(y) do
      E = E + (y[i]-yo[i])
    end
    return E
  end
end

function perceptron.Eval(inputs,der)
  outputVals={n=perceptron.outputNum}
  if table.getn(inputs)==table.getn(perceptron.weights) then
    for i=1,table.getn(perceptron.weights[1]) do
      outputVals[i]=0
      for j=1,table.getn(perceptron.weights) do
        outputVals[i] = outputVals[i] + perceptron.weights[j][i]*inputs[j]
      end
      if der==nil then
        outputVals[i] = perceptron.activation(outputVals[i])
      else
        outputVals[i] = perceptron.activation(outputVals[i],true)
      end
    end
  end
  return outputVals
end

function perceptron.Train(inputs,outputs)
  dt = perceptron.rate
  y=perceptron.Eval(inputs)
  dy=perceptron.Eval(inputs,true)
  startError = perceptron.error(y,outputs)
  errorDerivative = perceptron.error(y,outputs,true)
  for i=1,table.getn(perceptron.weights[1])do
    for j=1,table.getn(perceptron.weights)do
      perceptron.weights[i][j] = perceptron.weights[i][j] - errorDerivative*dt*dy[i]*inputs[j]
    end
  end
  finalError = perceptron.error(perceptron.Eval(inputs),outputs)
  print("Starting Error: "..startError..", Ending Error: "..finalError)
end

-------------------------------------------------------------------
-------------------------------------------------------------------
-------------------------------------------------------------------

perceptron.Topology({3,3})

outputs = perceptron.Eval({1,1,1})
for i=1,table.getn(outputs) do
  print(outputs[i])
end

for i=1,100000 do
  perceptron.Train({1,0,1},{0,1,0})
end
