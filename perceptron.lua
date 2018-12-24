perceptron = {}

perceptron.inputNum = 0
perceptron.outputNum = 0
perceptron.weights = {}
perceptron.rate = 0.0001
perceptron.trainCycles=0

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

function perceptron.EvalArgMax(inputs)
  output = perceptron.Eval(inputs)
  k=-100000000
  idx=0
  for i=1,table.getn(output) do
    if output[i]>k then
      k=output[i]
      idx=i
    end
  end
  return idx
end

function perceptron.Train(inputs,outputs,suppresslog)
  dt = perceptron.rate
  y=perceptron.Eval(inputs)
  dy=perceptron.Eval(inputs,true)
  --startError = perceptron.error(y,outputs)
  errorDerivative = perceptron.error(y,outputs,true)
  for i=1,table.getn(perceptron.weights)do
    for j=1,table.getn(perceptron.weights[1])do
      perceptron.weights[i][j] = perceptron.weights[i][j] - errorDerivative*dt*dy[j]*inputs[i]
    end
  end
  finalError = perceptron.error(perceptron.Eval(inputs),outputs)
  perceptron.trainCycles = perceptron.trainCycles+1
  if suppresslog==nil then
    print("Starting Error: "..startError..", Ending Error: "..finalError)
  elseif suppresslog=="sparse" and perceptron.trainCycles%100000==0 then
    print("Error: "..finalError)
  end
  
end

-------------------------------------------------------------------
-------------------------------------------------------------------
-------------------------------------------------------------------

perceptron.Topology({3,2})

outputs = perceptron.Eval({1,0,1})
for i=1,table.getn(outputs) do
  print(outputs[i])
end

for i=1,1000000 do
  perceptron.Train({1,0,1},{1,0},"sparse")
end

outputs = perceptron.Eval({1,0,1})
for i=1,table.getn(outputs) do
  print(outputs[i])
end

print(perceptron.EvalArgMax({1,0,1}))
