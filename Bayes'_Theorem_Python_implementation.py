import numpy as np
import matplotlib.pyplot as plt
import gc

plt.close('all')
gc.collect()
truePositiveRate = 0.95
trueNegativeRate = 0.90
trueNegAndFalsePos = trueNegativeRate + (1 - truePositiveRate)
truePosAndFalseNeg = truePositiveRate + (1 - trueNegativeRate)
steps = 100
def BayesRule(prevalence, sensitivity, specificity):
    return (prevalence * sensitivity)/ ((prevalence * sensitivity)+((1-prevalence)*(1-specificity)))
testEffectiveness = []
xvalues = []
prevelences = 10**-5
while prevelences < 0.5:
    xvalues.append(prevelences)
    testEffectiveness.append(BayesRule(prevelences, truePositiveRate, trueNegativeRate))
    prevelences += (0.5-10**-5)/steps
plt.plot(xvalues, testEffectiveness)
plt.axhline(y=0.5, color='r', linestyle='--', label='Prevalence where test is 50/50')
plt.xscale('log')
plt.xlabel('Prevalence')
plt.ylabel('Probability of being sick given a positive test')
plt.title('Effectiveness of a test with 95% sensitivity and 90% specificity')
plt.grid()
plt.show()
