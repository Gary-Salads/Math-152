import numpy as np
import matplotlib.pyplot as plt

truePositiveRate = 0.95
trueNegativeRate = 0.90
trueNegAndFalsePos = trueNegativeRate + (1 - truePositiveRate)
truePosAndFalseNeg = truePositiveRate + (1 - trueNegativeRate)
def BayesRule(prevalence, sensitivity, specificity):
    return (prevalence * sensitivity)/ ((prevalence * sensitivity)+((1-prevalence)*(1-specificity)))
testEffectiveness = []
prevelences = 10**-5
while prevelences < 0.5:
    testEffectiveness.append(BayesRule(prevelences, truePositiveRate, trueNegativeRate))
    prevelences *= 10
plt.plot(prevelences, testEffectiveness)
plt.xscale('log')
plt.xlabel('Prevalence')
plt.ylabel('Probability of being sick given a positive test')
plt.title('Effectiveness of a test with 95% sensitivity and 90% specificity')
plt.grid()
plt.show()
