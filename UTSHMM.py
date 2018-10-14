# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 19:42:46 2018

@author: User
"""


import random


from Bio import Alphabet
from Bio.Seq import MutableSeq

from Bio.HMM import MarkovModel
from Bio.HMM import Trainer


class stateHMM (Alphabet.Alphabet):
    letters = ['1', '2', '3', '4', '5']
    
class emissionHMM (Alphabet.Alphabet):
    letters = ['A','C' , 'T', 'G']
    
def initialHMM(num, currentstate):

    if currentstate == '1':
        if num <= 0.2:
            return 'A'
        elif num <= 0.3:
            return 'C'
        elif num <= 0.4:
            return 'T'
        else:
            return 'G'
       
    elif currentstate == '2':
        if num <= 0.1:
            return 'A'
        elif num <= 0.15 :
            return 'C'
        elif num <= 0.2:
            return 'T'
        else:
            return 'G'
    
    elif currentstate == '3':
        if num <= 0.3:
            return 'A'
        elif num <= 0.4:
            return 'C'
        elif num <= 0.5:
            return 'T'
        else:
            return 'G'
        
    elif currentstate == '4':
        if num <= 0.2:
            return 'A'
        elif num <= 0.3:
            return 'C'
        elif num <= 0.35:
            return 'T'
        else:
            return 'G'
    
    elif currentstate == '5':
        if num <= 0.1:
            return 'A'
        elif num <= 0.3:
            return 'C'
        elif num <= 0.5:
            return 'T'
        else:
            return 'G'

hmmbuild = MarkovModel.MarkovModelBuilder(stateHMM(), emissionHMM())
hmmbuild.allow_all_transitions()
hmmbuild.set_random_probabilities()

def trainHMM(num):
    currentstate = '1'
    emisseq = MutableSeq('', emissionHMM())
    stateseq = MutableSeq('', stateHMM())

    for x in range(num):
        stateseq.append(currentstate)
        chance_num = random.random()

        newtrain = initialHMM(chance_num, currentstate)
        emisseq.append(newtrain)

        chance_num = random.random()
        if currentstate == '1':
            if 0.00 < chance_num <= 0.02:
                currentstate = '2'
            elif 0.02 < chance_num <= 0.03:
                currentstate = '3'
            elif 0.03 < chance_num <= 0.2:
                currentstate = '4'
            elif 0.2 < chance_num <= 0.35:
                currentstate = '5'
            else:
                currentstate = '1'
        elif currentstate == '2':
            if 0.35 < chance_num <= 0.4:
                currentstate = '3'
            elif 0.4 < chance_num <= 0.5:
                currentstate = '4'
            elif 0.5 < chance_num <= 0.55:
                currentstate = '5'
            else :
                currentstate = '2'
                
        elif currentstate == '3':
            if 0.4 < chance_num <= 0.5:
                currentstate = '4'
            elif 0.5 < chance_num <= 0.6:
                currentstate = '5'
            else:
                currentstate = '3'
    
        elif currentstate == '4':
            if 0.2 < chance_num <= 0.46:
                currentstate = '5'
            else:
                currentstate = '4'
        
        elif currentstate == '5':
            if 0.1 < chance_num <= 0.35:
                currentstate = '1'
            else:
                currentstate = '5'
            
            
    return emisseq.toseq(), stateseq.toseq()

emission , states = trainHMM(100)
TrainingSeq = Trainer.TrainingSequence(emission, states)

def stop_training(conchange, num_iterations):
    if conchange < 0.001:
        return 1
    elif num_iterations >= 10:
        return 1
    else:
        return 0

 
baum_welch = hmmbuild.get_markov_model()
trainers = Trainer.BaumWelchTrainer(baum_welch)
trainerhmm = trainers.train([TrainingSeq], stop_training )
predic , prob = trainerhmm.viterbi( emission, stateHMM())

print('EMISSION : ' , emission)
print('STATES : ' , states)
print('PREDICTION : ', predic)

a = ['A','C', 'C' , 'G' , 'T']
pre , pr = trainerhmm.viterbi(a, stateHMM())
print(a)
print('prediction : ', pre)
print('prob : ', pr)

#print(trainerhmm.transition_prob)
#print(trainerhmm.emission_prob)





#%%
