# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 19:42:46 2018

@author: User
"""

import Bio.Seq
import Bio.Alphabet
import os 
import sys

from Bio.HMM import MarkovModel
from Bio.HMM import Trainer

#define HMM
#hmmstate = ['1','2','3','4','5']
#hmmemmision = ['A','C','G','T']


hmmstry= {
                "A": {
                   "1" : {"1": 0.4, "2": 0.2, "3": 0.1 , "4": 0.2 , "5": 0.1 },
                   "2" : {"1": 0.2, "2": 0.1, "3": 0.3 , "4": 0.3 , "5": 0.1 },
                   "3" : {"1": 0.1, "2": 0.2, "3": 0.2 , "4": 0.1 , "5": 0.4 },
                   "4" : {"1": 0.2, "2": 0.2, "3": 0.2 , "4": 0.1 , "5": 0.3 },
                   "5" : {"1": 0.3, "2": 0.2, "3": 0.1 , "4": 0.1 , "5": 0.3 }
               },
                "B": {
                       "1" : {"A": 0.3, "C": 0.3 , "T": 0.2 , "G": 0.2 },
                       "2" : {"A": 0.5, "C": 0.2 , "T": 0.1 , "G": 0.2 },
                       "3" : {"A": 0.2, "C": 0.3 , "T": 0.4 , "G": 0.1 },
                       "4" : {"A": 0.2, "C": 0.2 , "T": 0.3 , "G": 0.3 },
                       "5" : {"A": 0.1, "C": 0.4 , "T": 0.3 , "G": 0.2 }
                },
                "pi": {"1": 0.4, "2": 0.2, "3": 0.2 , "4": 0.1 , "5" : 0.1 }        
            }
        

hmmtry = [[0.4 , 0.2 , 0.1 , 0.2 , 0.1 ],
          [0.2 , 0.1 , 0.3 , 0.3 , 0.1 ],
          [0.1 , 0.2 , 0.2 , 0.1 , 0.4 ],
          [0.2 , 0.2 , 0.2 , 0.1 , 0.3 ],
          [0.3 , 0.2 , 0.1 , 0.1 , 0.3 ]
          ]


hmmrandom= { 
                "A": {
                   "Coin 1" : {"Coin 1": 0.1, "Coin 2": 0.3, "Coin 3": 0.6},
                   "Coin 2" : {"Coin 1": 0.8, "Coin 2": 0.13, "Coin 3": 0.07},
                   "Coin 3" : {"Coin 1": 0.33, "Coin 2": 0.33, "Coin 3": 0.34}
               },
                "B": {
                       "Coin 1" : {"Heads": 0.15, "Tails": 0.85},
                       "Coin 2" : {"Heads": 0.5, "Tails": 0.5},
                       "Coin 3" : {"Heads": 0.5, "Tails": 0.5}
                },
                "pi": [["Coin 1", 0.33], ["Coin 2",0.33], ["Coin 3", 0.34]]
        
        }




class initialHMM(object):
    
    def __init__ (self):
        self.model = hmmstry
        self.A = self.model.get("A")
        self.B = self.model.get("B")
        self.states = ['1','2','3','4','5']
        self.symbols =['A','C','T','G']
        self.N = len(self.states)
        self.M = len(self.symbols)
        self.pi = [["1", 0.4],[ "2", 0.2],["3", 0.2] , ["4", 0.1] , ["5" , 0.1]]
        
    def backward(self, obs):
        self.bwk = [{} for t in range(len(obs))]
        T = len(obs)
        # Initialize base cases (t == T)
        for y in self.states:
            self.bwk[T-1][int(y)] = 1 #self.A[y]["Final"] #self.pi[y] * self.B[y][obs[0]]
        for t in reversed(range(T-1)):
            for y in self.states:
                self.bwk[t][int(y)] = sum((self.bwk[t+1][y1] * self.A[int(y)][y1] * self.B[y1][obs[t+1]]) for y1 in self.states)
        prob = sum((self.pi[int(y)]* self.B[int(y)][obs[0]] * self.bwk[0][int(y)]) for y in self.states)
        return prob

    def forward(self, obs):
        self.fwd = [{}]     
        # Initialize base cases (t == 0)
        for y in self.states:
            self.fwd[0][int(y)] = self.pi[int(y)][0] * self.B[int(y)][obs[0]]
            
        # Run Forward algorithm for t > 0
        for t in range(1, len(obs)):
            self.fwd.append({})     
            for y in self.states:
                self.fwd[t][int(y)] = sum((self.fwd[t-1][y0] * self.A[y0][int(y)] * self.B[int(y)][obs[t]]) for y0 in self.states)
        prob = sum((self.fwd[len(obs) - 1][s]) for s in self.states)
        return prob


    def viterbi(self, obs):
        vit = [{}]
        path = {}     
        # Initialize base cases (t == 0)
        for y in self.states:
            vit[0][y] = self.pi[y] * self.B[y][obs[0]]
            path[y] = [y]
     
        # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            vit.append({})
            newpath = {}     
            for y in self.states:
                (prob, state) = max((vit[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]], y0) for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]     
            # Don't need to remember the old paths
            path = newpath
        n = 0           # if only one element is observed max is sought in the initialization values
        if len(obs)!=1:
            n = t
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return (prob, path[state])

    def forward_backward(self, obs): # returns model given the initial model and observations        
        gamma = [{} for t in range(len(obs))] # this is needed to keep track of finding a state i at a time t for all i and all t
        zi = [{} for t in range(len(obs) - 1)]  # this is needed to keep track of finding a state i at a time t and j at a time (t+1) for all i and all j and all t
        # get alpha and beta tables computes
        p_obs = self.forward(obs)
        self.backward(obs)
        # compute gamma values
        for t in range(len(obs)):
            for y in self.states:
                gamma[t][y] = (self.fwd[t][y] * self.bwk[t][y]) / p_obs
                if t == 0:
                    self.pi[y] = gamma[t][y]
                #compute zi values up to T - 1
                if t == len(obs) - 1:
                    continue
                zi[t][y] = {}
                for y1 in self.states:
                    zi[t][y][y1] = self.fwd[t][y] * self.A[y][y1] * self.B[y1][obs[t + 1]] * self.bwk[t + 1][y1] / p_obs
        # now that we have gamma and zi let us re-estimate
        for y in self.states:
            for y1 in self.states:
                # we will now compute new a_ij
                val = sum([zi[t][y][y1] for t in range(len(obs) - 1)]) #
                val /= sum([gamma[t][y] for t in range(len(obs) - 1)])
                self.A[y][y1] = val
        # re estimate gamma
        for y in self.states:
            for k in self.symbols: # for all symbols vk
                val = 0.0
                for t in range(len(obs)):
                    if obs[t] == k :
                        val += gamma[t][y]                 
                val /= sum([gamma[t][y] for t in range(len(obs))])
                self.B[y][k] = val
        return

models_dir = os.path.join('.', 'models') #

seq0 = ('A', 'C', 'G', 'A')
seq1 = ('A', 'S', 'T')
seq2 = ('A', 'C', 'C' , 'G', 'T')

observation_list = [seq0, seq1, seq2]

if __name__ == '__main__':
    #test the forward algorithm and backward algorithm for same observations and verify they produce same output
    #we are computing P(O|model) using these 2 algorithms.
    model_file = "coins1.json" # this is the model file name - you can create one yourself and set it in this variable
    hmm = initialHMM()
    print ("Using the model from file: ", model_file, " - You can modify the parameters A, B and pi in this file to build different HMM models")
    
    total1 = total2 = 0 # to keep track of total probability of distribution which should sum to 1
    for obs in observation_list:
        p1 = hmm.forward(obs)
        p2 = hmm.backward(obs)
        total1 += p1
        total2 += p2
        print ("Observations = ", obs, " Fwd Prob = ", p1, " Bwd Prob = ", p2, " total_1 = ", total1, " total_2 = ", total2)

    # test the Viterbi algorithm
    observations = seq1 + seq0 + seq2  # you can set this variable to any arbitrary length of observations
    prob, hidden_states = hmm.viterbi(observations)
    print ("Max Probability = ", prob, " Hidden State Sequence = ", hidden_states)

    print ("Learning the model through Forward-Backward Algorithm for the observations", observations)
    model_file = "random1.json"
    #hmm = initialHMM(os.path.join(model_file))
    print ("Using the model from file: ", model_file, " - You can modify the parameters A, B and pi in this file to build different HMM models")
    hmm.forward_backward(observations)

    print ("The new model parameters after 1 iteration are: ")
    print ("A = ", hmm.A)
    print ("B = ", hmm.B)
    print ("pi = ", hmm.pi)
        

    
    
    #%%
    
def AwalTraining (self):
    stateseq = self.Seq('ACGT',hmmstate)
    emmisionseq = self.Seq('',hmmemmision)
    
#%%


    #%%



#%%
