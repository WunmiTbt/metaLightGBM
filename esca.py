import numpy as np
import numpy
import time
import random
import math 
import csv
from numpy import cos, pi, sqrt, fabs, sin, sum, floor, abs, arange
from scipy.stats import levy

class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual = []
        self.convergence = []
        self.optimizer = ""
        self.objfname = ""
        self.startTime = 0
        self.endTime = 0
        self.executionTime = 0
        self.lb = 0
        self.ub = 0
        self.dim = 0
        self.popnum = 0
        self.maxiers = 0

def Check(indi, LB, UB):
    for i in range(len(LB)):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
        else:
            pass
    return indi
        
def ESCA(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # destination_pos
    Dest_pos = np.zeros(dim)
    Dest_score = float("inf")

    Worst_pos = np.zeros(dim)
    Worst_score = float("inf")

    fitness = np.zeros(SearchAgents_no)

    off = np.zeros(dim)

    if np.isscalar(ub):
        ub = np.ones(dim) * ub
        lb = np.ones(dim) * lb

    # Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            np.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    for i in range(0, SearchAgents_no):
        Positions[i]= Check(Positions[i], lb, ub)
        fitness[i] = objf(Positions[i, :])
        
    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()

    # Loop counter
    #print('SCA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    ChaosValue = np.random.rand()  # Initial value for Logistic map
    a = 4  # Parameter for Logistic map

    worstIdx = np.argmax(fitness)
    Worst_pos = Positions[worstIdx].copy()
    Worst_score = fitness[worstIdx]

    destIdx = np.argmin(fitness)
    Dest_pos = Positions[destIdx].copy()
    Dest_score = fitness[destIdx]

    # Main loop
    for l in range(0, Max_iter):

        curBest = Dest_pos
        curWorst = Worst_pos

        maxDist = np.linalg.norm(curBest - curWorst + np.finfo(float).eps, ord=2)
        safeRadius = np.random.uniform(0.8, 1.2) * maxDist

        Xmean = np.mean(Positions, axis=0)

        ChaosValue = a * ChaosValue * (1 - ChaosValue)
        w1 = (1 - l / Max_iter) ** (1 - np.tan(np.pi * (ChaosValue * np.random.rand() - 0.5)) * l / Max_iter)

        w = 2
        factor = np.random.uniform(-1, 1) * np.cos(np.pi * (l + 1) / (Max_iter / 10)) * (1 - np.round((l + 1) * w / Max_iter) / w)
        
        # Eq. (3.4)
        a = 2
        Max_iteration = Max_iter
        r1 = a - l * ((a) / Max_iteration)  # r1 decreases linearly from a to 0

        # Update the Position of search agents
        for i in range(0, SearchAgents_no):

            if np.random.rand() < 0.5:
                #EMBGO https://doi.org/10.1016/j.aej.2024.11.035
                distance = np.linalg.norm(curBest - Positions[i] + np.finfo(float).eps, ord=2)
                if distance < safeRadius:
                    off = Positions[i] + np.sin(np.random.rand() * 2 * np.pi) * (curBest - Positions[i]) + np.sin(np.random.rand() * 2 * np.pi) * (Xmean - Positions[i])
                else:    
                    #https://doi.org/10.1016/j.aej.2024.09.109
                    off = Positions[i] + np.random.uniform(-1, 1, dim)
                              
            else:

                for j in range(0, dim):
                    # Update r2, r3, and r4 for Eq. (3.3)
                    r2 = (2 * np.pi) * random.random()
                    r3 = 2 * random.random()
                    r4 = random.random()

                    #used w1 from https://doi.org/10.1007/s10586-024-04716-9
                    # Eq. (3.3)
                    if r4 < (0.5):
                        # Eq. (3.1)
                        off[j] = Positions[i, j] + w1 * (
                            r1 * np.sin(r2) * abs(r3 * Dest_pos[j] - Positions[i, j])
                        )
                        off[j] = Positions[i, j] + w1 * (
                            r1 * np.cos(r2) * abs(r3 * Dest_pos[j] - Positions[i, j])
                        )

            off = Check(off, lb, ub)
            off_fitness = objf(off)
            if off_fitness < fitness[i]:
                Positions[i] = off.copy()
                fitness[i] = off_fitness

        worstIdx = np.argmax(fitness)
        Worst_pos = Positions[worstIdx].copy()
        Worst_score = fitness[worstIdx]

        destIdx = np.argmin(fitness)
        Dest_pos = Positions[destIdx].copy()
        Dest_score = fitness[destIdx]

        if type(Dest_score) == numpy.ndarray:
            Dest_score = Dest_score[0]
        Convergence_curve[l] = Dest_score

    print(f"{Dest_score:e}")
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "ESCA"
    s.objfname = objf.__name__
    s.best = Dest_score
    s.bestIndividual = Dest_pos
    return s
