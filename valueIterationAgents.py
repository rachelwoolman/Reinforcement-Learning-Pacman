# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for i in range(self.iterations):
            newValues = util.Counter()
            for state in self.mdp.getStates():
                # for each state compute it's value given what we know about its rewards..
     
                # get best action from this state
                action = self.computeActionFromValues(state)

                if action == None:
                    newValues[state] = 0
                else:
                    qValueOfAction = self.computeQValueFromValues(state, action)
                    newValues[state] = qValueOfAction
            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        total = 0
        # for each of the possibile actions in the mdp
        for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
            total += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
        
        return total


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        # if at a terminal state return none
        if self.mdp.isTerminal(state):
            return None
        
        # get all actions from this state
        possibleActions = self.mdp.getPossibleActions(state)

        # loop over all actions from this state and compute q value for those states
        # return action with highest q value

        qValuesFromActions = []
        for action in possibleActions:
            qValue = (self.computeQValueFromValues(state, action))
            qValuesFromActions.append((qValue, action))

        (qValue, action) = max (qValuesFromActions)
        return action
        
    

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates();
        for i in range(self.iterations):
            # get one state
            # modulo ensures we cycle though the states
            state = states[i % len(states)]
            action = self.computeActionFromValues(state)

            if action != None:
                qValueOfAction = self.computeQValueFromValues(state, action)
                self.values[state] = qValueOfAction


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predessesors = {}

        # make an empty set for each state
        for state in self.mdp.getStates():
            predessesors[state] = set()
        
        # add all predecessors of each state
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action): 
                    if prob > 0:
                        predessesors[nextState].add(state)
            
        # initialize empty priority queue
        priorQueue = util.PriorityQueue()

        for state in self.mdp.getStates():
            qValuesForState = []
            # if not a terminal state, compute diff
            if not self.mdp.isTerminal(state):
                # compute qValues for all possible actions
                for action in self.mdp.getPossibleActions(state):
                    qValue = self.computeQValueFromValues(state, action)
                    qValuesForState.append((qValue, action))

                # get the max qValue
                maxQValue = max(qValuesForState)[0]
                diff = abs(maxQValue - self.values[state])
                priorQueue.push(state, -diff)

        for i in range(self.iterations):
            if priorQueue.isEmpty():
                break
            state = priorQueue.pop()
            # after popping state from queue update it's values
            if not self.mdp.isTerminal(state):
                qValuesForState = []
                for action in self.mdp.getPossibleActions(state):
                    qValue = self.computeQValueFromValues(state, action)
                    qValuesForState.append((qValue, action))
                maxQValue = max(qValuesForState)[0]
                self.values[state] = maxQValue

            for predecessor in predessesors[state]:
                # if not a terminal state, compute diff
                if not self.mdp.isTerminal(predecessor):
                    qValuesForState = []
                    # compute qValues for all possible actions
                    for action in self.mdp.getPossibleActions(predecessor):
                        qValue = self.computeQValueFromValues(predecessor, action)
                        qValuesForState.append((qValue, action))

                    # get the max qValue
                    maxQValue = max(qValuesForState)[0]
                    diff = abs(maxQValue - self.values[predecessor])
                    if diff > self.theta:
                        priorQueue.update(predecessor, -diff)



        
