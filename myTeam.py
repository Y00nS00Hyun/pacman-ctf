# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from collections import deque
from game import Directions, Actions
from util import nearestPoint

SIGHT_RANGE = 5

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

_BELIEFS = {}
_INIT_DONE = False
_DEAD_END_DEPTH = None
_AGENT_TARGET = {}


def _computeDeadEndDepth(walls):
  width, height = walls.width, walls.height
  legal = set((x, y) for x in range(width) for y in range(height) if not walls[x][y])

  def neighborsOf(p):
    x, y = p
    out = []
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
      nx, ny = x + dx, y + dy
      if 0 <= nx < width and 0 <= ny < height and not walls[nx][ny]:
        out.append((nx, ny))
    return out

  alive = set(legal)
  depth = {p: 0 for p in legal}
  current = 1
  while True:
    to_remove = [p for p in alive
                 if sum(1 for n in neighborsOf(p) if n in alive) <= 1]
    if not to_remove:
      break
    for p in to_remove:
      depth[p] = current
      alive.discard(p)
    current += 1
    if current > 200:
      break
  return depth


def _initBeliefs(gameState, opponents):
  global _BELIEFS, _INIT_DONE
  if _INIT_DONE:
    return
  for opp in opponents:
    start = gameState.getInitialAgentPosition(opp)
    b = util.Counter()
    b[start] = 1.0
    _BELIEFS[opp] = b
  _INIT_DONE = True


class BaseAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.boundary = self._computeBoundary(gameState)
    self.midBoundary = self.boundary[len(self.boundary) // 2]

    walls = gameState.getWalls()
    self.walls = walls
    self.legalPositions = [(x, y) for x in range(walls.width)
                                  for y in range(walls.height)
                                  if not walls[x][y]]

    if self.index == min(self.getTeam(gameState)):
      _BELIEFS.clear()
      global _INIT_DONE, _DEAD_END_DEPTH
      _INIT_DONE = False
      _DEAD_END_DEPTH = None
    _initBeliefs(gameState, self.getOpponents(gameState))

    if _DEAD_END_DEPTH is None:
      globals()['_DEAD_END_DEPTH'] = _computeDeadEndDepth(walls)
    self.deadEndDepth = _DEAD_END_DEPTH

    self.boundaryDist = {}
    for pos in self.legalPositions:
      if self.boundary:
        self.boundaryDist[pos] = min(self.getMazeDistance(pos, b) for b in self.boundary)

    self.recentPositions = deque(maxlen=8)

  def _computeBoundary(self, gameState):
    layout = gameState.data.layout
    width, height = layout.width, layout.height
    x = (width // 2) - 1 if self.red else (width // 2)
    return [(x, y) for y in range(height) if not gameState.hasWall(x, y)]

  def _computePatrolPoints(self, gameState):
    if not self.boundary:
      return [self.midBoundary]
    n = len(self.boundary)
    indices = sorted(set([n // 4, n // 2, 3 * n // 4]))
    return [self.boundary[i] for i in indices]

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    return successor

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def distanceToHome(self, pos):
    if not self.boundary:
      return 0
    key = (int(pos[0]), int(pos[1]))
    if key in self.boundaryDist:
      return self.boundaryDist[key]
    return min(self.getMazeDistance(pos, b) for b in self.boundary)

  def isOurSide(self, pos):
    x = int(pos[0])
    midX = self.walls.width // 2
    return x < midX if self.red else x >= midX

  def _elapseTime(self, opp):
    newBelief = util.Counter()
    for pos, prob in list(_BELIEFS[opp].items()):
      if prob <= 0:
        continue
      possibilities = Actions.getLegalNeighbors(pos, self.walls)
      possibilities.append(pos)
      share = prob / len(possibilities)
      for n in possibilities:
        newBelief[n] += share
    _BELIEFS[opp] = newBelief

  def _observe(self, opp, gameState):
    myPos = gameState.getAgentPosition(self.index)
    noisyDistances = gameState.getAgentDistances()
    if noisyDistances is None:
      return
    obs = noisyDistances[opp]

    newBelief = util.Counter()
    for pos in self.legalPositions:
      trueDist = util.manhattanDistance(myPos, pos)
      if trueDist <= SIGHT_RANGE:
        continue
      prob = gameState.getDistanceProb(trueDist, obs)
      if prob > 0:
        newBelief[pos] = _BELIEFS[opp][pos] * prob

    if newBelief.totalCount() == 0:
      start = gameState.getInitialAgentPosition(opp)
      newBelief[start] = 1.0
    else:
      newBelief.normalize()
    _BELIEFS[opp] = newBelief

  def updateBeliefs(self, gameState):
    for opp in self.getOpponents(gameState):
      exactPos = gameState.getAgentPosition(opp)
      if exactPos is not None:
        b = util.Counter()
        b[exactPos] = 1.0
        _BELIEFS[opp] = b
      else:
        self._elapseTime(opp)
        self._observe(opp, gameState)

  def getMostLikelyPos(self, opp):
    belief = _BELIEFS.get(opp)
    if not belief or belief.totalCount() == 0:
      return None
    return max(belief.items(), key=lambda kv: kv[1])[0]

  def getEnemyEstimates(self, gameState):
    out = []
    for opp in self.getOpponents(gameState):
      visiblePos = gameState.getAgentPosition(opp)
      state = gameState.getAgentState(opp)
      if visiblePos is not None:
        out.append((opp, visiblePos, state.isPacman, True))
      else:
        est = self.getMostLikelyPos(opp)
        if est is None:
          continue
        isPac = self.isOurSide(est)
        out.append((opp, est, isPac, False))
    return out


class OffensiveAgent(BaseAgent):

  SAFE_DISTANCE = 5
  RETURN_THRESHOLD = 4

  def chooseAction(self, gameState):
    self.updateBeliefs(gameState)
    self.recentPositions.append(gameState.getAgentPosition(self.index))

    actions = gameState.getLegalActions(self.index)
    myState = gameState.getAgentState(self.index)
    timeleft = gameState.data.timeleft

    # 1. Emergency return: dynamic based on actual distance to home
    homeDistance = self.distanceToHome(gameState.getAgentPosition(self.index))
    if myState.numCarrying > 0 and timeleft < homeDistance * 4 + 20:
      return min(actions, key=lambda a: self.distanceToHome(
          self.getSuccessor(gameState, a).getAgentState(self.index).getPosition()))

    # 2. Endgame defense: winning big with low time
    if self._endgameDefenseMode(gameState):
      return self._endgameDefenseAction(gameState, actions)

    # 3. Secondary defense: chase nearby invader when on home side
    invaderChase = self._chaseNearbyInvader(gameState, actions)
    if invaderChase is not None:
      return invaderChase

    deadly = self._deadlyNextPositions(gameState)
    if deadly:
      safe = [a for a in actions if self._nextGridPos(gameState, a) not in deadly]
      if safe:
        actions = safe

    if self._inTacticalSituation(gameState):
      values = [self._minimaxValue(gameState, a) for a in actions]
    else:
      values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def _endgameDefenseMode(self, gameState):
    score = self.getScore(gameState)
    timeleft = gameState.data.timeleft
    myState = gameState.getAgentState(self.index)
    enemy_carrying = sum(gameState.getAgentState(opp).numCarrying
                         for opp in self.getOpponents(gameState))
    effective_score = score - enemy_carrying
    return effective_score >= 8 and timeleft < 400 and myState.numCarrying == 0

  def _endgameDefenseAction(self, gameState, actions):
    myState = gameState.getAgentState(self.index)
    # If in enemy territory, return home
    if myState.isPacman:
      return min(actions, key=lambda a: self.distanceToHome(
          self.getSuccessor(gameState, a).getAgentState(self.index).getPosition()))
    # On home side: chase invaders if visible
    invaderChase = self._chaseNearbyInvader(gameState, actions)
    if invaderChase is not None:
      return invaderChase
    # Patrol midBoundary
    return min(actions, key=lambda a: self.getMazeDistance(
        self.getSuccessor(gameState, a).getAgentState(self.index).getPosition(),
        self.midBoundary))

  def _chaseNearbyInvader(self, gameState, actions):
    myState = gameState.getAgentState(self.index)
    if myState.isPacman or myState.scaredTimer > 0:
      return None
    myPos = gameState.getAgentPosition(self.index)
    visibleInvaders = []
    for opp in self.getOpponents(gameState):
      oppPos = gameState.getAgentPosition(opp)
      oppState = gameState.getAgentState(opp)
      if oppPos is not None and oppState.isPacman:
        visibleInvaders.append(oppPos)
    if not visibleInvaders:
      return None
    closest = min(visibleInvaders, key=lambda p: self.getMazeDistance(myPos, p))
    if self.getMazeDistance(myPos, closest) > 8:
      return None
    scored = []
    for action in actions:
      successor = self.getSuccessor(gameState, action)
      nextPos = successor.getAgentState(self.index).getPosition()
      value = -self.getMazeDistance(nextPos, closest)
      if action == Directions.STOP:
        value -= 2
      scored.append((value, action))
    best = max(v for v, a in scored)
    return random.choice([a for v, a in scored if v == best])

  def _inTacticalSituation(self, gameState):
    myState = gameState.getAgentState(self.index)
    if not myState.isPacman:
      return False
    myPos = gameState.getAgentPosition(self.index)
    for opp in self.getOpponents(gameState):
      opp_pos = gameState.getAgentPosition(opp)
      opp_state = gameState.getAgentState(opp)
      if opp_pos is None or opp_state.isPacman or opp_state.scaredTimer > 1:
        continue
      if self.getMazeDistance(myPos, opp_pos) <= 6:
        return True
    return False

  def _minimaxValue(self, gameState, my_action):
    succ = self.getSuccessor(gameState, my_action)
    my_pos = succ.getAgentState(self.index).getPosition()
    base_my_pos = gameState.getAgentState(self.index).getPosition()

    if my_pos == self.start and base_my_pos != self.start:
      return -1e6

    # Collect all visible non-scared ghosts
    ghost_indices = []
    for opp in self.getOpponents(gameState):
      opp_pos = gameState.getAgentPosition(opp)
      opp_state = gameState.getAgentState(opp)
      if opp_pos is None or opp_state.isPacman or opp_state.scaredTimer > 1:
        continue
      ghost_indices.append(opp)

    if not ghost_indices:
      return self.evaluate(gameState, my_action)

    # Simulate each ghost independently; take min (most threatening)
    overall_worst = float('inf')
    for ghost_idx in ghost_indices:
      worst = float('inf')
      for ghost_action in succ.getLegalActions(ghost_idx):
        succ2 = succ.generateSuccessor(ghost_idx, ghost_action)
        my_pos2 = succ2.getAgentState(self.index).getPosition()
        if my_pos2 == self.start and base_my_pos != self.start:
          return -1e6
        v = self.evaluate(succ2, Directions.STOP)
        if v < worst:
          worst = v
      overall_worst = min(overall_worst, worst)
    return overall_worst

  def _nextGridPos(self, gameState, action):
    succ = self.getSuccessor(gameState, action)
    p = succ.getAgentState(self.index).getPosition()
    return (int(p[0]), int(p[1]))

  def _deadlyNextPositions(self, gameState):
    deadly = set()
    myPos = gameState.getAgentPosition(self.index)
    for opp in self.getOpponents(gameState):
      opp_pos = gameState.getAgentPosition(opp)
      opp_state = gameState.getAgentState(opp)
      if opp_state.isPacman:
        continue
      if opp_state.scaredTimer > 1:
        continue
      if opp_pos is None:
        # Particle filter estimate: only mark if very likely nearby
        est = self.getMostLikelyPos(opp)
        if est is not None and self.getMazeDistance(myPos, est) <= 2:
          deadly.add((int(est[0]), int(est[1])))
        continue
      possibilities = Actions.getLegalNeighbors(opp_pos, self.walls)
      possibilities.append(opp_pos)
      for p in possibilities:
        deadly.add((int(p[0]), int(p[1])))
    return deadly

  def _getThreats(self, gameState):
    threats = []
    for opp, pos, isPacGuess, visible in self.getEnemyEstimates(gameState):
      state = gameState.getAgentState(opp)
      if state.scaredTimer > 1:
        continue
      isGhost = (not state.isPacman) if visible else (not isPacGuess)
      if not isGhost:
        continue
      threats.append((opp, pos, visible))
    return threats

  def _getGhostStatus(self, gameState):
    scared, active = [], []
    for opp in self.getOpponents(gameState):
      s = gameState.getAgentState(opp)
      if s.isPacman:
        continue
      (scared if s.scaredTimer > 1 else active).append(opp)
    return scared, active

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)
    if foodList:
      nearThreats = self._getThreats(gameState)
      ghostClose = False
      if nearThreats and myState.isPacman:
        minThreatDist = min(self.getMazeDistance(myPos, p) for _, p, _ in nearThreats)
        if minThreatDist <= self.SAFE_DISTANCE:
          ghostClose = True
      if ghostClose:
        safeFood = [f for f in foodList if self.deadEndDepth.get(f, 0) == 0]
        target = safeFood if safeFood else foodList
      else:
        target = foodList
      features['distanceToFood'] = min(self.getMazeDistance(myPos, f) for f in target)

    features['carriedFood'] = myState.numCarrying
    features['distanceToHome'] = self.distanceToHome(myPos)

    threats = self._getThreats(gameState)
    minGhostDist = None
    if myState.isPacman and threats:
      ghostDists = []
      for opp, pos, visible in threats:
        d = self.getMazeDistance(myPos, pos)
        ghostDists.append((d, visible))
      visibleClose = [d for d, v in ghostDists if v and d <= 3]
      if visibleClose:
        minGhostDist = min(visibleClose)
      else:
        minGhostDist = min(d for d, v in ghostDists)
      features['ghostDistance'] = min(minGhostDist, self.SAFE_DISTANCE)
      if minGhostDist <= 1:
        features['ghostThreat'] = 1
    else:
      features['ghostDistance'] = self.SAFE_DISTANCE

    if myState.isPacman and minGhostDist is not None and minGhostDist <= self.SAFE_DISTANCE:
      posKey = (int(myPos[0]), int(myPos[1]))
      features['deadEndDepth'] = self.deadEndDepth.get(posKey, 0)

    capsules = self.getCapsules(successor)
    if capsules and myState.isPacman:
      features['distanceToCapsule'] = min(self.getMazeDistance(myPos, c) for c in capsules)
    else:
      features['distanceToCapsule'] = 0

    scaredDists = []
    for opp in self.getOpponents(successor):
      oppState = successor.getAgentState(opp)
      oppPos = successor.getAgentPosition(opp)
      if oppState.scaredTimer > 3 and not oppState.isPacman and oppPos is not None:
        scaredDists.append(self.getMazeDistance(myPos, oppPos))
    features['distanceToScaredGhost'] = min(scaredDists) if scaredDists else 0

    if action == Directions.STOP:
      features['stop'] = 1

    succPos = (int(myPos[0]), int(myPos[1]))
    visits = sum(1 for p in self.recentPositions if p == succPos)
    if visits >= 2:
      features['loopVisits'] = visits * visits

    return features

  def getWeights(self, gameState, action):
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    carrying = myState.numCarrying

    threats = self._getThreats(gameState)
    ghostNear = False
    minGhostDistW = float('inf')
    if myState.isPacman and threats:
      minGhostDistW = min(self.getMazeDistance(myPos, p) for _, p, _ in threats)
      if minGhostDistW <= self.SAFE_DISTANCE:
        ghostNear = True

    scaredGhosts, activeGhosts = self._getGhostStatus(gameState)

    weights = {
      'successorScore': 100,
      'distanceToFood': -1,
      'ghostDistance': 20,
      'ghostThreat': -1000,
      'stop': -100,
      'carriedFood': 0,
      'deadEndDepth': -15,
      'distanceToCapsule': 0,
      'loopVisits': -8,
    }

    homeDistance = self.distanceToHome(myPos)

    # Instant return: near boundary with any food
    if carrying >= 1 and homeDistance <= 2:
      weights['distanceToHome'] = -200
      return weights

    # Comeback return: carrying enough food to flip the score
    score = self.getScore(gameState)
    if score < 0 and carrying >= abs(score) + 1:
      weights['distanceToHome'] = -300
      return weights

    # All-but-two endgame: sprint aggressively for remaining food
    remaining = len(self.getFood(gameState).asList())
    if remaining <= 4:
      weights['successorScore'] = 500
      weights['distanceToFood'] = -5
      weights['deadEndDepth'] = -5

    if scaredGhosts and not activeGhosts:
      # All ghosts scared: ignore avoidance entirely, chase scared ghosts
      weights['ghostDistance'] = 0
      weights['ghostThreat'] = 0
      weights['deadEndDepth'] = 0
      weights['distanceToHome'] = 0
      weights['distanceToScaredGhost'] = -8
      return weights
    elif scaredGhosts and activeGhosts:
      # Mixed: keep active ghost avoidance, but also pursue scared ghosts
      weights['distanceToScaredGhost'] = -8

    # Proactive capsule: ghost approaching from 10 steps out
    if myState.isPacman and threats:
      if minGhostDistW <= 10:
        weights['distanceToCapsule'] = -4
    if ghostNear:
      weights['distanceToCapsule'] = -25

    effective_threshold = self.RETURN_THRESHOLD
    if score <= -3:
      effective_threshold = max(2, self.RETURN_THRESHOLD - 1)
    elif score >= 5:
      effective_threshold = self.RETURN_THRESHOLD + 1

    if carrying >= effective_threshold or ghostNear:
      weights['distanceToHome'] = -15 - 3 * carrying
    else:
      weights['distanceToHome'] = 0

    return weights


class DefensiveAgent(BaseAgent):

  def registerInitialState(self, gameState):
    BaseAgent.registerInitialState(self, gameState)
    self.lastFood = self.getFoodYouAreDefending(gameState).asList()
    self.currentTarget = self.midBoundary
    self.patrolPoints = self._computePatrolPoints(gameState)
    self.patrolIdx = 0

  def chooseAction(self, gameState):
    self.updateBeliefs(gameState)
    self.recentPositions.append(gameState.getAgentPosition(self.index))
    myPos = gameState.getAgentPosition(self.index)

    # Emergency offense: losing badly with low time
    if self._emergencyOffenseMode(gameState):
      return self._emergencyOffenseAction(gameState)

    # Advance patrol index when near current patrol point
    if self.getMazeDistance(myPos, self.patrolPoints[self.patrolIdx]) <= 1:
      self.patrolIdx = (self.patrolIdx + 1) % len(self.patrolPoints)

    currentFood = self.getFoodYouAreDefending(gameState).asList()
    eatenPositions = set(self.lastFood) - set(currentFood)
    if eatenPositions:
      eaten = next(iter(eatenPositions))
      bestOpp, bestProb = None, -1
      for opp in self.getOpponents(gameState):
        prob = _BELIEFS.get(opp, util.Counter())[eaten]
        if prob > bestProb:
          bestProb, bestOpp = prob, opp
      if bestOpp is not None:
        b = util.Counter()
        b[eaten] = 1.0
        _BELIEFS[bestOpp] = b
    self.lastFood = currentFood

    self.currentTarget = self._pickTarget(gameState)

    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def _emergencyOffenseMode(self, gameState):
    score = self.getScore(gameState)
    timeleft = gameState.data.timeleft
    return score <= -6 and timeleft < 800

  def _emergencyOffenseAction(self, gameState):
    myState = gameState.getAgentState(self.index)
    myPos = gameState.getAgentPosition(self.index)
    foodList = self.getFood(gameState).asList()
    actions = gameState.getLegalActions(self.index)
    if not actions:
      return Directions.STOP
    # Bank carried food before attacking
    if myState.numCarrying > 0:
      return min(actions, key=lambda a: self.distanceToHome(
          self.getSuccessor(gameState, a).getAgentState(self.index).getPosition()))
    if not foodList:
      return random.choice(actions)

    threats = []
    for opp in self.getOpponents(gameState):
      oppPos = gameState.getAgentPosition(opp)
      oppState = gameState.getAgentState(opp)
      if oppPos is not None and not oppState.isPacman and oppState.scaredTimer == 0:
        threats.append(oppPos)

    best_action = None
    best_score = float('-inf')
    for a in actions:
      if a == Directions.STOP:
        continue
      succ = self.getSuccessor(gameState, a)
      pos = succ.getAgentState(self.index).getPosition()
      food_score = -min(self.getMazeDistance(pos, f) for f in foodList)
      ghost_penalty = 0
      if threats and succ.getAgentState(self.index).isPacman:
        gd = min(self.getMazeDistance(pos, g) for g in threats)
        if gd <= 1:
          ghost_penalty = -1000
        elif gd <= 3:
          ghost_penalty = -100
      score = food_score + ghost_penalty
      if score > best_score:
        best_score = score
        best_action = a

    return best_action if best_action is not None else random.choice(actions)

  def _pickTarget(self, gameState):
    myPos = gameState.getAgentPosition(self.index)
    myFood = self.getFoodYouAreDefending(gameState).asList()
    myCapsules = self.getCapsulesYouAreDefending(gameState)
    important = myFood + myCapsules

    visibleInvaders = []
    estimatedInvaders = []
    for opp in self.getOpponents(gameState):
      visPos = gameState.getAgentPosition(opp)
      state = gameState.getAgentState(opp)
      if visPos is not None and state.isPacman:
        visibleInvaders.append(visPos)
      elif visPos is None:
        est = self.getMostLikelyPos(opp)
        if est is not None and self.isOurSide(est):
          estimatedInvaders.append(est)

    # Direct chase when invader is visible
    if visibleInvaders:
      return min(visibleInvaders, key=lambda p: self.getMazeDistance(myPos, p))

    def threatScore(invPos):
      if important:
        return min(self.getMazeDistance(invPos, f) for f in important)
      return self.getMazeDistance(myPos, invPos)

    if estimatedInvaders:
      invaderPos = min(estimatedInvaders, key=threatScore)
      if important:
        return min(important, key=lambda f: self.getMazeDistance(invaderPos, f))
      return invaderPos

    # No invaders: guard the food most vulnerable to quick grabs (closest to entry boundary)
    if myFood:
      return min(myFood, key=lambda f: self.distanceToHome(f))
    return self.patrolPoints[self.patrolIdx]

  def _getInvaders(self, gameState, successor):
    invaders = []
    for opp, pos, isPacGuess, visible in self.getEnemyEstimates(gameState):
      if visible:
        sState = successor.getAgentState(opp)
        sPos = sState.getPosition()
        if sState.isPacman and sPos is not None:
          invaders.append((opp, sPos, True))
      else:
        if isPacGuess:
          invaders.append((opp, pos, False))
    return invaders

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    features['onDefense'] = 0 if myState.isPacman else 1

    invaders = self._getInvaders(gameState, successor)
    visibleInvaders = [i for i in invaders if i[2]]
    features['numInvaders'] = len(visibleInvaders)

    if invaders:
      if visibleInvaders:
        dists = [self.getMazeDistance(myPos, p) for _, p, _ in visibleInvaders]
      else:
        dists = [self.getMazeDistance(myPos, p) for _, p, _ in invaders]
      features['invaderDistance'] = min(dists)

    features['distanceToTarget'] = self.getMazeDistance(myPos, self.currentTarget)

    if myState.scaredTimer > 0 and invaders:
      closest = min(self.getMazeDistance(myPos, p) for _, p, _ in invaders)
      if closest < 5:
        features['scaredAvoid'] = 1

    if action == Directions.STOP:
      features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev:
      features['reverse'] = 1

    succPos = (int(myPos[0]), int(myPos[1]))
    visits = sum(1 for p in self.recentPositions if p == succPos)
    if visits >= 2:
      features['loopVisits'] = visits * visits

    return features

  def getWeights(self, gameState, action):
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()

    weights = {
      'onDefense': 10000,
      'numInvaders': -1000,
      'invaderDistance': -3,
      'distanceToTarget': -5,
      'scaredAvoid': -500,
      'stop': -100,
      'reverse': -2,
      'loopVisits': -8,
    }

    visibleInv = []
    for opp in self.getOpponents(gameState):
      pos = gameState.getAgentPosition(opp)
      if pos is not None and gameState.getAgentState(opp).isPacman:
        visibleInv.append(pos)

    if visibleInv and myState.scaredTimer == 0:
      minDist = min(self.getMazeDistance(myPos, p) for p in visibleInv)
      if minDist <= 6:
        weights['invaderDistance'] = -20
        weights['distanceToTarget'] = 0

    if myState.scaredTimer > 0:
      visibleInvaderPositions = [
        gameState.getAgentPosition(opp)
        for opp in self.getOpponents(gameState)
        if gameState.getAgentPosition(opp) is not None
        and gameState.getAgentState(opp).isPacman
      ]
      if visibleInvaderPositions:
        if myState.scaredTimer > 10:
          weights['invaderDistance'] = 12  # shadow: stay at safe distance
        else:
          weights['invaderDistance'] = 30  # timer almost up: full flee
        weights['numInvaders'] = 0
        weights['distanceToTarget'] = 0
      else:
        weights['invaderDistance'] = 0

    return weights
