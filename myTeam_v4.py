# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
from util import nearestPoint

SIGHT_RANGE = 5  # mirrors capture.SIGHT_RANGE — agents see opponents within this

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


###############################
# Shared belief state (team)  #
###############################
# Both teammates share these dicts so a sighting by one informs the other.
# Keyed by opponent agent index.
_BELIEFS = {}       # opp_index -> util.Counter over positions
_INIT_DONE = False  # one-time init guard
_DEAD_END_DEPTH = None  # dict: position -> int depth (0 = junction/loop, >0 = N steps into dead-end)


def _computeDeadEndDepth(walls):
  """Iteratively peel cells with <=1 alive neighbor.
  Depth N = N peels needed to remove this cell (i.e., N steps into a dead-end branch).
  Cells on loops / junctions end with depth 0."""
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
    if current > 200:  # safety
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


##############
# Base class #
##############

class BaseAgent(CaptureAgent):
  """
  Shared utilities + particle-filter–style belief tracking for hidden enemies.
  """

  def registerInitialState(self, gameState):
    global _INIT_DONE, _BELIEFS
    # Each new game restarts agent objects but the module-level globals persist
    # across `-n` runs. Reset on the first agent's first init of a new game.
    if self.index == self.getTeam(gameState)[0] if hasattr(self, 'red') else False:
      pass  # we reset below using a different signal

    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.boundary = self._computeBoundary(gameState)
    self.midBoundary = self.boundary[len(self.boundary) // 2]

    walls = gameState.getWalls()
    self.walls = walls
    self.legalPositions = [(x, y) for x in range(walls.width)
                                  for y in range(walls.height)
                                  if not walls[x][y]]

    # Reset beliefs on every fresh game (first agent on team triggers it).
    if self.index == min(self.getTeam(gameState)):
      _BELIEFS.clear()
      global _INIT_DONE, _DEAD_END_DEPTH
      _INIT_DONE = False
      _DEAD_END_DEPTH = None
    _initBeliefs(gameState, self.getOpponents(gameState))

    # Dead-end depth map (computed once per game; shared across agents)
    if _DEAD_END_DEPTH is None:
      globals()['_DEAD_END_DEPTH'] = _computeDeadEndDepth(walls)
    self.deadEndDepth = _DEAD_END_DEPTH

  def _computeBoundary(self, gameState):
    layout = gameState.data.layout
    width, height = layout.width, layout.height
    x = (width // 2) - 1 if self.red else (width // 2)
    return [(x, y) for y in range(height) if not gameState.hasWall(x, y)]

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
    return min(self.getMazeDistance(pos, b) for b in self.boundary)

  def isOurSide(self, pos):
    x = int(pos[0])
    midX = self.walls.width // 2
    return x < midX if self.red else x >= midX

  # -------------------------------------------------------------------
  # Particle-filter–style belief updates
  # -------------------------------------------------------------------

  def _elapseTime(self, opp):
    """One opponent move: distribute mass uniformly over legal neighbors + STOP."""
    newBelief = util.Counter()
    for pos, prob in list(_BELIEFS[opp].items()):
      if prob <= 0:
        continue
      possibilities = Actions.getLegalNeighbors(pos, self.walls)
      possibilities.append(pos)  # STOP option
      share = prob / len(possibilities)
      for n in possibilities:
        newBelief[n] += share
    _BELIEFS[opp] = newBelief

  def _observe(self, opp, gameState):
    """Bayes update using noisy distance reading from this agent's position."""
    myPos = gameState.getAgentPosition(self.index)
    noisyDistances = gameState.getAgentDistances()
    if noisyDistances is None:
      return
    obs = noisyDistances[opp]

    newBelief = util.Counter()
    for pos in self.legalPositions:
      trueDist = util.manhattanDistance(myPos, pos)
      # If they were within sight, we'd see them. Since we don't, prob=0.
      if trueDist <= SIGHT_RANGE:
        continue
      prob = gameState.getDistanceProb(trueDist, obs)
      if prob > 0:
        newBelief[pos] = _BELIEFS[opp][pos] * prob

    if newBelief.totalCount() == 0:
      # Belief collapsed — likely respawn or filter divergence. Reset to start.
      start = gameState.getInitialAgentPosition(opp)
      newBelief[start] = 1.0
    else:
      newBelief.normalize()
    _BELIEFS[opp] = newBelief

  def updateBeliefs(self, gameState):
    """Call at the top of chooseAction."""
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
    """For each opponent, return (index, estPos, isPacmanGuess, visible)."""
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
        # Infer Pacman status from our side check (configuration is None when invisible)
        isPac = self.isOurSide(est)
        out.append((opp, est, isPac, False))
    return out


###################
# Offensive Agent #
###################

class OffensiveAgent(BaseAgent):
  """
  Goes into enemy territory to eat food.
  Uses belief estimates so it avoids ghosts even when they're out of sight.
  """

  SAFE_DISTANCE = 5
  RETURN_THRESHOLD = 5

  def chooseAction(self, gameState):
    self.updateBeliefs(gameState)
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def _getThreats(self, gameState):
    """Enemy ghosts (on their side, non-scared) — use visible pos or estimate."""
    threats = []
    for opp, pos, isPacGuess, visible in self.getEnemyEstimates(gameState):
      state = gameState.getAgentState(opp)
      if state.scaredTimer > 1:
        continue
      # Threat means: opponent is currently a ghost in their territory.
      # If visible we trust state.isPacman; if not, infer from estimated pos.
      isGhost = (not state.isPacman) if visible else (not isPacGuess)
      if not isGhost:
        continue
      threats.append((opp, pos, visible))
    return threats

  def _enemiesScared(self, gameState):
    """True if any non-pacman opponent is currently scared (we ate a capsule)."""
    for opp in self.getOpponents(gameState):
      s = gameState.getAgentState(opp)
      if s.scaredTimer > 1 and not s.isPacman:
        return True
    return False

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # --- Food ---
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)
    if foodList:
      features['distanceToFood'] = min(self.getMazeDistance(myPos, f) for f in foodList)

    # --- Return logic ---
    features['carriedFood'] = myState.numCarrying
    features['distanceToHome'] = self.distanceToHome(myPos)

    # --- Ghost threat (visible OR estimated) ---
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

    # --- Dead-end avoidance: punish stepping deeper into dead-ends when chased ---
    if myState.isPacman and minGhostDist is not None and minGhostDist <= self.SAFE_DISTANCE:
      posKey = (int(myPos[0]), int(myPos[1]))
      features['deadEndDepth'] = self.deadEndDepth.get(posKey, 0)

    # --- Capsule pickup: when threatened, capsule may be the best escape ---
    capsules = self.getCapsules(successor)
    if capsules and myState.isPacman:
      features['distanceToCapsule'] = min(self.getMazeDistance(myPos, c) for c in capsules)
    else:
      features['distanceToCapsule'] = 0

    if action == Directions.STOP:
      features['stop'] = 1

    return features

  def getWeights(self, gameState, action):
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    carrying = myState.numCarrying

    threats = self._getThreats(gameState)
    ghostNear = False
    minDist = None
    if myState.isPacman and threats:
      minDist = min(self.getMazeDistance(myPos, p) for _, p, _ in threats)
      if minDist <= self.SAFE_DISTANCE:
        ghostNear = True

    enemiesScared = self._enemiesScared(gameState)

    weights = {
      'successorScore': 100,
      'distanceToFood': -1,
      'ghostDistance': 20,
      'ghostThreat': -1000,
      'stop': -100,
      'carriedFood': 0,
      'deadEndDepth': -15,  # each step deeper into a dead-end is costly when chased
      'distanceToCapsule': 0,
    }

    # --- Scared mode: enemies can't hurt us, go aggressive ---
    if enemiesScared:
      weights['ghostDistance'] = 0     # no need to keep distance
      weights['ghostThreat'] = 0
      weights['deadEndDepth'] = 0      # dead-ends are safe
      weights['distanceToHome'] = 0    # don't retreat; keep eating
      return weights

    # --- Capsule: prefer it when ghost is close ---
    if ghostNear:
      # Slightly stronger pull than home so we prefer capsule if comparable distance
      weights['distanceToCapsule'] = -12

    # --- Return logic (only when not in scared mode) ---
    if carrying >= self.RETURN_THRESHOLD or ghostNear:
      weights['distanceToHome'] = -10 - 2 * carrying
    else:
      weights['distanceToHome'] = 0

    return weights


###################
# Defensive Agent #
###################

class DefensiveAgent(BaseAgent):
  """
  Patrols boundary; chases visible invaders; uses belief estimates to track
  invaders we can't currently see.
  """

  def registerInitialState(self, gameState):
    BaseAgent.registerInitialState(self, gameState)
    self.lastFood = self.getFoodYouAreDefending(gameState).asList()
    self.currentTarget = self.midBoundary

  def chooseAction(self, gameState):
    self.updateBeliefs(gameState)

    # Food disappearance → invader was there; collapse belief
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

    # Smart target selection (intercept the food invader is heading for)
    self.currentTarget = self._pickTarget(gameState)

    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def _pickTarget(self, gameState):
    """
    Smart target:
      - Closest visible invader → food (we're defending) closest to them = intercept point
      - Else estimated invader on our side → same intercept logic with estimate
      - Else patrol middle of boundary
    """
    myPos = gameState.getAgentPosition(self.index)

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

    invaderPos = None
    if visibleInvaders:
      invaderPos = min(visibleInvaders, key=lambda p: self.getMazeDistance(myPos, p))
    elif estimatedInvaders:
      invaderPos = min(estimatedInvaders, key=lambda p: self.getMazeDistance(myPos, p))

    if invaderPos is not None:
      myFood = self.getFoodYouAreDefending(gameState).asList()
      myCapsules = self.getCapsulesYouAreDefending(gameState)
      important = myFood + myCapsules
      if important:
        # Intercept at the food/capsule the invader is closest to
        return min(important, key=lambda f: self.getMazeDistance(invaderPos, f))
      return invaderPos

    return self.midBoundary

  def _getInvaders(self, gameState, successor):
    """Visible OR estimated invaders (Pacmen on our side)."""
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

    # Always pull toward chosen target (intercept point / patrol point)
    features['distanceToTarget'] = self.getMazeDistance(myPos, self.currentTarget)

    # Avoid getting eaten while scared — keep distance >= 3
    if myState.scaredTimer > 0 and invaders:
      closest = min(self.getMazeDistance(myPos, p) for _, p, _ in invaders)
      if closest < 3:
        features['scaredAvoid'] = 1

    if action == Directions.STOP:
      features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev:
      features['reverse'] = 1

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
    }

    # When a visible invader is close (and we're not scared), drop everything and chase.
    visibleInv = []
    for opp in self.getOpponents(gameState):
      pos = gameState.getAgentPosition(opp)
      if pos is not None and gameState.getAgentState(opp).isPacman:
        visibleInv.append(pos)

    if visibleInv and myState.scaredTimer == 0:
      minDist = min(self.getMazeDistance(myPos, p) for p in visibleInv)
      if minDist <= 3:
        weights['invaderDistance'] = -20
        weights['distanceToTarget'] = 0

    # Scared: never chase, just maintain distance
    if myState.scaredTimer > 0:
      weights['invaderDistance'] = 0

    return weights
