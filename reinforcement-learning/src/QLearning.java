import java.util.*;

/**
 * @author Hieu Le
 * @version 11/6/17
 * Implementation of the Q-learning reinforcement learning algorithm.
 */
public class QLearning {
    private static final double EPS = 1e-5;

    private final int[][] transitions;
    private final double[][] rewards;
    private final int goalState;
    private final int numStates;
    private final int numActions;

    private Random rng;
    private double[][] qualities;

    /**
     * Initializes the problem model ready to be trained and utilized
     * @param transitions mapping from state - action pair to next states
     * @param rewards numerical utility of taking an action from a given state
     * @param goalState final state
     */
    public QLearning(int[][] transitions, double[][] rewards, int goalState) {
        this.transitions = transitions;
        this.rewards = rewards;
        this.goalState = goalState;

        numStates = transitions.length;
        numActions = transitions[0].length;
        qualities = new double[numStates][numActions];
        rng = new Random(22061994);
    }

    /**
     * Trains the learning algorithm in a single episode
     * @param learningRate determines to what extent the newly acquired information will override
     *                     the information
     * @param discountFactor determines the importance of future rewards
     */
    public double train(final double learningRate, final double discountFactor) {
        int currentState = rng.nextInt(numStates);
        int currentAction = rng.nextInt(numActions);

        double totalQuality = 0.0;

        while (currentState != goalState) {
            int nextState = transitions[currentState][currentAction];

            int nextOptimalAction = 0;
            for (int nextAction = 1; nextAction < numActions; ++nextAction) {
                if (qualities[nextState][nextAction] > qualities[nextState][nextOptimalAction] + EPS) {
                    nextOptimalAction = nextAction;
                }
            }

            double maxQuality = qualities[nextState][nextOptimalAction];
            double currentQuality = qualities[currentState][currentAction];
            qualities[currentState][currentAction] = currentQuality +
                    learningRate * (rewards[currentState][currentAction] + discountFactor * maxQuality - currentQuality);
            totalQuality += qualities[currentState][currentAction];

            currentState = nextState;
            currentAction = nextOptimalAction;
        }

        return totalQuality;
    }

    /**
     * Generates a strategy to traverse from the given start state to the goal state
     * @return a list of actions to reach the goal state
     */
    public List<Integer> generateStrategy(final int startState) {
        int currentState = startState;
        boolean[] visited = new boolean[numStates];

        List<Integer> actions = new ArrayList<>();
        while (!visited[currentState] && currentState != goalState) {
            visited[currentState] = true;
            int optimalAction = 0;
            for (int action = 1; action < numActions; ++action) {
                if (qualities[currentState][action] > qualities[currentState][optimalAction] + EPS) {
                    optimalAction = action;
                }
            }

            actions.add(optimalAction);
            currentState = transitions[currentState][optimalAction];
        }

        return actions;
    }
}
