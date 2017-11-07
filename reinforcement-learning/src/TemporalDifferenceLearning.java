import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * @author Hieu Le
 * @version 11/6/17
 * Base class for a Temporal Difference Learning algorithm.
 */
public abstract class TemporalDifferenceLearning {

    protected final int[][] transitions;
    protected final double[][] rewards;
    protected final int goalState;
    protected final int numStates;
    protected final int numActions;
    protected double[][] qualities;

    private Random rng;
    private static final double EPS = 0.1;

    /**
     * Initializes the problem model ready to be trained and utilized
     * @param transitions mapping from state - action pair to next states
     * @param rewards numerical utility of taking an action from a given state
     * @param goalState final state
     */
    public TemporalDifferenceLearning(int[][] transitions, double[][] rewards, int goalState) {
        this.transitions = transitions;
        this.rewards = rewards;
        this.goalState = goalState;

        numStates = transitions.length;
        numActions = transitions[0].length;
        qualities = new double[numStates][numActions];
        rng = new Random(22061994);
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

    /**
     * Selects an arbitrary state to explore
     */
    protected int selectState() {
        return rng.nextInt(numStates);
    }

    /**
     * Selects the next action for a given state based on epsilon-greedy policy
     */
    protected int selectAction(int state) {
        int selectedAction = -1;
        int[] choices = new int[numActions];
        double maxQuality = Double.MIN_VALUE;
        int maxDV = 0;

        if (rng.nextDouble() < EPS) {
            selectedAction = -1;
        } else {
            for (int action = 0; action < numActions; ++action) {
                if (qualities[state][action] > maxQuality) {
                    selectedAction = action;
                    maxQuality = qualities[state][action];
                    maxDV = 0;
                    choices[maxDV] = selectedAction;
                } else if (qualities[state][action] == maxQuality) {
                    ++maxDV;
                    choices[maxDV] = action;
                }
            }

            if (maxDV > 0) {
                selectedAction = choices[rng.nextInt(maxDV + 1)];
            }
        }

        // Select random action if all qualities are 0 or exploring.
        if (selectedAction == -1) {
            selectedAction = rng.nextInt(numActions);
        }

        return selectedAction;
    }

    /**
     * Trains the learning algorithm in a single episode
     * @param learningRate determines to what extent the newly acquired information will override
     *                     the information
     * @param discountFactor determines the importance of future rewards
     */
    public abstract double train(final double learningRate, final double discountFactor);
}
