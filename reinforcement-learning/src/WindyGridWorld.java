import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * @author Hieu Le
 * @version 11/6/17
 * Application of reinforcement learning algorithms to solve the "windy grid world" problem.
 */
public class WindyGridWorld {
    private static final int NUM_ACTIONS = 4;
    private static final int[] DR = {-1, 0, 1, 0};
    private static final int[] DC = {0, 1, 0, -1};
    private static final String[] ACTIONS = {"Up", "Right", "Down", "Left"};

    /**
     * Converts a position on the grid to its index according to row-major order
     */
    private static int getStateIndex(int row, int col, int[][] grid) {
        return row * grid[row].length + col;
    }

    /**
     * Restricts a given position to the range [0, dimension)
     */
    private static int sanitize(int position, int dimension) {
        return Math.min(dimension - 1, Math.max(position, 0));
    }

    /**
     * Determines the shortest number of steps to reach at goalState from startState using
     * Bread-First Search. Used to verify the output of Q-Learning.
     */
    private static List<Integer> bfs(int[][] transitions, int startState, int goalState) {
        int[] lastStates = new int[transitions.length];
        int[] lastActions = new int[transitions.length];
        Arrays.fill(lastStates, -1);
        Arrays.fill(lastActions, -1);

        Queue<Integer> frontier = new LinkedList<>();
        frontier.add(startState);
        lastStates[startState] = lastActions[startState] = 0;

        while (!frontier.isEmpty()) {
            int size = frontier.size();
            for (int i = 0; i < size; ++i) {
                int currentState = frontier.poll();

                for (int action = 0; action < NUM_ACTIONS; ++action) {
                    int nextState = transitions[currentState][action];
                    if (lastStates[nextState] == -1) {
                        lastStates[nextState] = currentState;
                        lastActions[nextState] = action;
                        frontier.add(nextState);
                    }
                }
            }
        }

        if (lastStates[goalState] == -1)
            return null;

        LinkedList<Integer> actions = new LinkedList<>();
        int currentState = goalState;
        while (currentState != startState) {
            actions.addFirst(lastActions[currentState]);
            currentState = lastStates[currentState];
        }

        return actions;
    }

    private static void printPath(int[][] transitions, double[][] rewards, int startState,
                                  int goalState, List<Integer> actions) {
        int numCols = 10;
        double totalReward = 0.0;
        int currentState = startState;
        for (int action : actions) {
            System.err.printf("%d %d %s\n", currentState / numCols, currentState % numCols, ACTIONS[action]);
            totalReward += rewards[currentState][action];
            currentState = transitions[currentState][action];
        }

        if (currentState != goalState) {
            throw new RuntimeException("Goal state is not reached");
        }

        System.err.printf("Optimal reward: %f\n", totalReward);
        System.err.printf("Optimal number of steps: %d\n", actions.size());
    }

    public static void main(String[] args) {
        final int[][] gridWorld = {
                {0, 0, 0, 1, 1, 1, 2, 2, 1, 0},
                {0, 0, 0, 1, 1, 1, 2, 2, 1, 0},
                {0, 0, 0, 1, 1, 1, 2, 2, 1, 0},
                {0, 0, 0, 1, 1, 1, 2, 2, 1, 0},
                {0, 0, 0, 1, 1, 1, 2, 2, 1, 0},
                {0, 0, 0, 1, 1, 1, 2, 2, 1, 0},
                {0, 0, 0, 1, 1, 1, 2, 2, 1, 0},
        };

        final int numRows = gridWorld.length;
        final int numCols = gridWorld[0].length;
        final int numStates = numRows * numCols;
        final int startState = getStateIndex(3, 0, gridWorld);
        final int goalState = getStateIndex(3, 7, gridWorld);

        int[][] transitions = new int[numStates][NUM_ACTIONS];
        double[][] rewards = new double[numStates][NUM_ACTIONS];

        for (int row = 0; row < numRows; ++row) {
            for (int col = 0; col < numCols; ++col) {
                int currentState = row * numCols + col;
                for (int action = 0; action < NUM_ACTIONS; ++action) {
                    int nextRow = sanitize(row + DR[action], numRows);
                    int nextCol = sanitize(col + DC[action], numCols);
                    // Pushes the agent up if it moves onto a windy column.
                    nextRow = sanitize(nextRow - gridWorld[nextRow][nextCol], numRows);

                    int nextState = getStateIndex(nextRow, nextCol, gridWorld);
                    transitions[currentState][action] = nextState;

                    rewards[currentState][action] = nextState == goalState ? 100 : -1;
                }
            }
        }

        QLearning learner = new QLearning(transitions, rewards, goalState);
        final int NUM_EPISODES = 8000;
        for (int episode = 0; episode < NUM_EPISODES; ++episode) {
            double totalQuality = learner.train(0.10, 0.80);
            System.out.printf("%d %f\n", episode, totalQuality);
        }

        System.err.println("Q-LEARNING OUTPUT:");
        List<Integer> actualActions = learner.generateStrategy(startState);
        printPath(transitions, rewards, startState, goalState, actualActions);

        System.err.println("-----------------------------------------------");
        System.err.println("EXPECTED OUTPUT:");
        printPath(transitions, rewards, startState, goalState, bfs(transitions, startState, goalState));
    }
}
