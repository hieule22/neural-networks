/**
 * @author Hieu Le
 * @version 11/6/17
 * Implementation of the Q-learning reinforcement learning algorithm.
 */
public class QLearning extends TemporalDifferenceLearning {

    public QLearning(int[][] transitions, double[][] rewards, int goalState) {
        super(transitions, rewards, goalState);
    }

    @Override
    public double train(final double learningRate, final double discountFactor) {
        int currentState = selectState();

        double totalQuality = 0.0;

        while (currentState != goalState) {
            int currentAction = selectAction(currentState);
            int nextState = transitions[currentState][currentAction];

            int nextOptimalAction = 0;
            for (int nextAction = 1; nextAction < numActions; ++nextAction) {
                if (qualities[nextState][nextAction] > qualities[nextState][nextOptimalAction]) {
                    nextOptimalAction = nextAction;
                }
            }

            double maxQuality = qualities[nextState][nextOptimalAction];
            double currentQuality = qualities[currentState][currentAction];
            qualities[currentState][currentAction] = currentQuality +
                    learningRate * (rewards[currentState][currentAction] + discountFactor * maxQuality - currentQuality);
            totalQuality += qualities[currentState][currentAction];

            currentState = nextState;
        }

        return totalQuality;
    }
}
