/**
 * @author Hieu Le
 * @version 11/6/17
 * Implementation of the SARSA reinforcement learning algorithm.
 */
public class Sarsa extends TemporalDifferenceLearning {

    public Sarsa(int[][] transitions, double[][] rewards, int goalState) {
        super(transitions, rewards, goalState);
    }

    @Override
    public double train(final double learningRate, final double discountFactor) {
        int currentState = selectState();
        int currentAction = selectAction(currentState);

        double totalQuality = 0.0;

        while (currentState != goalState) {
            int nextState = transitions[currentState][currentAction];
            int nextAction = selectAction(nextState);

            double currentQuality = qualities[currentState][currentAction];
            qualities[currentState][currentAction] = currentQuality +
                    learningRate * (rewards[currentState][currentAction] +
                            discountFactor * qualities[nextState][nextAction] - currentQuality);
            totalQuality += qualities[currentState][currentAction];

            currentState = nextState;
            currentAction = nextAction;
        }

        return totalQuality;
    }
}
