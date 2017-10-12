import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Name: Hieu Le - htl5683@truman.edu
 * Implementation of the Learning Vector Quantization (LVQ) clustering algorithm.
 */

public class LVQ
{
    private static final int NUMBER_OF_CLUSTERS = 6;
    private static final int VEC_LEN = 2;
    private static final int TRAINING_PATTERNS = 42;

    private static final double DECAY_RATE = 0.96; // About 100 iterations.
    private static final double MIN_ALPHA = 0.01;

    private static double alpha = 0.6;
    private static double d[] = null; // Network nodes. The "clusters"

    //Weight matrix.
    private static double weights[][] = null;

    //Training patterns.
    private static int mPattern[][] = null;
    private static int mTarget[] = null;
    private static String[] mFontNames;

    private static void initialize() {
        List<List<int[]>> clusters = new ArrayList<>(NUMBER_OF_CLUSTERS);
        for (int i = 0; i < NUMBER_OF_CLUSTERS; ++i)
            clusters.add(new ArrayList<>(TRAINING_PATTERNS / NUMBER_OF_CLUSTERS));

        int size = 0;
        Random rng = new Random(22061994);
        while (size < TRAINING_PATTERNS) {
            int x = rng.nextInt(21) - 10;
            int y = rng.nextInt(21) - 10;
            int category = getCategory(x, y);

            if (clusters.get(category).size() < TRAINING_PATTERNS / NUMBER_OF_CLUSTERS) {
                clusters.get(category).add(new int[]{x, y});
                ++size;
            }
        }

        mPattern = new int[TRAINING_PATTERNS][];
        mTarget = new int[TRAINING_PATTERNS];
        mFontNames = new String[TRAINING_PATTERNS];
        int index = 0;
        for (int i = 0; i < clusters.get(0).size(); ++i) {
            for (int j = 0; j < clusters.size(); ++j) {
                mPattern[index] = clusters.get(j).get(i);
                mTarget[index] = j;
                mFontNames[index] = (char)('A' + j) + String.valueOf(i);
                ++index;
            }
        }

        d = new double[NUMBER_OF_CLUSTERS]; // Network nodes.

        // Weight matrix to be filled with values between 0.0 and 1.0
        weights = new double[NUMBER_OF_CLUSTERS][VEC_LEN];
    }

    private static int getCategory(int x, int y) {
        if (x < -4)
            return y >= 0 ? 0 : 3;
        else if (x < 3)
            return y >= 0 ? 1 : 4;
        else
            return y >= 0 ? 2 : 5;
    }

    private static void initializeWeights(int clusterNumber, int trainingPattern[]) {
        // clusterNumber = the output node (cluster) to assign the pattern to.
        // trainingPattern = the pattern which the output node will respond to.

        // Initialize weights.
        for(int i = 0; i < VEC_LEN; i++)
            weights[clusterNumber][i] = trainingPattern[i];
    }

    private static void training() {
        int dMin;

        while(alpha > MIN_ALPHA) {
            for(int VecNum = 0; VecNum < TRAINING_PATTERNS; VecNum++) {
                // Compute input for all nodes.
                computeInput(mPattern, VecNum);

                // See which is smaller?
                dMin = minimum(d);

                // Update the weights on the winning unit.
                updateWeights(VecNum, dMin);
            }

            // Reduce the learning rate.
            alpha = DECAY_RATE * alpha;
        }
    }

    private static int getCluster(int[] inputPattern) {
        // Compute input for all nodes.
        computeInput(inputPattern);

        // See which is smaller?
        return minimum(d);
    }

    private static void updateWeights(int vectorNumber, int dMin) {
        for(int i = 0; i < VEC_LEN; i++) {
            // Update the winner.
            if(dMin == mTarget[vectorNumber])
                weights[dMin][i] += (alpha * (mPattern[vectorNumber][i] - weights[dMin][i]));
            else
                weights[dMin][i] -= (alpha * (mPattern[vectorNumber][i] - weights[dMin][i]));
        }
    }

    private static void computeInput(int[][] vectorArray, int vectorNumber) {
        // Overloaded function.  See computeInput below.
        clearArray(d);
        for(int i = 0; i < NUMBER_OF_CLUSTERS; i++) {
            for(int j = 0; j < VEC_LEN; j++) {
                d[i] += Math.pow((weights[i][j] - vectorArray[vectorNumber][j]), 2);
            }
        }
    }

    private static void computeInput(int[] vectorArray) {
        // Overloaded function.  See computeInput above.
        clearArray(d);
        for(int i = 0; i < NUMBER_OF_CLUSTERS; i++) {
            for(int j = 0; j < VEC_LEN; j++) {
                d[i] += Math.pow((weights[i][j] - vectorArray[j]), 2);
            }
        }
    }

    private static void clearArray(double[] anArray) {
        for(int i = 0; i < NUMBER_OF_CLUSTERS; i++) {
            anArray[i] = 0.0;
        }
    }

    private static int minimum(double[] nodeArray) {
        int winner = 0;
        boolean foundNewWinner;
        boolean done = false;

        while(!done) {
            foundNewWinner = false;
            for(int i = 0; i < NUMBER_OF_CLUSTERS; i++) {
                if(i != winner) {             // Avoid self-comparison.
                    if(nodeArray[i] < nodeArray[winner]) {
                        winner = i;
                        foundNewWinner = true;
                    }
                }
            }

            if(!foundNewWinner)
                done = true;
        }
        return winner;
    }

    public static void main(String[] args) {
        initialize();

        for(int i = 0; i < NUMBER_OF_CLUSTERS; i++) {
            initializeWeights(i, mPattern[i]);
            System.out.printf("Weights for cluster %d initialized to pattern %s\n", i, mFontNames[i]);
        }

        System.out.println("\nInitial weight matrix: ");
        for (double[] row : weights) {
            for (double cell : row) System.out.print(cell + " ");
            System.out.println();
        }

        System.out.println("\nStarting training phase!\n");
        training();

        // Display results
        for(int i = 0; i < TRAINING_PATTERNS; i++) {
            int actualCluster = getCluster(mPattern[i]);
            System.out.printf("Pattern %s belongs to cluster %d.\n", mFontNames[i], actualCluster);
        }

        System.out.printf("\nStarting testing phase!\n");
        int result = 0;
        int numTests = 1000;
        Random rng = new Random(10122017);
        for (int i = 0; i < numTests; ++i) {
            int x = rng.nextInt(21) - 10;
            int y = rng.nextInt(21) - 10;
            int expectedCategory = getCategory(x, y);
            int actualCategory = getCluster(new int[]{x, y});
            System.out.printf("Point: (%d, %d), Expected: %d, Actual: %d\n", x, y, expectedCategory, actualCategory);
            if (expectedCategory == actualCategory)
                ++result;
        }
        System.out.printf("Number of correct categorizations out of %d tests: %d\n", numTests, result);
    }
}