package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.LossAugmentedLinearModel;
import edu.berkeley.nlp.util.IntCounter;

public class AwesomeLossAugmentedModel implements LossAugmentedLinearModel<PrimalSVMDatum>
{
    public UpdateBundle getLossAugmentedUpdateBundle(PrimalSVMDatum datum, IntCounter weights) {
        IntCounter goldFeatures = datum.getGoldFeatures();
        double maxObjective = Double.NEGATIVE_INFINITY;
        IntCounter argmax = null;
        double lossOfGuess = 0;
        double kObjective;
        double kLoss;
        double[] kBestLoss = datum.getKBestListLoss();
        int[][] kBestFeatures = datum.getKBestListfeatures();

        for(int i = 0; i < kBestLoss.length; i++)
        {
            IntCounter kFeature = createFeatureCounterList(kBestFeatures[i]);
            kObjective = kFeature.dotProduct(weights);
            kLoss = kBestLoss[i];

            if((kObjective + kLoss) > maxObjective)
            {
                maxObjective = kObjective+kLoss;
                argmax = kFeature;
                lossOfGuess = kLoss;
            }
        }

        UpdateBundle bundle = new UpdateBundle(goldFeatures, argmax, lossOfGuess);
        return bundle;
    }

    public IntCounter createFeatureCounterList(int[] featureList)
    {
        IntCounter featureCounter = new IntCounter();
        for (int feature : featureList) {
            featureCounter.incrementCount(feature, 1);
        }
        return featureCounter;
    }
}
