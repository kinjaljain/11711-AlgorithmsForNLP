package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.ParsingReranker;
import edu.berkeley.nlp.assignments.rerank.student.MyFeatureExtractor;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.parser.EnglishPennTreebankParseEvaluator;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.IntCounter;
import edu.berkeley.nlp.util.Pair;

import java.lang.reflect.Array;
import java.util.*;

// voted perceptron
public class AveragePerceptronReranker implements ParsingReranker {
    Indexer<String> featureIndexer = new Indexer<String>();
    ArrayList<PerceptronDatum> trainData = new ArrayList<PerceptronDatum>();
    MyFeatureExtractor featureExtractor = new MyFeatureExtractor();
    IntCounter featureWeights = new IntCounter();
    int epochs = 10;

    public AveragePerceptronReranker(Iterable<Pair<KbestList,Tree<String>>> kbestListsAndGoldTrees){
        System.out.println("Extracting features");
        HashSet<String> labelsToIgnore = new HashSet<String>(Collections.singleton("ROOT"));
        HashSet<String> punctuationTags = new HashSet<String>(Arrays.asList(".", "?", "!", "(", ")", ",", ";",
                "{", "}", "[", "]", "\\", "\"", "'", ":", "/", "|", "~", "`", "``"));
        EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String> evaluator = new
                EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String>(labelsToIgnore, punctuationTags);
        int counter = 0;
        for(Pair<KbestList,Tree<String>> kbestListAndGoldTree : kbestListsAndGoldTrees)
        {
            KbestList kBestList = kbestListAndGoldTree.getFirst();
            Tree<String> goldTree = kbestListAndGoldTree.getSecond();
            List<Tree<String>> kBestTrees = kBestList.getKbestTrees();
            int k = kBestTrees.size();
            int[][] featureList = new int[k][];
            boolean goldTreeIsPresent = false;
            int bestTreeIndex = -1;
            double bestF1 = Double.NEGATIVE_INFINITY;
            int[] goldList;
            double[] kBestF1 = new double[k];
//            double[] scores = kBestList.getScores();
//            double maxScore = Double.NEGATIVE_INFINITY;
//            double minScore = Double.POSITIVE_INFINITY;

            //fetch minScore and maxScore values
//            for (int i = 0; i < k; i++) {
//                if(scores[i]> maxScore){
//                    maxScore = scores[i];
//                }
//                if(scores[i]< minScore){
//                    minScore = scores[i];
//                }
//            }
            //feature vectors for the GoldTree and KBestTrees
            int scoreBin;
            for (int i = 0; i < k; i++) {
//                scoreBin = getScoreBin(maxScore, minScore, scores[i]);
                int[] curFeatureList = featureExtractor.extractFeatures(kBestList, i,
                        featureIndexer, true);
                featureList[i] = curFeatureList;
                kBestF1[i] = evaluator.evaluateF1(kBestList.getKbestTrees().get(i), goldTree);
                if (kBestF1[i] == 1){
                    counter += 1;
                    goldTreeIsPresent = true;
                }
                if (kBestF1[i] > bestF1){
                    bestTreeIndex = i;
                    bestF1 = kBestF1[i];
                }
            }
            //if goldTree is not present in the kBestList, use the most closest one for training
//            scoreBin = getScoreBin(maxScore, minScore, scores[bestTreeIndex]);
            goldList = featureExtractor.extractFeatures(kBestList, bestTreeIndex,
                    featureIndexer,true);
            IntCounter goldFeatures = createFeatureCounter(goldList);

            PerceptronDatum data = new PerceptronDatum(bestTreeIndex, goldFeatures, featureList);
            trainData.add(data);
        }
        System.out.println("Number of gold trees present: " + counter);
        System.out.println("Feature extraction complete!");
        System.out.println("Training Data size: " + trainData.size());
        System.out.println("Number of features: " + featureIndexer.size());

        //initializing weights;
        for(int i=0; i<featureIndexer.size(); i++)
            featureWeights.put(i,0);

        System.out.println("Start Training Perceptron");

        for (int i=0; i<epochs; i++) {
            System.out.println("Epoch: " + i);
            trainPerceptron(trainData);
        }
    }

    public void trainPerceptron(ArrayList<PerceptronDatum> trainData){
        int counter = 0;
            for(PerceptronDatum data: trainData){
                IntCounter goldFeatures = data.getGoldFeatures();
                int[][] kBestFeatures = data.getKBestListfeatures();
                double maxObjective = Double.NEGATIVE_INFINITY;
                int goldTreeIndex = data.getGoldIndex();
                int bestPredictedTreeIndex = -1;
                double kObjective;

                int k = kBestFeatures.length;
                for (int i = 0; i < k; i++) {
                    kObjective = getValue(createFeatureCounter(kBestFeatures[i]));
                    if (kObjective >= maxObjective) {
                        maxObjective = kObjective;
                        bestPredictedTreeIndex = i;
                    }
                }
                if(bestPredictedTreeIndex != goldTreeIndex) {
                    counter += 1;
                    IntCounter bestPredictedTreeFeatures = createFeatureCounter(kBestFeatures[bestPredictedTreeIndex]);
                    IntCounter updateFeatureWeights = new IntCounter();
                    for(int key : goldFeatures.keySet())
                    {
                        updateFeatureWeights.put(key, goldFeatures.get(key));
                    }
                    for(int key : bestPredictedTreeFeatures.keySet())
                    {
                        updateFeatureWeights.put(key, goldFeatures.get(key) -
                                bestPredictedTreeFeatures.get(key));
                    }
                    // System.out.println("Features to be updated: " + updateFeatureWeights.size());
                    // Update weights every 2000 mistakes by averaging
                    if(counter % 2000 == 0){
                        IntCounter oldFeatureWeights = featureWeights;
                        for (int key : updateFeatureWeights.keySet()) {
                            featureWeights.put(key, (updateFeatureWeights.get(key) + oldFeatureWeights.get(key))/2);
                        }
                    }
                    else{
                        for (int key : updateFeatureWeights.keySet()) {
                            featureWeights.incrementCount(key, updateFeatureWeights.get(key));
                        }
                    }
                }
            }
//        }
        System.out.println(counter);
    }

    public double getValue(IntCounter bestPredictedFeatures){
//        return featureWeights.dotProduct(bestPredictedFeatures);
        double val=0;
        for(int key: bestPredictedFeatures.keySet()){
            val += featureWeights.get(key)*bestPredictedFeatures.get(key);
        }
        return val;
    }

    public Tree<String> getBestParse(List<String> sentence, KbestList kbestList) {
        int k = kbestList.getKbestTrees().size();
        int bestTreeIndex = -1;
        double maxObjective=Double.NEGATIVE_INFINITY;
        double kObjective;
//        double[] scores = kbestList.getScores();
//        double maxScore = Double.NEGATIVE_INFINITY;
//        double minScore = Double.POSITIVE_INFINITY;
//        int scoreBin;
//
//        for (int i = 0; i < k; i++) {
//            if(scores[i]> maxScore){
//                maxScore = scores[i];
//            }
//            if(scores[i]< minScore){
//                minScore = scores[i];
//            }
//        }

        for (int i = 0; i < k; i++) {
//            scoreBin = getScoreBin(maxScore, minScore, scores[i]);
            int[] featsArr = featureExtractor.extractFeatures(kbestList, i,
                    featureIndexer, false);
            kObjective = getValue(createFeatureCounter(featsArr));
            if (kObjective>maxObjective){
                maxObjective=kObjective;
                bestTreeIndex = i;
            }
        }
        return kbestList.getKbestTrees().get(bestTreeIndex);
    }

    public int getScoreBin(double maxS,double minS, double score){
        double diff=(maxS-minS)/5;
        int bin;
        if ((maxS-diff)<=score) bin=1;
        else if((maxS-2*diff)<=score) bin=2;
        else if((maxS-3*diff)<=score) bin=3;
        else if((maxS-4*diff)<=score) bin=4;
        else bin=5;
        return bin;
    }

    public IntCounter createFeatureCounter(int[] featuresList){
        IntCounter featureCounter = new IntCounter();
        for(int feature: featuresList){
            featureCounter.incrementCount(feature,1);
        }
//        double val;
//        IntCounter newFeatureCounter = new IntCounter();
//        for(int feature: featuresList){
//            val = featureCounter.getCount(feature);
//            if(val > 5)
//                newFeatureCounter.put(feature, val);
//        }
//        return newFeatureCounter;
        return featureCounter;
    }
}