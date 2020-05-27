package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.ParsingReranker;
import edu.berkeley.nlp.assignments.rerank.PrimalSubgradientSVMLearner;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.parser.EnglishPennTreebankParseEvaluator;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.IntCounter;
import edu.berkeley.nlp.util.Pair;

import java.util.*;

public class PrimalSVMRerankerBinaryLoss implements ParsingReranker{
    // primal svm
    Indexer<String> featureIndexer = new Indexer<String>();
    ArrayList<PrimalSVMDatum> trainData = new ArrayList<PrimalSVMDatum>();
    MyFeatureExtractor2 featureExtractor = new MyFeatureExtractor2();
    IntCounter featureWeights = new IntCounter();
    HashSet<String> labelsToIgnore = new HashSet<String>(Collections.singleton("ROOT"));
    HashSet<String> punctuationTags = new HashSet<String>(Arrays.asList(".", "?", "!", "(", ")", ",", ";",
            "{", "}", "[", "]", "\\", "\"", "'", ":", "/", "|", "~", "`", "``"));
    EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String> evaluator = new
            EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String>(labelsToIgnore, punctuationTags);
    double stepSize  = 1e-3;
    double regConstant = 1e-4;
    int batchSize = 100;
    int epochs = 20;

    public PrimalSVMRerankerBinaryLoss(Iterable<Pair<KbestList, Tree<String>>> kbestListsAndGoldTrees) {
        for(Pair<KbestList,Tree<String>> kBestListAndGoldTree : kbestListsAndGoldTrees) {
            KbestList kBestList = kBestListAndGoldTree.getFirst();
            Tree<String> goldTree = kBestListAndGoldTree.getSecond();
            List<Tree<String>> kBestTrees = kBestList.getKbestTrees();
            int k = kBestTrees.size();
            double[] kBestListLoss = new double[k];
            int[][] featureList = new int[k][];
            boolean goldTreeIsPresent = false;
            int bestTreeIndex = -1;
            double bestTreeLoss = Double.POSITIVE_INFINITY;
            int[] goldList;
            double[] scores = kBestList.getScores();
            double maxScore = Double.NEGATIVE_INFINITY;
            double minScore = Double.POSITIVE_INFINITY;
            int scoreBin;

            for (int i = 0; i < k; i++) {
                if(scores[i]> maxScore){
                    maxScore = scores[i];
                }
                if(scores[i]< minScore){
                    minScore = scores[i];
                }
            }
//        MyFeatureExtractor2 featureExtractor = new MyFeatureExtractor2();
            for(int i=0; i<k; i++){
                scoreBin = getScoreBin(maxScore, minScore, scores[i]);
                int[] curFeatureList = featureExtractor.extractFeaturesWithScore(kBestList, i, scoreBin,
                        featureIndexer, true);
                featureList[i] = curFeatureList;
                kBestListLoss[i] = 1 - evaluator.evaluateF1(kBestList.getKbestTrees().get(i), goldTree);
                if(kBestListLoss[i] == 0){
                    goldTreeIsPresent = true;
                }
                if(kBestListLoss[i] < bestTreeLoss){
                    bestTreeIndex = i;
                    bestTreeLoss = kBestListLoss[i];
                }
            }
            //if goldTree is not present in the kBestList, use the most closest one for training
//            if(!goldTreeIsPresent){
            scoreBin = getScoreBin(maxScore, minScore, scores[bestTreeIndex]);
            goldList = featureExtractor.extractFeaturesWithScore(kBestList, bestTreeIndex, scoreBin,
                    featureIndexer,true);
//            }
//            else {
//                goldList = featureExtractor.extractGoldTreeFeatures(goldTree, featureIndexer,
//                        true);
//            }
            IntCounter goldFeatures = createFeatureCounter(goldList);
            for (int i = 0; i < k; i++) {
                if (i==bestTreeIndex){
                    kBestListLoss[i] = 0;
                }
                else{
                    kBestListLoss[i] = 1;
                }
            }

            PrimalSVMDatum data = new PrimalSVMDatum(goldFeatures, kBestListLoss, featureList);
            trainData.add(data);
        }
        System.out.println(trainData.size());
        int numFeatures = featureIndexer.size();
        System.out.println(featureIndexer.size());

        PrimalSubgradientSVMLearner<PrimalSVMDatum> learner =
                new PrimalSubgradientSVMLearner<PrimalSVMDatum>(stepSize, regConstant,
                        numFeatures, batchSize);
        AwesomeLossAugmentedModel lm = new AwesomeLossAugmentedModel();
        featureWeights = learner.train(featureWeights, lm, trainData, epochs);
    }

    public Tree<String> getBestParse(List<String> sentence, KbestList kbestList) {
        double maxObjective = Double.NEGATIVE_INFINITY;
        int bestIndex = -1;
        double kObjective =  0;
        List<Tree<String>> kBestTrees = kbestList.getKbestTrees();
        int k = kBestTrees.size();
        double[] scores = kbestList.getScores();
        double maxScore = Double.NEGATIVE_INFINITY;
        double minScore = Double.POSITIVE_INFINITY;
        int scoreBin;

        for (int i = 0; i < k; i++) {
            if(scores[i]> maxScore){
                maxScore = scores[i];
            }
            if(scores[i]< minScore){
                minScore = scores[i];
            }
        }

        for(int i=0; i<k; i++){
            scoreBin = getScoreBin(maxScore, minScore, scores[i]);
            IntCounter kFeatures = createFeatureCounter(featureExtractor.extractFeaturesWithScore(kbestList, i,
                    scoreBin, featureIndexer, false));
            kObjective = kFeatures.dotProduct(featureWeights);
            if(kObjective > maxObjective){
                maxObjective = kObjective;
                bestIndex = i;
            }
        }
        return kBestTrees.get(bestIndex);
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
