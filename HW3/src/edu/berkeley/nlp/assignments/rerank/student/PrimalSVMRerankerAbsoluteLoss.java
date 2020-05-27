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

// primal svm
public class PrimalSVMRerankerAbsoluteLoss implements ParsingReranker {
    Indexer<String> featureIndexer = new Indexer<String>();
    ArrayList<PrimalSVMDatum> trainData = new ArrayList<PrimalSVMDatum>();
    MyFeatureExtractor featureExtractor = new MyFeatureExtractor();
    IntCounter featureWeights = new IntCounter();
    HashSet<String> labelsToIgnore = new HashSet<String>(Collections.singleton("ROOT"));
    HashSet<String> punctuationTags = new HashSet<String>(Arrays.asList(".", "?", "!", "(", ")", ",", ";",
            "{", "}", "[", "]", "\\", "\"", "'", ":", "/", "|", "~", "`", "``"));
    EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String> evaluator = new
            EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String>(labelsToIgnore, punctuationTags);
    double stepSize  = 1e-2;
    double regConstant = 1e-4;
    int batchSize = 512;
    int epochs = 30;

    public PrimalSVMRerankerAbsoluteLoss(Iterable<Pair<KbestList, Tree<String>>> kbestListsAndGoldTrees) {
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

            for (int i = 0; i < k; i++) {
                int[] curFeatureList = featureExtractor.extractFeatures(kBestList, i,
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
            goldList = featureExtractor.extractFeatures(kBestList, bestTreeIndex, featureIndexer,
                    true);
//            }
//            else {
//                goldList = featureExtractor.extractGoldTreeFeatures(goldTree, featureIndexer,
//                        true);
//            }
            IntCounter goldFeatures = createFeatureCounter(goldList);

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
        MyFeatureExtractor featureExtractor = new MyFeatureExtractor();
        List<Tree<String>> kBestTrees = kbestList.getKbestTrees();
        int k = kBestTrees.size();
        for(int i=0; i<k; i++){
            IntCounter kFeatures = createFeatureCounter(featureExtractor.extractFeatures(kbestList, i,
                    featureIndexer, false));
            kObjective = kFeatures.dotProduct(featureWeights);
            if(kObjective > maxObjective){
                maxObjective = kObjective;
                bestIndex = i;
            }
        }
        return kBestTrees.get(bestIndex);
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
