package edu.berkeley.nlp.assignments.parsing.student;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.berkeley.nlp.assignments.parsing.*;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Indexer;

public class GenerativeParser implements Parser
{
    SimpleLexicon lexicon;
    double[][][] scoreChart;
    Indexer<String> labelIndexer;
    BackPointer[][][] backPointer;
    short[][][] unaryPointer;
    List<Integer> binaryLeftChildren;
    List<Integer> binaryRightChildren;
    List<Integer> unarychild;
    Grammar grammar;
    UnaryClosure uc;
    int totalNonTerminals;
    short ntree;
    short total_sentences;
    long nanos;
    long current;
    short unaries;
    short binaries;

    public GenerativeParser(List<Tree<String>> trainTrees)
    {
        System.out.print("Annotating / binarizing training trees ... ");
        System.out.print(trainTrees.size());
        // trying to find the best maps based on reduced training size
//        if (trainTrees.size() > 4000){ trainTrees = trainTrees.subList(4000, trainTrees.size()-1);}
        List<Tree<String>> annotatedTrainTrees = annotateTrees(trainTrees);
        System.out.println("done.");
        System.out.print("Building grammar ... ");
        grammar = Grammar.generativeGrammarFromTrees(annotatedTrainTrees);
        System.out.println("done. (" + grammar.getLabelIndexer().size() + " states)");
        lexicon = new SimpleLexicon(annotatedTrainTrees);
        labelIndexer = grammar.getLabelIndexer();
        totalNonTerminals = labelIndexer.size();
        uc = new UnaryClosure(labelIndexer, grammar.getUnaryRules());
        unaries = 0;
        binaries = 0;
        System.out.println(uc);
        Set<Integer> unarychildren = new HashSet<Integer>();
//        4.8, 4.5
        // putting a threshold of probability for unary and binary grammar rules
        for(UnaryRule rule : grammar.getUnaryRules())
        {
            if(rule.getScore() > -4.8) {
                unarychildren.add(rule.getChild());
                unaries += 1;
            }
        }
        System.out.println(unaries);
        // 40 test //7737 //8577
        unarychild = new ArrayList<Integer>(unarychildren);
        ntree = 1;
        Set<Integer> tempSet = new HashSet<Integer>();
        for(BinaryRule rule : grammar.getBinaryRules())
        {
            if(rule.getScore() > -4.5) {
                tempSet.add(rule.getLeftChild());
                binaries += 1;
            }
        }
        System.out.println(binaries);
        // 40 test //11639 //15523
        binaryLeftChildren = new ArrayList<Integer>(tempSet);
//        for(BinaryRule rule : grammar.getBinaryRules())
//        {
//            tempSet.add(rule.getRightChild());
//        }
//        binaryRightChildren = new ArrayList<Integer>(tempSet);
        nanos = System.nanoTime();
        current = System.nanoTime();
        System.out.println("Time taken........................................................................................" +
                "............." + (double) (current - nanos)/(1e9)+ "\n");
        nanos = current;
    }

    // Stores unaryChild for unary rules and binaryLeftChild, binaryRightChild, splitIndex for binary rules
    public static class BackPointer {
        public int binaryLeftChild = -1;
        public int binaryRightChild = -1;
        public int splitIndex = -1;
    }

    // Returns the most probable parse tree for a given sentence
    public Tree<String> getBestParse(List<String> sentence)
    {
        fillChart(sentence);
        int n = sentence.size();
        int rootIndex = labelIndexer.addAndGetIndex("ROOT");
        Tree<String> bestParseTree;

        if(Double.isInfinite(scoreChart[0][n][rootIndex]))
            bestParseTree = new Tree<String>("ROOT", Collections.singletonList(new Tree<String>("JUNK")));
        else
            bestParseTree = buildTree(true, sentence, 0, n, rootIndex);
        System.out.println("tree" + ".............................................................................................." +
                "......" + ntree + "\n");
        current = System.nanoTime();
        System.out.println("Time taken........................................................................................" +
                "............." + (double) (current - nanos)/(1e9)+ "\n");
        nanos = current;
        ntree += 1;
        total_sentences +=1;
//        return TreeAnnotations.unAnnotateTree(bestParseTree);
        return Binarize.unAnnotateTree(bestParseTree);

    }

    private List<Tree<String>> annotateTrees(List<Tree<String>> trees)
    {
        List<Tree<String>> annotatedTrees = new ArrayList<Tree<String>>();
        for (Tree<String> tree : trees)
        {
            annotatedTrees.add(Binarize.annotateTree(tree));
//            annotatedTrees.add(TreeAnnotations.annotateTreeLosslessBinarization(tree));

        }
        return annotatedTrees;
    }

    private void fillChart(List<String> sentence) {
        System.out.println("Making chart.");
        // handle reflexive unaries
        int words = sentence.size();
        scoreChart = new double[words + 1][words + 1][totalNonTerminals];
        unaryPointer = new short[words + 1][words + 1][totalNonTerminals];
        backPointer = new BackPointer[words + 1][words + 1][totalNonTerminals];

        for (int i = 0; i <= words; i++) {
            for (int j = 0; j <= words; j++) {
                for (int k = 0; k < totalNonTerminals; k++) {
                    scoreChart[i][j][k] = Double.NEGATIVE_INFINITY;
                    unaryPointer[i][j][k] = -1;
                }
            }
        }
        for (int i = 0; i < words; i++) {
            for (int k = 0; k < totalNonTerminals; k++) {
                String nonTerminal = labelIndexer.get(k);
                double score = lexicon.scoreTagging(sentence.get(i), nonTerminal);
                if (!(Double.isNaN(score)) && !(Double.isInfinite(score))) {
                    scoreChart[i][i + 1][k] = score;
                }
            }
            for(int childNonTerminal: unarychild){
                for(UnaryRule rule: uc.getClosedUnaryRulesByChild(childNonTerminal)){
                    double ruleScore = rule.getScore();
                    if (!Double.isInfinite(ruleScore) && !(Double.isNaN(ruleScore))) {
                        int parentNonTerminal = rule.parent;
                        if(parentNonTerminal == childNonTerminal)
                            continue;
                        double rightNonTerminalScore = scoreChart[i][i + 1][childNonTerminal];
                        if (!Double.isInfinite(rightNonTerminalScore) && !(Double.isNaN(rightNonTerminalScore))) {
                            double current_prob = ruleScore + rightNonTerminalScore;
                            if (current_prob > scoreChart[i][i + 1][parentNonTerminal]) {
                                scoreChart[i][i + 1][parentNonTerminal] = current_prob;
                                unaryPointer[i][i + 1][parentNonTerminal] = (short) childNonTerminal;
                            }
                        }
                    }
                }
            }
        }
        for (int span = 2; span <= words; span++) {
            for (int start = 0; start <= (words - span); start++) {
                int end = start + span;
                for (int split = start + 1; split < end; split++) {
//                    for (int right : binaryRightChildren){
//                        double right_prob = scoreChart[split][end][right];
//                        if(!Double.isInfinite(right_prob) && !Double.isNaN(right_prob))
//                        {
//                            for(BinaryRule rule : grammar.getBinaryRulesByRightChild(right)) {
//                                double rule_prob = rule.getScore();
//                                if (!Double.isInfinite(rule_prob) && !Double.isNaN(rule_prob)) {
//                                    int parentNonTerminal = rule.getParent();
//                                    int left = rule.getLeftChild();
//                                    double current_prob;
//                                    double left_prob = scoreChart[start][split][left];
//                                    if (!Double.isInfinite(left_prob) && !Double.isNaN(left_prob)) {
//                                        current_prob = rule_prob + left_prob + right_prob;
//                                        if (current_prob > scoreChart[start][end][parentNonTerminal]) {
//                                            scoreChart[start][end][parentNonTerminal] = current_prob;
//                                            backPointer[start][end][parentNonTerminal] = new BackPointer();
//                                            backPointer[start][end][parentNonTerminal].binaryLeftChild = left;
//                                            backPointer[start][end][parentNonTerminal].binaryRightChild = right;
//                                            backPointer[start][end][parentNonTerminal].splitIndex = split;
//                                        }
//                                    }
//                                }
//                            }
//                        }
//                    }
                    for (int left : binaryLeftChildren){
                        double left_prob = scoreChart[start][split][left];
                        if(!Double.isInfinite(left_prob) && !Double.isNaN(scoreChart[start][split][left]))
                        {
                            for(BinaryRule rule : grammar.getBinaryRulesByLeftChild(left)) {
                                double rule_prob = rule.getScore();
                                if (!Double.isInfinite(rule_prob) && !Double.isNaN(rule_prob)) {
                                    int parentNonTerminal = rule.getParent();
                                    int right = rule.getRightChild();
                                    double current_prob;
                                    double right_prob = scoreChart[split][end][right];
                                    if (!Double.isInfinite(right_prob) && !Double.isNaN(right_prob)) {
                                        current_prob = rule_prob + left_prob + right_prob;
                                        if (current_prob > scoreChart[start][end][parentNonTerminal]) {
                                            scoreChart[start][end][parentNonTerminal] = current_prob;
                                            backPointer[start][end][parentNonTerminal] = new BackPointer();
                                            backPointer[start][end][parentNonTerminal].binaryLeftChild = left;
                                            backPointer[start][end][parentNonTerminal].binaryRightChild = right;
                                            backPointer[start][end][parentNonTerminal].splitIndex = split;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for(int childNonTerminal: unarychild){
                    for(UnaryRule rule: uc.getClosedUnaryRulesByChild(childNonTerminal)){
                        double ruleScore = rule.getScore();
                        if (!Double.isInfinite(ruleScore) && !(Double.isNaN(ruleScore))) {
                            int parentNonTerminal = rule.parent;
                            if(parentNonTerminal == childNonTerminal)
                                continue;
                            double rightNonTerminalScore = scoreChart[start][end][childNonTerminal];
                            if (!Double.isInfinite(rightNonTerminalScore) && !(Double.isNaN(rightNonTerminalScore))) {
                                double current_prob = ruleScore + rightNonTerminalScore;
                                if (current_prob > scoreChart[start][end][parentNonTerminal]) {
                                    scoreChart[start][end][parentNonTerminal] = current_prob;
                                    unaryPointer[start][end][parentNonTerminal] = (short) childNonTerminal;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private Tree<String> buildTree(boolean isUnaryRule, List<String> sentence, int start, int end, int parentIndex){
//        System.out.println("Building.");
        Tree<String> currentTree;
        String terminal_word = sentence.get(start);
        if(end-start==1){
//            BackPointer bp = backPointer[start][end][parentIndex];
            if(unaryPointer[start][end][parentIndex] == -1) {
                //no unary rule present now
                Tree<String> child = new Tree<String>(terminal_word);
                currentTree = new Tree<String>(labelIndexer.get(parentIndex), Collections.singletonList(child));
                return currentTree;
            }
            //handle non-terminal unaries
            int childIndex = unaryPointer[start][end][parentIndex];
            UnaryRule ur = new UnaryRule(parentIndex, childIndex);
            List<Integer> cp = uc.getPath(ur);
            if(cp != null && cp.size()>2){
                int unaryClosureLength = cp.size();
                Tree<String> child = new Tree<String>(terminal_word);
                List<Tree<String>> prevTree = new ArrayList<Tree<String>>();
                prevTree.add(child);
                Tree<String> lastUnaryNodeTree = new Tree<String>(labelIndexer.get(cp.get(unaryClosureLength-1)), prevTree);
                currentTree = expandUnaryClosureRule(cp, lastUnaryNodeTree);
            }
            else {
                Tree<String> child = new Tree<String>(terminal_word);
                List<Tree<String>> prevTree = new ArrayList<Tree<String>>();
                prevTree.add(child);
                Tree<String> lastUnaryNode = new Tree<String>(labelIndexer.get(childIndex), prevTree);
                List<Tree<String>> lastUnaryNodeTree = new ArrayList<Tree<String>>();
                lastUnaryNodeTree.add(lastUnaryNode);
                currentTree = new Tree<String>(labelIndexer.get(parentIndex), lastUnaryNodeTree);
            }
            return currentTree;
        }
        if(isUnaryRule){
            //unary back pointing
            int childIndex = unaryPointer[start][end][parentIndex];
            if(childIndex == -1){
                currentTree = buildTree(false, sentence, start, end, parentIndex);
            }
            else{
                UnaryRule ur = new UnaryRule(parentIndex, childIndex);
                List<Integer> cp = uc.getPath(ur);
                if(cp != null && cp.size()>2){
                    int unaryClosureLength = cp.size();
                    Tree<String> lastUnaryNodeTree = buildTree(false, sentence, start, end,cp.get(unaryClosureLength - 1));
                    currentTree = expandUnaryClosureRule(cp, lastUnaryNodeTree);
                }
                else{
                    List<Tree<String>> child = new ArrayList<Tree<String>>();
                    child.add(buildTree(false, sentence, start, end, childIndex));
                    currentTree = new Tree<String>(labelIndexer.get(parentIndex), child);
                }
            }
            return currentTree;
        }
        else{
            //binary back pointing
            if(backPointer[start][end][parentIndex]== null || backPointer[start][end][parentIndex].splitIndex == -1){

                currentTree = buildTree(true, sentence, start, end, parentIndex);
            }
            else{
                int splitIndex = backPointer[start][end][parentIndex].splitIndex;
                int binaryLeftChild = backPointer[start][end][parentIndex].binaryLeftChild;
                int binaryRightChild = backPointer[start][end][parentIndex].binaryRightChild;
                Tree<String> binaryLeftChildTree = buildTree(true, sentence, start, splitIndex, binaryLeftChild);
                Tree<String> binaryRightChildTree = buildTree(true, sentence, splitIndex, end, binaryRightChild);
                List<Tree<String>> children = new ArrayList<Tree<String>>();
                children.add(binaryLeftChildTree);
                children.add(binaryRightChildTree);
                currentTree = new Tree<String>(labelIndexer.get(parentIndex), children);

            }
            return currentTree;
        }
    }

    private Tree<String> expandUnaryClosureRule(List<Integer> unaryClosurePath, Tree<String> lastUnaryNodeTree)
    {
        Tree<String> currentChild = lastUnaryNodeTree;
        for(int i = (unaryClosurePath.size() - 2); i>=0; i--)
        {
            List<Tree<String>> child = new ArrayList<Tree<String>>();
            child.add(currentChild);
            currentChild = new Tree<String>(labelIndexer.get(unaryClosurePath.get(i)), child);
        }
        return currentChild;
    }
}