package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.SurfaceHeadFinder;
import edu.berkeley.nlp.ling.AnchoredTree;
import edu.berkeley.nlp.ling.Constituent;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Indexer;

import java.util.*;

public class MyFeatureExtractor2{
    public int[] extractFeaturesWithScore(KbestList kbestList, int idx, int scoreBin, Indexer<String> featureIndexer,
                                 boolean addFeaturesToIndexer) {
        List<Integer> features = new ArrayList<Integer>();
        addFeature("ScoreBin=" + scoreBin, features, featureIndexer, addFeaturesToIndexer);
        extractFeatures(kbestList, idx, features, featureIndexer, addFeaturesToIndexer);
        int[] featsArr = new int[features.size()];
        for (int i = 0; i < features.size(); i++) {
            featsArr[i] = features.get(i);
        }
        return featsArr;
    }
    public void extractFeatures(KbestList kbestList, int idx, List<Integer> features, Indexer<String> featureIndexer,
                                 boolean addFeaturesToIndexer) {
//        System.out.println("Extracting features for index: " + idx);
        Tree<String> tree = kbestList.getKbestTrees().get(idx);
        AnchoredTree<String> anchoredTree = AnchoredTree.fromTree(tree);
        Collection<Constituent<String>> constituents = tree.toConstituentList();
        List<String> poss = tree.getPreTerminalYield();
        List<String> words = tree.getYield();
        SurfaceHeadFinder shf = new SurfaceHeadFinder();

        // position feature
        addFeature("Posn=" + idx, features, featureIndexer, addFeaturesToIndexer);
//        System.out.println("Position added");

        addAllAncestors(tree, features, featureIndexer, addFeaturesToIndexer);

//                        right branching feature
        String[] punctuations = {".", "?", "!", "(", ")", ",", ";", "{", "}", "[", "]", "\\", "\"",
                "'", ":", "/", "|", "~", "`"};
        Set<String> possiblePunctuations = Collections.unmodifiableSet(new HashSet<String>(Arrays.asList(punctuations)));
        int non_punc_index = words.size() - 1;

        for (; non_punc_index >= 0; non_punc_index--) {
            if (!possiblePunctuations.contains(words.get(non_punc_index)))
                break;
        }
        int rightBranchLength = 0;
        AnchoredTree<String> curr_tree = anchoredTree;
        Set<AnchoredTree<String>> childTrees = new HashSet<>();
        List<AnchoredTree<String>> children;
        while (!curr_tree.isLeaf() && curr_tree.getChildren().size() > 0) {
            if (childTrees.contains(curr_tree)) {
                break;
            }
            childTrees.add(curr_tree);
            children = curr_tree.getChildren();
            int num_child = children.size();
            for (int k = num_child - 1; k >= 0; k--) {
                if (children.get(k).isLeaf()) {
                    continue;
                }
                if (children.get(k).getStartIdx() <= non_punc_index && children.get(k).getEndIdx() > non_punc_index) {
                    curr_tree = children.get(num_child - 1);
                    rightBranchLength += 1;
                    break;
                }
            }
        }
        addFeature("RightBranchingNT=" + getBin(rightBranchLength), features, featureIndexer, addFeaturesToIndexer);
//                System.out.println("right branching added");

        for (AnchoredTree<String> subtree : anchoredTree.toSubTreeList()) {
            if (!subtree.isPreTerminal() && !subtree.isLeaf()) {
                int startIndex = subtree.getStartIdx();
                int endIndex = subtree.getEndIdx();
                int numChildren = subtree.getChildren().size();
                List<String> subtree_poss = poss.subList(startIndex, endIndex);
                String parentLabel = subtree.getLabel();
                String rule = "Rule=" + parentLabel + " ->";
                for (AnchoredTree<String> child : subtree.getChildren()) {
                    rule += " " + child.getLabel();
                }
                // rule feature
//                addFeature(rule, features, featureIndexer, addFeaturesToIndexer);
//                System.out.println("Rule added");

                // word ancestors 2, 3
//                printPaths(subtree, features, featureIndexer, addFeaturesToIndexer);

                // span length feature
                int spanLength = subtree.getSpanLength();
//                addFeature("Rule=" + rule + "_SpanLen=" + getBin(spanLength), features,
//                        featureIndexer, addFeaturesToIndexer);
//                System.out.println("Rule span length added");

                // span left context & first word & span right context & last word
                if (startIndex > 0 && endIndex < words.size() -1) {
//                    addFeature("Rule=" + rule + "_LeftContext=" + words.get(startIndex - 1), features,
//                            featureIndexer, addFeaturesToIndexer);
//                    addFeature("Rule=" + rule + "_FirstWord=" + words.get(startIndex), features,
//                            featureIndexer, addFeaturesToIndexer);
//                    addFeature("Rule=" + rule + "_RightContext=" + words.get(endIndex), features,
//                            featureIndexer, addFeaturesToIndexer);
//                    addFeature("Rule=" + rule + "_LastWord=" + words.get(endIndex - 1), features,
//                            featureIndexer, addFeaturesToIndexer);
//                    addFeature("Rule=" + rule + "_FirstWord=" + words.get(startIndex) +
//                            "_LastWord=" + words.get(endIndex - 1) + "_SpanLen=" +
//                            getBin(spanLength), features, featureIndexer, addFeaturesToIndexer);
                    addFeature("Rule=" + rule + "_FirstWord=" + words.get(startIndex) +
                                    "_LastWord=" + words.get(endIndex - 1) , features, featureIndexer, addFeaturesToIndexer);
//                    addFeature("Rule=" + rule + "_FirstWord=" + words.get(startIndex) +
//                            "_LastWord=" + words.get(endIndex - 1) + "_SpanLen=" +
//                            getBin(spanLength), features, featureIndexer, addFeaturesToIndexer);
                    addFeature("Rule=" + rule + "_LeftContext=" + words.get(startIndex - 1) +
                            "_RightContext=" + words.get(endIndex), features, featureIndexer, addFeaturesToIndexer);
//                    addFeature("Rule=" + rule + "_FirstWord=" + words.get(startIndex) +
//                                    "_LastWord=" + words.get(endIndex - 1) + "_SpanLen=" +
//                                    getBin(spanLength) + "_LeftContext=" + words.get(startIndex - 1) +
//                            "_RightContext=" + words.get(endIndex), features, featureIndexer, addFeaturesToIndexer);
                }
//                System.out.println("left Context, right context, left word, right word added");

                // span shape feature
                String spanShape = parentLabel + "->";
                // experiment with entire rule instead of just parent label
                for (int i = startIndex; i < endIndex; i++) {
                    char firstSymbol = words.get(i).charAt(0);
                    if (firstSymbol >= 'A' && firstSymbol <= 'Z')
                        spanShape += "X";
                    else if (firstSymbol >= 'a' && firstSymbol <= 'z')
                        spanShape += "x";
                    else if (firstSymbol >= '0' && firstSymbol <= '9')
                        spanShape += "N";
                    else
                        spanShape += firstSymbol;
                }
                addFeature("Rule=" + rule + "_SpanShape=" + spanShape, features, featureIndexer, addFeaturesToIndexer);
////                System.out.println("span shape added");

                children = subtree.getChildren();
//                // split point feature
                if (numChildren == 2) {
                    String splitRule = "SplitRule=" + parentLabel + " ->";
                    AnchoredTree<String> leftChild = children.get(0);
                    AnchoredTree<String> rightChild = children.get(1);
                    String splitPoint = words.get(leftChild.getEndIdx());
                    splitRule = splitRule + "(" + leftChild.getLabel() + "..." + splitPoint + ")" +
                            rightChild.getLabel() + ")";
//                    splitRule = parentLabel + "(" + leftChild.getLabel() + "..." + splitPoint + ")" +
//                            rightChild.getLabel() + ")";
                    addFeature(splitRule, features, featureIndexer, addFeaturesToIndexer);
                }
////                    System.out.println("split point added");

                // heavyness feature
                int distanceToSentenceEnd = words.size() - endIndex;
                addFeature("Heavyness=" + parentLabel + "_" + spanLength + "_" + distanceToSentenceEnd,
                        features, featureIndexer, addFeaturesToIndexer);
//                System.out.println("heavyness added");

                // trigram rule added
                for (int i = 0; i < (children.size() - 2); i++) {
                    String trigram = parentLabel + "->";
                    for (int j = i; j <= i + 2; j++) {
                        trigram += ".." + children.get(j).getLabel();
                    }
                    addFeature(trigram, features, featureIndexer, addFeaturesToIndexer);
                }
//                System.out.println("ngram rule added");

////                //head feature
                addFeature("Rule=" + rule + "Head=" + shf.findHead(parentLabel, subtree_poss), features, featureIndexer, addFeaturesToIndexer);
////
////                //neighbor feature
                addFeature("Rule=" + rule + "_" + parentLabel + "_" + spanLength + "_" + subtree_poss,
                        features, featureIndexer, addFeaturesToIndexer);
            }
        }
    }

    private String getBin(int length){
        if (length <= 5)
            return Integer.toString(length);
        else if (length <=10)
            return "<=10";
        else if (length <= 20)
            return "<=20";
        else
            return ">20";
    }

    private void addAncestorRules(Tree<String> subtree, String[] path, int pathLen, List<Integer> features,
                         Indexer<String> featureIndexer, boolean addNew)
    {
        if (subtree == null)
            return;

        path[pathLen] = subtree.getLabel();
        pathLen++;

        if (subtree.isLeaf()){
            if (pathLen > 4){
                String ancestors = "Ancestor=4_" + path[pathLen-1] + "->" + path[pathLen-2] + "_" + path[pathLen-3] +
                        "_" + path[pathLen-4] + "_" + path[pathLen-5];
                addFeature(ancestors, features, featureIndexer, addNew);
            }
        }

        else
        {
            // try all subtrees
            List<Tree<String>> children = subtree.getChildren();
            int num_children = children.size();
            for(int i=0; i< num_children; i++) {
                addAncestorRules(subtree.getChildren().get(i), path, pathLen, features, featureIndexer, addNew);
            }
        }
    }

    private void addAllAncestors(Tree<String> subtree, List<Integer> features, Indexer<String> featureIndexer, boolean addNew)
    {
        String[] path = new String[1000];
        addAncestorRules(subtree, path, 0, features, featureIndexer, addNew);
    }

    private void addFeature(String feat, List<Integer> features, Indexer<String> featureIndexer, boolean addNew) {
        if (addNew || featureIndexer.contains(feat)) {
            features.add(featureIndexer.addAndGetIndex(feat));
        }
    }
}
