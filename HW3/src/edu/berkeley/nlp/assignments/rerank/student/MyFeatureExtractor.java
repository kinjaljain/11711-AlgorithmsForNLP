package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.SurfaceHeadFinder;
import edu.berkeley.nlp.ling.AnchoredTree;
import edu.berkeley.nlp.ling.Constituent;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Indexer;

import java.util.*;

public class MyFeatureExtractor{
    public int[] extractFeatures(KbestList kbestList, int idx, Indexer<String> featureIndexer,
                                 boolean addFeaturesToIndexer) {
//        System.out.println("Extracting features for index: " + idx);
        Tree<String> tree = kbestList.getKbestTrees().get(idx);
        AnchoredTree<String> anchoredTree = AnchoredTree.fromTree(tree);
//        Collection<Constituent<String>> constituents = tree.toConstituentList();
        List<String> poss = tree.getPreTerminalYield();
        List<String> words = tree.getYield();
        SurfaceHeadFinder shf = new SurfaceHeadFinder();
        List<Integer> feats = new ArrayList<Integer>();

        // position feature
        addFeature("Posn=" + idx, feats, featureIndexer, addFeaturesToIndexer);
//        System.out.println("Position added");

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
                addFeature(rule, feats, featureIndexer, addFeaturesToIndexer);
//                System.out.println("Rule added");

                // word ancestors 2, 3
//                printPaths(subtree, feats, featureIndexer, addFeaturesToIndexer);

                // span length feature
                int spanLength = subtree.getSpanLength();
                addFeature("Rule=" + rule + "_SpanLen=" + getBin(spanLength), feats,
                        featureIndexer, addFeaturesToIndexer);
//                System.out.println("Rule span length added");

                // span left context & first word
                if (startIndex > 0) {
                    addFeature("Rule=" + rule + "_LeftContext=" + words.get(startIndex - 1), feats,
                            featureIndexer, addFeaturesToIndexer);
                    addFeature("Rule=" + rule + "_FirstWord=" + words.get(startIndex), feats,
                            featureIndexer, addFeaturesToIndexer);
                }
//                System.out.println("left Context added");

                // span right context & last word
                if (endIndex < words.size()-1) {
                    addFeature("Rule=" + rule + "_RightContext=" + words.get(endIndex), feats,
                            featureIndexer, addFeaturesToIndexer);
                    addFeature("Rule=" + rule + "_LastWord=" + words.get(endIndex + 1), feats,
                            featureIndexer, addFeaturesToIndexer);
                }
//                System.out.println("right Context added");

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
                addFeature("Rule=" + rule + "_SpanShape=" + spanShape, feats, featureIndexer, addFeaturesToIndexer);
//                System.out.println("span shape added");

                List<AnchoredTree<String>> children = subtree.getChildren();
                // split point feature
                if (numChildren == 2) {
                    String splitRule = "SplitRule=" + parentLabel + " ->";
                    AnchoredTree<String> leftChild = children.get(0);
                    AnchoredTree<String> rightChild = children.get(1);
                    String splitPoint = words.get(leftChild.getEndIdx());
                    splitRule = splitRule + "(" + leftChild.getLabel() + "..." + splitPoint + ")" +
                            rightChild.getLabel() + ")";
                    addFeature(splitRule, feats, featureIndexer, addFeaturesToIndexer);
//                    System.out.println("split point added");
                }

                // heavyness feature
                int distanceToSentenceEnd = words.size() - endIndex;
                addFeature("Heavyness=" + parentLabel + "_" + spanLength + "_" + distanceToSentenceEnd,
                        feats, featureIndexer, addFeaturesToIndexer);
//                System.out.println("heavyness added");

                // ngram rule added
                for(int i=0; i<(children.size()-2); i++)
                {
                    String ngramRule = parentLabel + "->";
                    for(int j=i; j<i+3; j++)
                    {
                        ngramRule =  ngramRule + "_" + children.get(j).getLabel();
                    }
                    addFeature(ngramRule, feats, featureIndexer, addFeaturesToIndexer);
                }
//                System.out.println("ngram rule added");


//                right branching feature
                String[] punctuations = {".", "?", "!", "(", ")", ",", ";", "{", "}", "[", "]", "\\", "\"",
                        "'", ":", "/", "|", "~", "`"};
                Set<String> possiblePunctuations = Collections.unmodifiableSet(new HashSet<String>(Arrays.asList(punctuations)));
                int non_punc_index = words.size() - 1;

                for (; non_punc_index >= 0; non_punc_index--) {
                    if (!possiblePunctuations.contains(words.get(non_punc_index)))
                        break;
                }
                if(non_punc_index<0){
                    continue;
                }
                int rightBranchLength = 0;
                AnchoredTree<String> curr_tree = subtree;
                Set<AnchoredTree<String>> childTrees = new HashSet<>();
                while (!curr_tree.isLeaf() && curr_tree.getChildren().size() > 0) {
                    if(childTrees.contains(curr_tree)){
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
                addFeature("Rule=" + rule + "RightBranchingNT=" + getBin(rightBranchLength), feats, featureIndexer, addFeaturesToIndexer);
//                System.out.println("right branching added");

//                //head feature
//                addFeature("Rule=" + rule + "Head=" + shf.findHead(parentLabel, subtree_poss), feats, featureIndexer, addFeaturesToIndexer);
//
//                //neighbor feature
//                addFeature("Rule=" + rule + "_"  + parentLabel + "_" + spanLength + "_" + subtree_poss, feats, featureIndexer, addFeaturesToIndexer);
            }
        }

        int[] featsArr = new int[feats.size()];
            for (int i = 0; i < feats.size(); i++) {
                featsArr[i] = feats.get(i);
            }
            return featsArr;
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

    void printPathsRecur(AnchoredTree<String> subtree, String[] path, int pathLen, List<Integer> feats,
                         Indexer<String> featureIndexer, boolean addNew)
    {
        if (subtree == null)
            return;

        // append this node to the path array
        path[pathLen] = subtree.getLabel();
        pathLen++;

        // it's a leaf, so print the path that led to here
        if (subtree.isLeaf()){
            if (pathLen > 3){
                String ancestors = "Ancestor=3=" + path[pathLen-1] + "->" + path[pathLen-2] + "_" + path[pathLen-3] +
                        "_" + path[pathLen-4];
                addFeature(ancestors, feats, featureIndexer, addNew);
//                ancestors = "Ancestor=2=" + path[pathLen-1] + "->" + path[pathLen-2] + "_" + path[pathLen-3];
//                addFeature(ancestors, feats, featureIndexer, addNew);
            }
//            else if (pathLen > 2){
//                String ancestors = "Ancestor=2=" + path[pathLen-1] + "->" + path[pathLen-2] + "_" + path[pathLen-3];
//                addFeature(ancestors, feats, featureIndexer, addNew);
//            }
        }

        else
        {
            // otherwise try all subtrees
            List<AnchoredTree<String>> children = subtree.getChildren();
            int num_children = children.size();
            for(int i=0; i< num_children; i++) {
                printPathsRecur(subtree.getChildren().get(i), path, pathLen, feats, featureIndexer, addNew);
            }
        }
    }

    void printPaths(AnchoredTree<String> subtree, List<Integer> feats, Indexer<String> featureIndexer, boolean addNew)
    {
        String[] path = new String[1000];
        printPathsRecur(subtree, path, 0, feats, featureIndexer, addNew);
    }

    private void addFeature(String feat, List<Integer> feats, Indexer<String> featureIndexer, boolean addNew) {
        if (addNew || featureIndexer.contains(feat)) {
            feats.add(featureIndexer.addAndGetIndex(feat));
        }
    }
}
