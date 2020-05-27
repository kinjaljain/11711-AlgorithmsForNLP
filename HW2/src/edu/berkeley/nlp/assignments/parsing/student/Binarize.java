package edu.berkeley.nlp.assignments.parsing.student;

import java.util.*;

import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.ling.Trees;
import edu.berkeley.nlp.util.Filter;

public class Binarize
{
    public static Tree<String> annotateTree(Tree<String> unAnnotatedTree)
    {
        return binarizeTree(unAnnotatedTree);
    }

    public static Tree<String> unAnnotateTree(Tree<String> annotatedTree)
    {
        Tree<String> debinarizedTree = Trees.spliceNodes(annotatedTree, new Filter<String>()
        {
            public boolean accept(String s) {
                return s.startsWith("@");
            }
        });
        Tree<String> unAnnotatedTree = (new Trees.LabelNormalizer()).transformTree(debinarizedTree);
        return unAnnotatedTree;
    }

    private static Tree<String> binarizeTree(Tree<String> tree)
    {
        String label = tree.getLabel();
        if (tree.isLeaf()) return new Tree<String>(label);
//        for v=1, don't call the function below
        parentAnnotation(tree);
        if (tree.getChildren().size() == 1) {
            return new Tree<String>(label, Collections.singletonList(binarizeTree(tree.getChildren().get(0))));
        }
        if (tree.getChildren().size() == 2) {

            List<Tree<String>> children = new ArrayList<Tree<String>>();
            children.add(binarizeTree(tree.getChildren().get(0)));
            children.add(binarizeTree(tree.getChildren().get(1)));
            return new Tree<String>(label, children);
        }
        String newLabel = "@" + label + "->_";
        Tree<String> currentTree = binarizeTreeHelper(tree, 0, newLabel);
        return new Tree<String>(label, currentTree.getChildren());
    }

    // binarizes the tree by growing right side (left handed language)
    private static Tree<String> binarizeTreeHelper(Tree<String> tree, int numChildrenGenerated, String label)
    {
        Tree<String> leftTree = tree.getChildren().get(numChildrenGenerated); //0th
        List<Tree<String>> children = new ArrayList<Tree<String>>();
        children.add(binarizeTree(leftTree));
        if (numChildrenGenerated < tree.getChildren().size() - 1) {
            String newLabel = markovization(label, leftTree.getLabel());
            Tree<String> rightTree = binarizeTreeHelper(tree, numChildrenGenerated + 1, newLabel); //rest
            children.add(rightTree);
        }
        return new Tree<String>(label, children);
    }

    // vertical parent annotation for ancestor history
    private static void parentAnnotation(Tree<String> tree)
    {
//        v=2;
        String label = tree.getLabel();
        String parentLabel = label.split("\\^")[0];
        parentLabel = parentLabel.replace("@", "");
        String childLabel;
        for (Tree<String> child : tree.getChildren()) {
            if (child.isPreTerminal() || child.isPhrasal()) {
                childLabel = child.getLabel().concat("^" + parentLabel);
                child.setLabel(childLabel);
            }
        }
    }

    // horizontal sibling markovization for left sibling history
    private static String markovization(String label, String siblingLabel)
    {
        // change h to restrict the sibling context history
        int h = 2;
        siblingLabel = siblingLabel.split("\\^")[0].trim();
        String[] label_ = label.split("_");
        StringBuilder trueLabel = new StringBuilder();
        if(label_.length < 1)
            return trueLabel.toString();
        if(label_.length < h){
            trueLabel = new StringBuilder(label  + siblingLabel);
        }
        else{
            trueLabel = new StringBuilder(label_[0] + "_");
            for(int i=h; i>1; i--){
                trueLabel.append(label_[label_.length - i + 1]);
                trueLabel.append("_");
            }
            trueLabel.append(siblingLabel);
        }
        return trueLabel.toString();
    }
}
