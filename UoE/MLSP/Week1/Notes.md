
# Linear Discriminant Functions

**Linear discriminant functions**, also known as **linear classifiers**, are mathematical functions used in classification tasks to separate data points from different classes based on their feature values. These functions **model a linear decision boundary or hyperplane that separates the classes in the feature space**.

Here's an overview of linear discriminant functions:

1.  **Representation**: A linear discriminant function is typically represented as a linear combination of the input features. It takes the form: $f(x) = w^T x + b$ where $f(x$) represents the predicted class label for a given input vector $x, w$ is the weight vector, and $b$ is the bias or intercept term.
    
2.  **Decision Boundary**: The linear discriminant function defines a decision boundary or hyperplane in the feature space. The decision boundary is the locus of points where the function output changes, determining the classification of the input points. In a binary classification problem, the decision boundary separates the data points of one class from the other class.   

**Linear discriminant functions assume that the decision boundaries between classes are linear.** While they may not capture complex relationships present in the data, linear classifiers can be effective when classes are separable by linear decision boundaries or in situations where interpretability and simplicity are important.

Overall, linear discriminant functions play a crucial role in classification tasks by separating data points based on linear decision boundaries or hyperplanes, enabling the prediction of class labels for unseen data.